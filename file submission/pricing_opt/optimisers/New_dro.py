import numpy as np
import cvxpy as cp
from pricing_opt.base import BasePricingOptimiser
from pricing_opt.optimisers.New_deter import New_DeterministicPricingOptimiser


class DistributionallyRobustPricingOptimiserRC_V4(New_DeterministicPricingOptimiser):
    """
    Linear 'rc' spec (no CCP, no squared term):
      Pflat = intercept
            + coef_ratec * (r_d - mu_d)
            + coef_xgap  * (r_d - mean_r)
            + coef_drate * (r_d - prev_d)
            + coef_promo * BPO

    Objective (baseline-aligned with deterministic):
      Maximize sum_{i in internal} FB0_i  -  rho * || L @ ((r - r_ref)/rate_unit) ||_2

    This version fixes ALL boolean/int indexing issues and uses a mask-vector
    for selecting internal accounts in objectives and constraints.
    """

    name = "dro-linear-rc-v4"

    def __init__(
        self,
        *args,
        Sigma: np.ndarray,
        rho: float = 10.0,
        rate_unit: float | None = None,
        r_ref: np.ndarray | None = None,
        normalize_L: bool = True,
        feature_scale: float = 1.0,
        sigma_units: str = "real",        # {"real","std","legacy"}
        rates_in_decimals: bool = True,   # assert rates like 0.02 for 2%
        eps_psd: float = 1e-10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if sigma_units not in {"real", "std", "legacy"}:
            raise ValueError("sigma_units must be one of {'real','std','legacy'}")
        self.sigma_units = sigma_units

        # --- PSD-safe Σ^{1/2} on internal block ---
        S = np.asarray(Sigma, float)
        I = int(len(self.idx_int))
        if S.shape == (self.p, self.p):
            S_int = S[np.ix_(self.idx_int if np.asarray(self.idx_int).dtype == bool
                             else np.asarray(self.idx_int, int),
                             self.idx_int if np.asarray(self.idx_int).dtype == bool
                             else np.asarray(self.idx_int, int))]
        elif S.shape == (I, I):
            S_int = S
        else:
            raise ValueError(f"Sigma must be (p,p) or (|I|,|I|); got {S.shape}")

        S_int = 0.5 * (S_int + S_int.T)
        w, V = np.linalg.eigh(S_int)
        w = np.clip(w, eps_psd, None)
        self.Sigma_sqrt = V @ np.diag(np.sqrt(w)) @ V.T

        # --- scaling & baseline for rates ---
        if rate_unit is None:
            m = float(np.nanmedian(np.abs(self.prev_rates)))
            rate_unit = 0.01 if m <= 1.0 else 1.0   # decimals vs percent
        self.rate_unit = float(rate_unit)

        self.r_ref = np.asarray(self.prev_rates if r_ref is None else r_ref, float)
        if self.r_ref.shape != (self.p,):
            raise ValueError("r_ref must have shape (p,)")

        self.normalize_L  = bool(normalize_L)
        self.rho          = float(rho)
        self.feature_scale = float(feature_scale)

        self._A_int = None   # (|I|, p)
        self._L_mat = None   # (|I|, p)

        self._assert_finite_all()
        self._assert_rate_units_consistent(rates_in_decimals)

        print("[DRO V4] class loaded. FB0-baseline objective; mask-based indexing; no duplicate solves.")

    # ---------- derivative A_int for the *linear* spec ----------
    def _compute_A_int(self) -> np.ndarray:
        """
        Returns A_int = d(Delta_real)/dr  (shape: |I| x p), i.e. derivative of *real* net delta.
        """
        p        = int(self.p)
        idx_raw  = np.asarray(self.idx_int)
        rows     = np.where(idx_raw)[0] if idx_raw.dtype == bool else idx_raw.astype(int)

        dest_map = np.asarray(self.dest_map, dtype=int)
        s        = self.feature_scale

        base_w = (self.coef_ratec + self.coef_xgap + self.coef_drate)  # (p*p,)
        D = (dest_map[:, None] == np.arange(p)[None, :]).astype(float)  # (p*p, p)

        # dPflat/dr_j = s * [ base_w * 1_{dest=j} - (coef_xgap / p) ]
        dPflat = s * (base_w[:, None] * D - (self.coef_xgap[:, None] / p))

        V3       = dPflat.reshape(p, p, p, order="C")
        col_sum  = V3.sum(axis=0)         # inbound to each dest
        row_sum  = V3.sum(axis=1)         # outbound from each origin
        dvec     = col_sum - row_sum      # (p,p)

        # IMPORTANT: no divide by scale_factor — this is d(Delta_real)/dr
        A_full   = dvec
        return A_full[rows, :]


    def _build_L_mat(self) -> np.ndarray:
        if self._A_int is None:
            self._A_int = self._compute_A_int()

        A_std = self._A_int  # actually d(Delta_real)/dr in our Option A

        if self.sigma_units == "real":
            # Σ is in REAL units already; no extra diag(std) scaling here
            A_eff = A_std
        elif self.sigma_units == "std":
            # If you ever use standardized units again, you’d rescale here.
            A_eff = A_std
        else:  # legacy
            A_eff = A_std

        L_raw = (self.Sigma_sqrt @ A_eff) / self.rate_unit  # (|I|, p)

        if self.normalize_L:
            try:
                opn = float(np.linalg.norm(L_raw, 2))
            except Exception:
                opn = float(np.linalg.svd(L_raw, compute_uv=False)[0])
            opn = max(opn, 1e-12)
            return L_raw / opn
        else:
            return L_raw

    def _ensure_A_and_L(self):
        if self._A_int is None or self._L_mat is None:
            self._A_int = self._compute_A_int()
            self._L_mat = self._build_L_mat()

    def _assert_finite_all(self):
        names = [
            "prev_rates", "rate_mean", "prev_balances",
            "mean_deltas", "std_deltas",
            "coef_ratec", "coef_xgap", "coef_drate", "coef_promo",
            "intercept_adj_flat", "intercept_flat",
        ]
        for nm in names:
            if hasattr(self, nm):
                a = getattr(self, nm)
                if a is None:
                    continue
                a = np.asarray(a)
                if not np.all(np.isfinite(a)):
                    bad = np.where(~np.isfinite(a))
                    raise ValueError(f"{nm} has non-finite entries at {bad}")

    def _assert_rate_units_consistent(self, rates_in_decimals: bool):
        pr = np.asarray(self.prev_rates, float)
        mx = float(np.nanmax(np.abs(pr)))
        if rates_in_decimals:
            if mx > 1.2:
                raise ValueError(
                    "Rates look like percent points, but rates_in_decimals=True. "
                    "Use decimals (e.g., 0.05 for 5%)."
                )
        else:
            if mx <= 1.2:
                raise ValueError(
                    "Rates look like decimals, but rates_in_decimals=False."
                )

    def solve(
        self,
        *,
        lower_rate: float = 0.005,
        upper_rate: float = 0.05,
        mass_guard: float | None = None,
        mass_guard_units: str = "std",   # {"std","real"}
        bpo_max: float | None = 0.01,          # per-pair cap
        promo_budget: float | None = None,     # global budget (sum BPO)
        solver: str | None = "ECOS",
        verbose: bool = False,
        **kw
    ):
        print("[DRO V4] solve() entered.")
        self._assert_finite_all()
        self._ensure_A_and_L()

        p        = int(self.p)
        idx_raw  = np.asarray(self.idx_int)

        # --- build 0/1 mask for internal accounts ---
        if idx_raw.dtype == bool:
            mask = idx_raw.astype(float)
            rows = np.where(idx_raw)[0]
        else:
            rows = idx_raw.astype(int)
            mask = np.zeros(p, dtype=float)
            mask[rows] = 1.0
        MASK = cp.Constant(mask)
        print(f"[DRO V4] internal_count={int(mask.sum())}")

        dest_map = np.asarray(self.dest_map, dtype=int)

        # ----- decision variables -----
        n_pairs = p * p
        r   = cp.Variable(p,       name="rate")
        BPO = cp.Variable(n_pairs, name="BPO")
                # Decision vars (add back)
        T      = cp.Variable(p*p, name="T")
        M      = cp.Variable(p*p, name="M")
        t_log  = cp.Variable(p*p, name="t_log")
        m_sqrt = cp.Variable(p*p, name="m_sqrt")



        # ----- feasibility -----
        cons = [
            r[~self.is_int] == self.prev_rates[~self.is_int],
            r[self.is_int]  >= lower_rate,
            r[self.is_int]  <= upper_rate,
            BPO >= 0,
        ]
                # Feasibility (bounds + epigraphs)
        cons += [
            T >= 0, T <= 1,
            M >= 0, M <= 1,
            t_log  >= 0, t_log  <= cp.log1p(T),
            m_sqrt >= 0, m_sqrt <= cp.sqrt(M),
]
        if bpo_max is not None:
            cons.append(BPO <= float(bpo_max))
        if promo_budget is not None:
            cons.append(cp.sum(BPO) <= float(promo_budget))

        # Prev-rate sanity (unit mismatch detector)
        prev_int = np.asarray(self.prev_rates)[self.is_int]
        frac_oob = np.mean((prev_int < lower_rate) | (prev_int > upper_rate))
        if frac_oob > 0.5:
            raise ValueError(
                f"More than half of prev internal rates lie outside [{lower_rate}, {upper_rate}]. "
                "Check rate units."
            )

        # ----- features (AFFINE in r) -----
        r_d     = r[dest_map]
        mu_d    = self.rate_mean[dest_map]
        prev_d  = self.prev_rates[dest_map]
        mean_r  = cp.sum(r) / p

        ratec_v = (r_d - mu_d)   * self.feature_scale
        xgap_v  = (r_d - mean_r) * self.feature_scale
        drate_v = (r_d - prev_d) * self.feature_scale

        intercept = getattr(self, "intercept_adj_flat", None)
        if intercept is None:
            intercept = getattr(self, "intercept_flat", None)
        if intercept is None:
            raise ValueError("Missing intercept: neither intercept_adj_flat nor intercept_flat is set.")

        # ----- flow predictions -----
# ----- flow predictions -----
        Pflat = (
            intercept
            + cp.multiply(self.coef_ratec, ratec_v)
            + cp.multiply(self.coef_xgap,  xgap_v)
            + cp.multiply(self.coef_drate, drate_v)
            + cp.multiply(self.coef_promo, BPO)
            # + cp.multiply(self.coef_logT,  t_log)   # add these back if you reintroduce promo shape features
            # + cp.multiply(self.coef_sqrt,  m_sqrt)
        )
        P          = cp.reshape(Pflat, (p, p), order="C")

        # Net flow in REAL units (no standardization)
        delta_s    = cp.sum(P, axis=0) - cp.sum(P, axis=1)   # (p,)
        delta_real = delta_s                                  # <-- KEY: Option A

        FB0        = self.prev_balances + delta_real

        # Optional guard; support both units for convenience
        if mass_guard is not None:
            if mass_guard_units == "std":
                delta_std_guard = delta_real / float(self.scale_factor)
                cons.append(cp.norm1(cp.multiply(MASK, delta_std_guard)) <= float(mass_guard))
            else:
                cons.append(cp.norm1(cp.multiply(MASK, delta_real)) <= float(mass_guard))

        # ----- DRO penalty (using d(Delta_real)/dr) -----
        L_const = cp.Constant(self._L_mat) if (self._L_mat is not None) else None
        r_dev   = (r - self.r_ref) / self.rate_unit
        dro_pen = 0.0
        if self.rho > 0.0 and L_const is not None:
            dro_pen = self.rho * cp.norm(L_const @ r_dev, 2)

        # ----- objective: baseline aligned -----
        obj  = cp.Maximize(MASK @ FB0 - (dro_pen if isinstance(dro_pen, cp.Expression) else 0.0))
        prob = cp.Problem(obj, cons)

        for s in ([solver] if solver else ["ECOS", "SCS"]):
            try:
                prob.solve(solver=s, verbose=verbose, **kw)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue

        # ---- diagnostics ----
        try:
            val_FB0   = float((MASK @ FB0).value)
            C_const   = float(np.dot(mask, self.prev_balances))
            val_delta = float(np.dot(mask, delta_real.value))
            print(f"[DRO V4] prob.value     = {prob.value:.6e}")
            print(f"[DRO V4] MASK@FB0       = {val_FB0:.6e}")
            print(f"[DRO V4] C_const        = {C_const:.6e}")
            print(f"[DRO V4] C_const+delta  = {(C_const + val_delta):.6e}")
            if isinstance(dro_pen, cp.Expression):
                print(f"[DRO V4] dro_pen       = {float(dro_pen.value):.6e}")
        except Exception:
            pass

        r_star   = None if r.value   is None else r.value.copy()
        FB_star  = None if FB0.value is None else FB0.value.copy()
        d_star   = None if delta_real.value is None else delta_real.value.copy()
        dro_val  = None if (isinstance(dro_pen, cp.Expression) and dro_pen.value is not None) else 0.0

        return r_star, FB_star, prob, d_star, self.prev_rates, dro_val


# Register under a NEW key to avoid stale binding
BasePricingOptimiser.register(DistributionallyRobustPricingOptimiserRC_V4, "dro-linear-rc-v4")
