from pricing_opt.base import BasePricingOptimiser
from pricing_opt.utils import _precompute
import numpy as np
import cvxpy as cp

try:
    import scipy.sparse as sp
except Exception:
    sp = None


class New_DeterministicPricingOptimiser(BasePricingOptimiser):
    """
    Stable deterministic optimiser (CCP on (r_d - mu_d)^2) with:
      • Selector matrices (no fragile advanced indexing in CVXPY)
      • Exact log1p/sqrt under MOSEK or SCS; ECOS uses secant hypographs
      • Optional trust region around the CCP anchor
      • Solver-specific routing for parameters
    Returns: r_star, FB0_star, prob, delta_real_star, prev_rates, BPO, T, M
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        product_index,
        reverse_index,
        F, CF, CR, C0,
        prev_balances,
        mean_deltas,
        std_deltas,
        prev_rates,
        rate_mean,
        scale_factor: float = 1e9,
    ):
        # ---- precompute maps and coefficient arrays ----
        pre = _precompute(
            product_index=product_index,
            reverse_index=reverse_index,
            F=F, CF=CF, CR=CR, C0=C0,
            prev_balances=prev_balances,
            prev_rates=prev_rates,
            scale_factor=scale_factor,
        )
        for k, v in pre.items():
            setattr(self, k, v)

        # vector params
        self.mean_deltas = np.asarray(mean_deltas, float).reshape(self.p)
        self.std_deltas  = np.asarray(std_deltas,  float).reshape(self.p)
        self.rate_mean   = np.asarray(rate_mean,  float).reshape(self.p)

        # ---- sanitise indices & build selectors (no advanced indexing in CVXPY) ----
        self.p        = int(self.p)
        self.dest_map = np.asarray(self.dest_map, dtype=np.int64).reshape(-1).copy()
        self.idx_int  = np.asarray(self.idx_int,  dtype=np.int64).reshape(-1).copy()
        self.is_int   = np.asarray(self.is_int,   dtype=bool).reshape(-1).copy()

        if self.dest_map.size != self.p * self.p:
            raise ValueError(f"dest_map length {self.dest_map.size} != p*p={self.p*self.p}")
        if self.dest_map.min() < 0 or self.dest_map.max() >= self.p:
            raise ValueError("dest_map out of range")
        if self.idx_int.size and (self.idx_int.min() < 0 or self.idx_int.max() >= self.p):
            raise ValueError("idx_int out of range")

        I = int(self.is_int.sum())
        N = self.p * self.p
        idx_ext = np.setdiff1d(np.arange(self.p, dtype=np.int64), self.idx_int, assume_unique=True)

        def onehot(rows, cols, shape):
            if sp is not None:
                return sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), cols)), shape=shape)
            M = np.zeros(shape)
            if len(rows):
                M[np.arange(len(rows)), cols] = 1.0
            return M

        # Selectors
        self.S_int  = onehot(np.arange(I), self.idx_int,        (I, self.p))  # pick internal rates
        self.S_ext  = onehot(np.arange(self.p - I), idx_ext,    (self.p - I, self.p))  # pick external rates
        self.G_dest = onehot(np.arange(N), self.dest_map,       (N, self.p))  # gather r at flow-level dest
        self.mask_int_vec = self.is_int.astype(float)  # 1/0 vector length p

        # cache for convenience
        self._I = I
        self._N = N

    # ---- numpy helper for features at the CCP anchor ----
    def _feats_numpy(self, r_curr: np.ndarray):
        y0 = r_curr[self.dest_map]                          # flow-level rates
        dr = y0 - self.prev_rates[self.dest_map]            # Δrate vs prev
        xg = y0 - y0.mean()                                 # centred per-flow
        return y0, dr, xg

    # ---- solver routing (avoid passing wrong params to the wrong solver) ----
    @staticmethod
    def _route_solve(prob: cp.Problem, solver: str, verbose: bool, **kw):
        s = str(solver).upper() if isinstance(solver, str) else str(solver)
        if "MOSEK" in s:
            mosek_params = kw.pop("mosek_params", None)
            if mosek_params is None:
                mosek_params = {
                    "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
                    "MSK_DPAR_INTPNT_TOL_PFEAS":   1e-8,
                    "MSK_DPAR_INTPNT_TOL_DFEAS":   1e-8,
                    "MSK_IPAR_NUM_THREADS":        4,
                    "MSK_IPAR_LOG":                1,
                }
            return prob.solve(solver="MOSEK", verbose=verbose, mosek_params=mosek_params)

        if "ECOS" in s:
            ecos_params = {
                "max_iters": 200000,
                "abstol":    1e-8,
                "reltol":    1e-8,
                "feastol":   1e-8,
            }
            ecos_params.update(kw)
            return prob.solve(solver="ECOS", verbose=verbose, **ecos_params)

        if "SCS" in s:
            scs_params = {
                "eps": 1e-6,
                "max_iters": 50000,
                "acceleration_lookback": 20,
                "scale": 1.0,
                "use_indirect": False,
                "verbose": verbose,
            }
            scs_params.update(kw)
            return prob.solve(solver="SCS", **scs_params)

        # fallback
        return prob.solve(solver=solver, verbose=verbose, **kw)

    # ---- main solve ----
    def solve(
        self,
        *,
        lower_rate: float = 0.05,
        upper_rate: float = 5.0,
        mass_guard: float | None = None,
        max_iter: int = 15,
        tol: float = 1e-5,
        solver: str = "MOSEK",           # MOSEK-first (exact log1p/sqrt)
        verbose: bool = False,
        tr_radius: float | None = 0.3,   # trust region on internal rates (None to disable)
        **solver_params
    ):
        p, N, I = self.p, self._N, self._I
        mu      = self.rate_mean
        scale_f = self.scale_factor

        # decision variables
        r      = cp.Variable(p, name="rate")
        BPO    = cp.Variable(N, name="BPO")
        T      = cp.Variable(N, name="T")
        M      = cp.Variable(N, name="M")
        t_log  = cp.Variable(N, name="t_log")
        m_sqrt = cp.Variable(N, name="m_sqrt")

        # selector constants
        S_int_c  = cp.Constant(self.S_int)    # (I, p)
        S_ext_c  = cp.Constant(self.S_ext)    # (p-I, p)
        G_dest_c = cp.Constant(self.G_dest)   # (N, p)

        # fixed feasibility (no advanced indexing)
        cons = [
            S_ext_c @ r == S_ext_c @ cp.Constant(self.prev_rates),  # freeze external rates
            S_int_c @ r >= lower_rate * np.ones(I),
            S_int_c @ r <= upper_rate * np.ones(I),
            BPO >= 0, BPO <= 0.01,
            T   >= 0, T   <= 1.0,
            M   >= 0, M   <= 1.0,
            t_log >= 0, m_sqrt >= 0,
        ]

        # exact atoms if solver supports them; else piecewise-linear hypographs
        use_exact = ("MOSEK" in str(solver).upper()) or ("SCS" in str(solver).upper())
        if use_exact:
            cons += [t_log <= cp.log1p(T), m_sqrt <= cp.sqrt(M)]
        else:
            # ECOS path: secant hypographs
            def hypograph_secants(x, y, knots, f):
                out = []
                for x0, x1 in zip(knots[:-1], knots[1:]):
                    y0, y1 = f(x0), f(x1)
                    beta = (y1 - y0) / (x1 - x0)
                    alpha = y0 - beta * x0
                    out.append(y <= alpha + beta * x)
                return out
            knots_T = np.concatenate([np.linspace(0.0, 0.2, 6),
                                      np.linspace(0.2, 1.0, 9)[1:]])
            knots_M = np.linspace(0.0, 1.0, 13)
            cons += hypograph_secants(T, t_log,  knots_T, np.log1p)
            cons += hypograph_secants(M, m_sqrt, knots_M, np.sqrt)

        # CCP initial anchor
        r_curr = self.prev_rates.copy()
        r_curr[self.is_int] = np.clip(r_curr[self.is_int], lower_rate, upper_rate)

        best_obj = -np.inf
        best_sol = None
        prob = None

        for it in range(max_iter):
            # --- features at current linearisation point (NumPy only) ---
            y0_np, dr_np, xg_np = self._feats_numpy(r_curr)
            mu_d   = mu[self.dest_map]
            grad   = 2.0 * (y0_np - mu_d)  # (N,)

            # --- linearised square term: (r_d - mu_d)^2 ≈ const + grad*(r_d - mu_d) ---
            r_d_cp   = G_dest_c @ r
            const_t  = (y0_np - mu_d)**2 - grad * (y0_np - mu_d)
            rcsq_aff = const_t + cp.multiply(grad, r_d_cp - mu_d)

            # --- flow predictor ---
            Pflat = (
                self.intercept_adj_flat
                + cp.multiply(self.coef_rcsq, rcsq_aff)
                + cp.multiply(self.coef_xgap,  xg_np)
                + cp.multiply(self.coef_drate, dr_np)
                + cp.multiply(self.coef_promo, BPO)
                + cp.multiply(self.coef_logT,  t_log)
                + cp.multiply(self.coef_sqrt,  m_sqrt)
            )
            P = cp.reshape(Pflat, (p, p), order="C")

            # --- balances ---
            delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)     # (p,)
            delta_std    = delta_scaled / scale_f
            delta_real   = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
            FB0          = self.prev_balances + delta_real

            # --- per-iteration constraints (trust region + optional mass guard) ---
            cons_iter = list(cons)
            if tr_radius is not None and I > 0:
                cons_iter.append(cp.norm_inf(S_int_c @ (r - r_curr)) <= tr_radius)
            if mass_guard is not None and I > 0:
                cons_iter.append(cp.norm1(S_int_c @ delta_std) <= mass_guard)

            # --- objective: maximise internal balances ---
            obj  = cp.Maximize(cp.sum(S_int_c @ FB0))
            prob = cp.Problem(obj, cons_iter)

            # solve with proper routing/params
            val = self._route_solve(prob, solver, verbose, **solver_params)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                if verbose:
                    print(f"[iter {it}] status={prob.status}")
                break

            r_next = r.value.copy()
            improved = (val is not None) and (val > best_obj + tol)
            if improved:
                best_obj = val
                best_sol = (r_next, FB0.value.copy(), delta_real.value, BPO.value, T.value, M.value)
                r_curr = r_next
            else:
                if verbose:
                    print(f"[iter {it}] converged (no improvement)")
                break

        # fallback if no improvement recorded
        if best_sol is None:
            best_sol = (r_curr, None, None, None, None, None)

        r_star, FB0_star, delta_real_star, BPO_star, T_star, M_star = best_sol
        return r_star, FB0_star, prob, delta_real_star, self.prev_rates, BPO_star, T_star, M_star


# keep registry behaviour
BasePricingOptimiser.register(New_DeterministicPricingOptimiser, "new_de")


class DeterministicPricingOptimiserEC(BasePricingOptimiser):
    """
    Stable deterministic optimiser (CCP on (r_d - mu_d)^2) with:
      • Selector matrices (no fragile advanced indexing in CVXPY)
      • Exact log1p/sqrt under MOSEK or SCS; ECOS uses secant hypographs
      • Optional trust region around the CCP anchor
      • Solver-specific routing for parameters
    Returns: r_star, FB0_star, prob, delta_real_star, prev_rates, BPO, T, M
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        product_index,
        reverse_index,
        F, CF, CR, C0,
        prev_balances,
        mean_deltas,
        std_deltas,
        prev_rates,
        rate_mean,
        scale_factor: float = 1e9,
    ):
        # ---- precompute maps and coefficient arrays ----
        pre = _precompute(
            product_index=product_index,
            reverse_index=reverse_index,
            F=F, CF=CF, CR=CR, C0=C0,
            prev_balances=prev_balances,
            prev_rates=prev_rates,
            scale_factor=scale_factor,
        )
        for k, v in pre.items():
            setattr(self, k, v)

        # vector params
        self.mean_deltas = np.asarray(mean_deltas, float).reshape(self.p)
        self.std_deltas  = np.asarray(std_deltas,  float).reshape(self.p)
        self.rate_mean   = np.asarray(rate_mean,  float).reshape(self.p)

        # ---- sanitise indices & build selectors (no advanced indexing in CVXPY) ----
        self.p        = int(self.p)
        self.dest_map = np.asarray(self.dest_map, dtype=np.int64).reshape(-1).copy()
        self.idx_int  = np.asarray(self.idx_int,  dtype=np.int64).reshape(-1).copy()
        self.is_int   = np.asarray(self.is_int,   dtype=bool).reshape(-1).copy()

        if self.dest_map.size != self.p * self.p:
            raise ValueError(f"dest_map length {self.dest_map.size} != p*p={self.p*self.p}")
        if self.dest_map.min() < 0 or self.dest_map.max() >= self.p:
            raise ValueError("dest_map out of range")
        if self.idx_int.size and (self.idx_int.min() < 0 or self.idx_int.max() >= self.p):
            raise ValueError("idx_int out of range")

        I = int(self.is_int.sum())
        N = self.p * self.p
        idx_ext = np.setdiff1d(np.arange(self.p, dtype=np.int64), self.idx_int, assume_unique=True)

        def onehot(rows, cols, shape):
            if sp is not None:
                return sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), cols)), shape=shape)
            M = np.zeros(shape)
            if len(rows):
                M[np.arange(len(rows)), cols] = 1.0
            return M

        # Selectors
        self.S_int  = onehot(np.arange(I), self.idx_int,        (I, self.p))  # pick internal rates
        self.S_ext  = onehot(np.arange(self.p - I), idx_ext,    (self.p - I, self.p))  # pick external rates
        self.G_dest = onehot(np.arange(N), self.dest_map,       (N, self.p))  # gather r at flow-level dest
        self.mask_int_vec = self.is_int.astype(float)  # 1/0 vector length p

        # cache for convenience
        self._I = I
        self._N = N

    # ---- numpy helper for features at the CCP anchor ----
    def _feats_numpy(self, r_curr: np.ndarray):
        y0 = r_curr[self.dest_map]                          # flow-level rates
        dr = y0 - self.prev_rates[self.dest_map]            # Δrate vs prev
        xg = y0 - y0.mean()                                 # centred per-flow
        return y0, dr, xg

    # ---- solver routing (avoid passing wrong params to the wrong solver) ----
    @staticmethod
    def _route_solve(prob: cp.Problem, solver: str, verbose: bool, **kw):
        s = str(solver).upper() if isinstance(solver, str) else str(solver)
        if "MOSEK" in s:
            mosek_params = kw.pop("mosek_params", None)
            if mosek_params is None:
                mosek_params = {
                    "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
                    "MSK_DPAR_INTPNT_TOL_PFEAS":   1e-8,
                    "MSK_DPAR_INTPNT_TOL_DFEAS":   1e-8,
                    "MSK_IPAR_NUM_THREADS":        4,
                    "MSK_IPAR_LOG":                1,
                }
            return prob.solve(solver="MOSEK", verbose=verbose, mosek_params=mosek_params)

        if "ECOS" in s:
            ecos_params = {
                "max_iters": 200000,
                "abstol":    1e-8,
                "reltol":    1e-8,
                "feastol":   1e-8,
            }
            ecos_params.update(kw)
            return prob.solve(solver="ECOS", verbose=verbose, **ecos_params)

        if "SCS" in s:
            scs_params = {
                "eps": 1e-6,
                "max_iters": 50000,
                "acceleration_lookback": 20,
                "scale": 1.0,
                "use_indirect": False,
                "verbose": verbose,
            }
            scs_params.update(kw)
            return prob.solve(solver="SCS", **scs_params)

        # fallback
        return prob.solve(solver=solver, verbose=verbose, **kw)

    # ---- main solve ----
    def solve(
        self,
        *,
        lower_rate: float = 0.05,
        upper_rate: float = 5.0,
        mass_guard: float | None = None,
        max_iter: int = 15,
        tol: float = 1e-5,
        solver: str = "MOSEK",           # MOSEK-first (exact log1p/sqrt)
        verbose: bool = False,
        tr_radius: float | None = 0.3,   # trust region on internal rates (None to disable)
        **solver_params
    ):
        p, N, I = self.p, self._N, self._I
        mu      = self.rate_mean
        scale_f = self.scale_factor

        # decision variables
        r      = cp.Variable(p, name="rate")
        BPO    = cp.Variable(N, name="BPO")
        T      = cp.Variable(N, name="T")
        M      = cp.Variable(N, name="M")
        t_log  = cp.Variable(N, name="t_log")
        m_sqrt = cp.Variable(N, name="m_sqrt")

        # selector constants
        S_int_c  = cp.Constant(self.S_int)    # (I, p)
        S_ext_c  = cp.Constant(self.S_ext)    # (p-I, p)
        G_dest_c = cp.Constant(self.G_dest)   # (N, p)

        # fixed feasibility (no advanced indexing)
        cons = [
            S_ext_c @ r == S_ext_c @ cp.Constant(self.prev_rates),  # freeze external rates
            S_int_c @ r >= lower_rate * np.ones(I),
            S_int_c @ r <= upper_rate * np.ones(I),
            BPO >= 0, BPO <= 0.01,
            T   >= 0, T   <= 1.0,
            M   >= 0, M   <= 1.0,
            t_log >= 0, m_sqrt >= 0,
        ]

        # exact atoms if solver supports them; else piecewise-linear hypographs
        use_exact = ("MOSEK" in str(solver).upper()) or ("SCS" in str(solver).upper())
        if use_exact:
            cons += [t_log <= cp.log1p(T), m_sqrt <= cp.sqrt(M)]
        else:
            # ECOS path: secant hypographs
            def hypograph_secants(x, y, knots, f):
                out = []
                for x0, x1 in zip(knots[:-1], knots[1:]):
                    y0, y1 = f(x0), f(x1)
                    beta = (y1 - y0) / (x1 - x0)
                    alpha = y0 - beta * x0
                    out.append(y <= alpha + beta * x)
                return out
            knots_T = np.concatenate([np.linspace(0.0, 0.2, 6),
                                      np.linspace(0.2, 1.0, 9)[1:]])
            knots_M = np.linspace(0.0, 1.0, 13)
            cons += hypograph_secants(T, t_log,  knots_T, np.log1p)
            cons += hypograph_secants(M, m_sqrt, knots_M, np.sqrt)

        # CCP initial anchor
        r_curr = self.prev_rates.copy()
        r_curr[self.is_int] = np.clip(r_curr[self.is_int], lower_rate, upper_rate)

        best_obj = -np.inf
        best_sol = None
        prob = None

        for it in range(max_iter):
            # --- features at current linearisation point (NumPy only) ---
            y0_np, dr_np, xg_np = self._feats_numpy(r_curr)
            mu_d   = mu[self.dest_map]
            grad   = 2.0 * (y0_np - mu_d)  # (N,)

            # --- linearised square term: (r_d - mu_d)^2 ≈ const + grad*(r_d - mu_d) ---
            r_d_cp   = G_dest_c @ r
            const_t  = (y0_np - mu_d)**2 - grad * (y0_np - mu_d)
            rcsq_aff = const_t + cp.multiply(grad, r_d_cp - mu_d)

            if hasattr(self, "coef_rate_lin"):
                # --- linear mode (for D3 injection) ---
                r_d     = r[self.dest_map]
                mean_rd = cp.sum(r_d) / float(self._N)
                xg_aff  = r_d - mean_rd

                Pflat = (
                    cp.Constant(self.intercept_flat)
                + cp.multiply(cp.Constant(self.coef_rate_lin),  r_d)
                + cp.multiply(cp.Constant(self.coef_xgap_flat), xg_aff)
                )
            else:
                # --- original predictor (D2, full nonlinear) ---
                Pflat = (
                    self.intercept_adj_flat
                    + cp.multiply(self.coef_rcsq, rcsq_aff)
                    + cp.multiply(self.coef_xgap,  xg_np)
                    + cp.multiply(self.coef_drate, dr_np)
                    + cp.multiply(self.coef_promo, BPO)
                    + cp.multiply(self.coef_logT,  t_log)
                    + cp.multiply(self.coef_sqrt,  m_sqrt)
                )

            P = cp.reshape(Pflat, (p, p), order="C")

            # --- balances ---
            delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)     # (p,)
            delta_std    = delta_scaled / scale_f
            delta_real   = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
            FB0          = self.prev_balances + delta_real

            # --- per-iteration constraints (trust region + optional mass guard) ---
            cons_iter = list(cons)
            if tr_radius is not None and I > 0:
                cons_iter.append(cp.norm_inf(S_int_c @ (r - r_curr)) <= tr_radius)
            if mass_guard is not None and I > 0:
                cons_iter.append(cp.norm1(S_int_c @ delta_std) <= mass_guard)

            # --- objective: maximise internal balances ---
            obj  = cp.Maximize(cp.sum(S_int_c @ FB0))
            prob = cp.Problem(obj, cons_iter)

            # solve with proper routing/params
            val = self._route_solve(prob, solver, verbose, **solver_params)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                if verbose:
                    print(f"[iter {it}] status={prob.status}")
                break

            r_next = r.value.copy()
            improved = (val is not None) and (val > best_obj + tol)
            if improved:
                best_obj = val
                best_sol = (r_next, FB0.value.copy(), delta_real.value, BPO.value, T.value, M.value)
                r_curr = r_next
            else:
                if verbose:
                    print(f"[iter {it}] converged (no improvement)")
                break

        # fallback if no improvement recorded
        if best_sol is None:
            best_sol = (r_curr, None, None, None, None, None)

        r_star, FB0_star, delta_real_star, BPO_star, T_star, M_star = best_sol
        return r_star, FB0_star, prob, delta_real_star, self.prev_rates, BPO_star, T_star, M_star


# keep registry behaviour
BasePricingOptimiser.register(DeterministicPricingOptimiserEC, "deterministic")
