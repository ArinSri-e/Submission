import numpy as np
import cvxpy as cp
from pricing_opt.base import BasePricingOptimiser
from pricing_opt.utils import _precompute

class CVaRPricingOptimiserEC(BasePricingOptimiser):
    """
    EC-style CVaR optimiser (selector matrices, safe atoms).
    - Drop-in replacement for old CVaRPricingOptimiser
    - Returns: r, FB0, delta_real, penalty, meta
    """

    name = "cvar_ec"

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
        Sigma: np.ndarray,
        alpha: float = 0.95,
        n_scen: int = 500,
        scen_seed: int = 42,
        scale_factor: float = 1e9,
    ):
        # --- pre
        #  (deterministic parts) ---
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

        self.mean_deltas = np.asarray(mean_deltas, float).reshape(self.p)
        self.std_deltas  = np.asarray(std_deltas,  float).reshape(self.p)
        self.rate_mean   = np.asarray(rate_mean,  float).reshape(self.p)

        self.alpha  = float(alpha)
        self.n_scen = int(n_scen)
        rng = np.random.default_rng(scen_seed)
        self.Z = rng.standard_normal(size=(self.n_scen, self.idx_int.size))

        # --- selectors ---
        p = int(self.p)
        I = int(self.is_int.sum())
        N = p * p
        idx_ext = np.setdiff1d(np.arange(p), self.idx_int, assume_unique=True)

        def onehot(rows, cols, shape):
            M = np.zeros(shape)
            if len(rows):
                M[np.arange(len(rows)), cols] = 1.0
            return M

        self.S_int  = onehot(np.arange(I), self.idx_int, (I, p))
        self.S_ext  = onehot(np.arange(p - I), idx_ext, (p - I, p))
        self.G_dest = onehot(np.arange(N), self.dest_map, (N, p))

        # --- Σ^{1/2} ---
        S = np.asarray(Sigma, float)
        if S.shape == (p, p):
            S_int = S[np.ix_(self.idx_int, self.idx_int)]
        elif S.shape == (I, I):
            S_int = S
        else:
            raise ValueError(f"Sigma must be {(p, p)} or {(I, I)}, got {S.shape}")
        self.Sigma_sqrt = np.linalg.cholesky(S_int + 1e-8 * np.eye(I))

        self.scale_factor = float(scale_factor)

    # ---------- solve ----------
    def solve(
        self,
        *,
        mixing: str = "pmr",      # "pmr" or "blend"
        rho: float = 0.0,
        tau: float | None = None,
        r0: np.ndarray | None = None,
        tr_radius: float = 0.3,
        tr_shrink: float = 0.5,
        tr_min: float = 1e-3,
        max_iter: int = 20,
        tol: float = 1e-5,
        solver: str = "MOSEK",
        verbose: bool = False,
        solver_params: dict | None = None,
    ):
        p, I, N = self.p, self.idx_int.size, self.p * self.p
        mu      = self.rate_mean
        scale_f = self.scale_factor

        # variables
        r     = cp.Variable(p, name="rate")
        BPO   = cp.Variable(N, name="BPO")
        T     = cp.Variable(N, name="T")
        M     = cp.Variable(N, name="M")
        t_log = cp.Variable(N, name="t_log")
        m_sqrt= cp.Variable(N, name="m_sqrt")

        S_int_c  = cp.Constant(self.S_int)
        S_ext_c  = cp.Constant(self.S_ext)
        G_dest_c = cp.Constant(self.G_dest)

        # feasibility
        cons_base = [
            S_ext_c @ r == S_ext_c @ cp.Constant(self.prev_rates),
            S_int_c @ r >= 0.05,
            S_int_c @ r <= 5.0,
            BPO >= 0, BPO <= 0.01,
            T   >= 0, T   <= 1.0,
            M   >= 0, M   <= 1.0,
            t_log >= 0, m_sqrt >= 0,
            t_log <= cp.log1p(T),
            m_sqrt <= cp.sqrt(M),
        ]

        # anchor
        r_curr = r0 if r0 is not None else self.prev_rates.copy()
        r_curr[self.is_int] = np.clip(r_curr[self.is_int], 0.05, 5.0)

        best_val, best_sol = -np.inf, None
        one_minus_alpha = max(1e-12, 1.0 - self.alpha)

        for it in range(max_iter):
            # --- linearised square term
            y0_np   = r_curr[self.dest_map]
            mu_d    = mu[self.dest_map]
            grad    = 2.0 * (y0_np - mu_d)
            const_t = (y0_np - mu_d)**2 - grad * (y0_np - mu_d)
            r_d_cp  = G_dest_c @ r
            rcsq_aff = const_t + cp.multiply(grad, r_d_cp - mu_d)

            # --- flow predictor
            Pflat = (
                self.intercept_adj_flat
                + cp.multiply(self.coef_rcsq, rcsq_aff)
                + cp.multiply(self.coef_drate, r_d_cp - self.prev_rates[self.dest_map])
                + cp.multiply(self.coef_drate, r_d_cp - self.prev_rates[self.dest_map])
                + cp.multiply(self.coef_promo, BPO)
                + cp.multiply(self.coef_logT,  t_log)
                + cp.multiply(self.coef_sqrt,  m_sqrt)
            )
            P = cp.reshape(Pflat, (p, p), order="C")

            delta_s    = cp.sum(P, axis=0) - cp.sum(P, axis=1)
            delta_std  = delta_s / scale_f
            delta_real = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
            FB0        = self.prev_balances + delta_real

            # --- CVaR
            A_int = self._compute_A_int(r_curr)
            A_risk = scale_f * A_int
            W_np   = self.Z @ self.Sigma_sqrt @ A_risk
            W      = cp.Constant(W_np)

            z = cp.Variable(self.n_scen, nonneg=True)
            t = cp.Variable()
            u = cp.Variable(self.n_scen, nonneg=True)
            Wr = W @ r
            cons_cvar = [z >= Wr, z >= -Wr, u >= z - t]
            cvar_expr = t + cp.sum(u) / (self.n_scen * one_minus_alpha)

            # constraints this iter
            cons_iter = cons_base + cons_cvar
            cons_iter.append(cp.norm(S_int_c @ (r - r_curr), 2) <= tr_radius)

            # objective
            profit = cp.sum(S_int_c @ FB0)
            if tau is not None:
                cons_iter.append(cvar_expr <= tau)
                obj = cp.Maximize(profit)
            elif mixing == "pmr":
                obj = cp.Maximize(profit - rho * cvar_expr)
            else:  # blend
                obj = cp.Maximize((1 - rho) * profit - rho * cvar_expr)

            prob = cp.Problem(obj, cons_iter)
            val = prob.solve(solver=solver, verbose=verbose, **(solver_params or {}))

            if prob.status not in ("optimal", "optimal_inaccurate"):
                break

            if val is not None and val > best_val + tol:
                best_val = val
                best_sol = (
                    r.value.copy(),
                    FB0.value.copy(),
                    delta_real.value.copy(),
                    float(cvar_expr.value),
                    float(profit.value),
                )
                r_curr = r.value.copy()
            else:
                break  # no improvement

            tr_radius = max(tr_radius * tr_shrink, tr_min)

        if best_sol is None:
            return r_curr, None, None, np.nan, {"status": "failed"}

        r_star, FB0_star, dreal_star, cvar_val, profit_val = best_sol
        meta = {
            "risk_abs": float(cvar_val),
            "profit": float(profit_val),
            "status": prob.status,
            "tr_used": tr_radius,   # record trust region used
        }
        return r_star, FB0_star, dreal_star, cvar_val, meta

    def _compute_A_int(self, r_ref: np.ndarray) -> np.ndarray:
        """
        Jacobian A_int = ∂Δ_std / ∂r  evaluated at r_ref, returning only internal rows.

        Shapes:
        - returns (I, p)
        - I = number of internal products
        """
        p = int(self.p)
        dest_map = np.asarray(self.dest_map, dtype=int)       # length N = p*p
        N = dest_map.size

        # Selector from product-rate vector r (length p) to flow-dest rates r_d (length N)
        D = (dest_map[:, None] == np.arange(p)[None, :]).astype(float)   # (N, p)

        # Gradients for the linearised square term at r_ref
        grad_dest = 2.0 * (np.asarray(r_ref, float) - self.rate_mean)    # (p,)
        grad_flat = grad_dest[dest_map]                                   # (N,)

        # Per-flow base weight for the δ_{dest=k} part
        #   rc-sq contributes: coef_rcsq * grad_flat
        #   xgap contributes:  coef_xgap * 1
        #   drate contributes: coef_drate * 1
        base_w = self.coef_rcsq * grad_flat + self.coef_xgap + self.coef_drate  # (N,)

        # dPflat/dr_k = base_w * D[:,k]  -  (coef_xgap / p)   (from mean(r_d) term)
        dPflat = base_w[:, None] * D - (self.coef_xgap[:, None] / float(p))     # (N, p)

        # Reshape to (p, p, p) and net inflow-outflow to node deltas
        V = dPflat.reshape(p, p, p, order="C")  # dims: [src, dest, k]
        dDelta = V.sum(axis=0) - V.sum(axis=1)  # (p, p): rows=dest, cols=k

        # Standardisation to Δ_std and restrict to internal rows
        A = dDelta / float(self.scale_factor)   # (p, p)
        return A[self.idx_int, :]               # (I, p)


# register
BasePricingOptimiser.register(CVaRPricingOptimiserEC, "cvar_ec")
