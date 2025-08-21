from pricing_opt.base import BasePricingOptimiser
from pricing_opt.utils import build_deter_flow_model, _precompute
from pricing_opt.data_types import OptimiserReturn
import cvxpy as cp
import numpy as np



import numpy as np
import cvxpy as cp

class DeterministicPricingOptimiser(BasePricingOptimiser):
    name = "deterministic"

    def __init__(
        self,
        *,
        product_index,
        reverse_index,
        F,
        CF,
        CR,
        C0,
        prev_balances,
        mean_deltas,
        std_deltas,
        prev_rates,
        rate_mean,
        scale_factor=1e9
    ):
        # Precompute maps and coefficient arrays
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
        # Ensure rate_mean is vector of length p
        self.rate_mean   = np.asarray(rate_mean, dtype=float).reshape(self.p)

    def solve(
        self,
        *,
        lower_rate: float = 0.05,
        upper_rate: float = 5.0,
        mass_guard: float | None = None,
        max_iter: int = 15,
        tol: float = 1e-5,
        solver: str = "ECOS",
        verbose: bool = False,
        **kw
    ):
        """
        CCP majorization on the (r - rate_mean)^2 term.
        Returns: r_star, FB0_star, prob, delta_real_star, prev_rates
        """
        p        = self.p
        idx_int  = self.idx_int
        dest_map = self.dest_map
        # rate_mean is an array of shape (p,)
        mu       = self.rate_mean
        scale_f  = self.scale_factor

        # Decision variables
        r   = cp.Variable(p, name="rate")
        BPO = cp.Variable(p*p, name="BPO")
        T   = cp.Variable(p*p, name="T")
        M   = cp.Variable(p*p, name="M")
        t_log  = cp.Variable(BPO.shape, name="t_log")
        m_sqrt = cp.Variable(BPO.shape, name="m_sqrt")
        # Fixed feasibility constraints
        cons_fixed = [
            r[~self.is_int] == self.prev_rates[~self.is_int],
            r[self.is_int]  >= lower_rate,
            r[self.is_int]  <= upper_rate,
            BPO >= 0, BPO <= 0.01,
            T   >= 0, T   <= 1.0,
            M   >= 0, M   <= 1.0,
            t_log <= cp.log(1 + T),  t_log >= 0,
            m_sqrt <= cp.sqrt(M),    m_sqrt >= 0
        ]

        # helper: compute flow-level features from current r
        def compute_feats(r_curr: np.ndarray):
            y0 = r_curr[dest_map]                      # flow-level rates
            dr = r_curr[dest_map] - self.prev_rates[dest_map]
            xg = r_curr[dest_map] - np.mean(y0)
            return y0, dr, xg

        # initialize iterate
        r_curr = self.prev_rates.copy()
        r_curr[self.is_int] = np.clip(r_curr[self.is_int], lower_rate, upper_rate)

        best_obj = -np.inf
        best_sol = None

        for it in range(max_iter):
            y0_np, drate_np, xgap_np = compute_feats(r_curr)

            # Build flow-level mean vector for square term
            mu_d = mu[dest_map]
            # Linearize the convex square term: f(y)=(y-mu)^2
            grad       = 2.0 * (y0_np - mu_d)
            const_term = (y0_np - mu_d)**2 - grad * (y0_np - mu_d)
            r_d_cp     = r[dest_map]
            rcsq_affine = const_term + grad * (r_d_cp - mu_d)

            # Build flow expression
            Pflat = (
                self.intercept_adj_flat
                + cp.multiply(self.coef_rcsq, rcsq_affine)
                + cp.multiply(self.coef_xgap,  xgap_np)
                + cp.multiply(self.coef_drate, drate_np)
                + cp.multiply(self.coef_promo, BPO)
                + cp.multiply(self.coef_logT, t_log)
                + cp.multiply(self.coef_sqrt, m_sqrt)
            )
            P = cp.reshape(Pflat, (p, p), order="C")

            # Balance equations
            delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)
            delta_std    = delta_scaled / scale_f
            delta_real   = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
            FB0          = self.prev_balances + delta_real

            # Assemble constraints
            cons = list(cons_fixed)
            if mass_guard is not None and idx_int.size > 0:
                cons.append(cp.norm(delta_std[idx_int], 1) <= mass_guard)

            # Surrogate objective (concave)
            obj = cp.Maximize(cp.sum(FB0[idx_int]))
            prob = cp.Problem(obj, cons)
            val = prob.solve(solver=solver, verbose=verbose, **kw)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                if verbose:
                    print(f"Iteration {it}: status {prob.status}")
                break

            # Update iterate
            r_next = r.value.copy()
            r_curr = r_next

            # Track best
            if val is not None and val > best_obj + tol:
                best_obj = val
                best_sol = (r.value.copy(), FB0.value.copy(), delta_real.value)
            else:
                if verbose:
                    print(f"Converged at iteration {it}")
                break

        # Fallback
        if best_sol is None:
            best_sol = (r_curr, None, None)

        r_star, FB0_star, delta_real_star = best_sol
        return r_star, FB0_star, prob, delta_real_star, self.prev_rates, BPO, T, M

BasePricingOptimiser.register(DeterministicPricingOptimiser, 'deterministic')

