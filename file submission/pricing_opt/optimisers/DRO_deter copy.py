from pricing_opt.base import BasePricingOptimiser
from pricing_opt.optimisers.deterministic import DeterministicPricingOptimiser
import cvxpy as cp
import numpy as np


# class DistributionallyRobustPricingOptimiser(DeterministicPricingOptimiser):
#     name = "dro"

#     def __init__(self, *args, Sigma: np.ndarray, rho: float = 100.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         S = np.asarray(Sigma, float)
#         I = int(self.is_int.sum())
#         if S.shape == (self.p, self.p):
#             S_int = S[np.ix_(self.idx_int, self.idx_int)]
#         elif S.shape == (I, I):
#             S_int = S
#         else:
#             raise ValueError(f"Sigma must be {(self.p, self.p)} or {(I, I)}, got {S.shape}")
#         S_int = S_int + 1e-8 * np.eye(I)  # jitter for numerical stability
#         self.Sigma_sqrt = np.linalg.cholesky(S_int)
#         self.rho = float(rho)

#     # ---------- helpers ----------
#     def _feats(self, r_curr: np.ndarray):
#         y0 = r_curr[self.dest_map]                          # flow-level rate
#         dr = y0 - self.prev_rates[self.dest_map]            # delta-rate vs prev (dest-based)
#         xg = y0 - y0.mean()                                 # centered rate per flow
#         return y0, dr, xg

#     def _build_A_int_current(self, grad_flow: np.ndarray) -> np.ndarray:
#         """
#         Build the Jacobian A_int of delta_std wrt r_int at the current CCP iterate.
#         grad_flow has shape (p*p,) and equals 2*(y0 - mu_d) evaluated at current iterate.
#         """
#         p       = self.p
#         idx_int = self.idx_int
#         N       = p * p

#         coef_rcsq = self.coef_rcsq    # (N,)
#         coef_xgap = self.coef_xgap    # (N,)
#         coef_drt  = self.coef_drate   # (N,)
#         dmap      = self.dest_map     # (N,)

#         # Note: xgap = y0 - mean(y0) -> d/d r_j = 1{dest=j} - 1/p  (since each dest has p flows)
#         xgap_mean_deriv = -1.0 / p

#         A = np.zeros((len(idx_int), len(idx_int)), dtype=float)

#         for jj, j in enumerate(idx_int):
#             mask_d = (dmap == j).astype(float)  # 1{dest=j} vector of length N

#             # dP_flat/dr_j at each flow (length N):
#             dP_drj = (
#                 coef_rcsq * grad_flow * mask_d               # rcsq via linearization
#                 + coef_xgap * (mask_d + xgap_mean_deriv)     # xgap: 1{dest=j} - 1/p
#                 + coef_drt  * mask_d                         # drate: 1{dest=j}
#                 # promo/log/sqrt are independent of r
#             )

#             mat  = dP_drj.reshape((p, p), order="C")
#             dvec = mat.sum(axis=0) - mat.sum(axis=1)         # derivative of delta_scaled wrt r_j
#             A[:, jj] = dvec[idx_int] / self.scale_factor     # map to int rows and scale to delta_std

#         return A

#     # ---------- solve ----------
#     def solve(
#         self,
#         *,
#         lower_rate: float = 0.05,
#         upper_rate: float = 5.0,
#         mass_guard: float | None = None,
#         max_iter: int = 15,
#         tol: float = 1e-5,
#         solver: str = "ECOS",
#         verbose: bool = False,
#         **kw
#     ):
#         p        = self.p
#         idx_int  = self.idx_int
#         mu       = self.rate_mean
#         rho      = self.rho

#         # Decision variables (parity with deterministic)
#         r     = cp.Variable(p,   name="rate")
#         BPO   = cp.Variable(p*p, name="BPO")
#         T     = cp.Variable(p*p, name="T")
#         M     = cp.Variable(p*p, name="M")
#         t_log = cp.Variable(p*p, name="t_log")
#         m_sqrt= cp.Variable(p*p, name="m_sqrt")

#         cons_fixed = [
#             r[~self.is_int] == self.prev_rates[~self.is_int],
#             r[self.is_int]  >= lower_rate,
#             r[self.is_int]  <= upper_rate,
#             BPO >= 0, BPO <= 0.01,
#             T   >= 0, T   <= 1.0,
#             M   >= 0, M   <= 1.0,
#             t_log >= 0,   t_log <= cp.log(T),
#             m_sqrt >= 0,  m_sqrt <= cp.sqrt(M),
#         ]

#         # CCP init
#         r_curr = self.prev_rates.copy()
#         r_curr[self.is_int] = np.clip(r_curr[self.is_int], lower_rate, upper_rate)

#         best_obj = -np.inf
#         best_sol = None
#         prob = None

#         for it in range(max_iter):
#             # Features at current linearization point
#             y0_np, dr_np, xg_np = self._feats(r_curr)
#             mu_d   = mu[self.dest_map]
#             grad   = 2.0 * (y0_np - mu_d)  # (N,)

#             # Linearized square term (r_d - mu_d)^2
#             const_t  = (y0_np - mu_d)**2 - grad*(y0_np - mu_d)
#             r_d_cp   = r[self.dest_map]
#             rcsq_aff = const_t + grad*(r_d_cp - mu_d)  # affine in r

#             # Flow predictor (identical to deterministic)
#             Pflat = (
#                 self.intercept_adj_flat
#                 + cp.multiply(self.coef_rcsq, rcsq_aff)
#                 + cp.multiply(self.coef_xgap,  xg_np)
#                 + cp.multiply(self.coef_drate, dr_np)
#                 + cp.multiply(self.coef_promo, BPO)
#                 + cp.multiply(self.coef_logT,  t_log)
#                 + cp.multiply(self.coef_sqrt,  m_sqrt)
#             )
#             P = cp.reshape(Pflat, (p, p), order="C")

#             # Balance equations
#             delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)  # (p,)
#             delta_std    = delta_scaled / self.scale_factor
#             delta_real   = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
#             FB0          = self.prev_balances + delta_real

#             # DRO penalty (rebuild A per-iterate; center and restrict to int)
#             A_np   = self._build_A_int_current(grad)          # (I, I)
#             A_c    = cp.Constant(A_np)
#             S_c    = cp.Constant(self.Sigma_sqrt)
#             r_int  = r[self.is_int]
#             r0_int = self.prev_rates[self.is_int]
#             dro_pen = rho * cp.norm(S_c @ (A_c @ (r_int - r0_int)), 2)

#             # Constraints
#             cons = list(cons_fixed)
#             if mass_guard is not None and idx_int.size > 0:
#                 cons.append(cp.norm(delta_std[idx_int], 1) <= mass_guard)

#             # Objective: deterministic objective minus DRO penalty
#             obj  = cp.Maximize(cp.sum(FB0[idx_int]) - dro_pen)
#             prob = cp.Problem(obj, cons)
#             val  = prob.solve(solver=solver, verbose=verbose, **kw)

#             if prob.status not in ("optimal", "optimal_inaccurate"):
#                 if verbose:
#                     print(f"Iteration {it}: status {prob.status}")
#                 break

#             r_next = r.value.copy()
#             if val is not None and val > best_obj + tol:
#                 best_obj = val
#                 best_sol = (
#                     r_next,
#                     FB0.value.copy(),
#                     delta_real.value,
#                     BPO.value,
#                     T.value,
#                     M.value,
#                     float(dro_pen.value),
#                 )
#                 r_curr = r_next
#             else:
#                 if verbose:
#                     print(f"Converged at iteration {it}")
#                 break

#         if best_sol is None:
#             # Fallback if no improvement was recorded
#             best_sol = (r_curr, None, None, None, None, None, None)

#         r_star, FB0_star, delta_real_star, BPO_star, T_star, M_star, dro_pen_val = best_sol
#         # Return signature aligned with deterministic (+ dro_pen at the end)
#         return r_star, FB0_star, prob, delta_real_star, self.prev_rates, BPO_star, T_star, M_star, dro_pen_val


# BasePricingOptimiser.register(DistributionallyRobustPricingOptimiser, 'dro')


# class DistributionallyRobustPricingOptimiserEC(_DetBase):
#     """
#     Deterministic + risk CAP (τ) on internal balance change.
#     Solves:  max   sum_{j in INT} FB0_j
#              s.t.  || Σ_int^{1/2} · delta_std_int(r,BPO,T,M) ||_2 <= τ
#                    box constraints, promo/log/sqrt hypographs

#     Notes
#     -----
#     - Σ may be provided as p×p (full) or I×I (internal only). We store both the
#       internal block (self.Sigma_int) and its Cholesky (self.Sigma_sqrt).
#     - Uses exact log1p/sqrt. Works well with MOSEK.
#     - Always returns concrete arrays (never None) when a feasible solution is found.
#     """

#     name = "dro_tau"

#     def __init__(self, *args, Sigma: np.ndarray, eig_floor: float = 1e-9, **kwargs):
#         super().__init__(*args, **kwargs)

#         # --- build internal Σ and its sqrt ---
#         S = np.asarray(Sigma, float)
#         I = int(self.is_int.sum())
#         if S.shape == (self.p, self.p):
#             S_int = S[np.ix_(self.idx_int, self.idx_int)]
#         elif S.shape == (I, I):
#             S_int = S
#         else:
#             raise ValueError(f"Sigma must be {(self.p, self.p)} or {(I, I)}, got {S.shape}")

#         # eigen-floor for numerical stability
#         w, V = np.linalg.eigh(S_int)
#         w = np.clip(w, eig_floor, None)
#         self.Sigma_int  = (V * w) @ V.T
#         self.Sigma_sqrt = (V * np.sqrt(w)) @ V.T

#     def solve(
#         self,
#         *,
#         lower_rate: float = 0.05,
#         upper_rate: float = 5.0,
#         mass_guard: float | None = None,
#         max_iter: int = 15,
#         tol: float = 1e-5,
#         solver: str = "MOSEK",
#         verbose: bool = False,
#         tau: float | None = None,         # <-- risk cap on ||Σ^{1/2} Δ_std_int||
#         **solver_params
#     ):
#         p       = self.p
#         mu      = self.rate_mean
#         idx_int = self.idx_int

#         # decision vars
#         r     = cp.Variable(p,   name="rate")
#         BPO   = cp.Variable(p*p, name="BPO")
#         T     = cp.Variable(p*p, name="T")
#         M     = cp.Variable(p*p, name="M")
#         t_log = cp.Variable(p*p, name="t_log")
#         m_sqrt= cp.Variable(p*p, name="m_sqrt")

#         # feasibility/box + exact atoms
#         cons_fixed = [
#             r[~self.is_int] == self.prev_rates[~self.is_int],
#             r[self.is_int]  >= lower_rate,
#             r[self.is_int]  <= upper_rate,
#             BPO >= 0, BPO <= 0.01,
#             T   >= 0, T   <= 1.0,
#             M   >= 0, M   <= 1.0,
#             t_log <= cp.log1p(T),  t_log >= 0,
#             m_sqrt <= cp.sqrt(M),  m_sqrt >= 0,
#         ]

#         # CCP init (same as deterministic)
#         r_curr = self.prev_rates.copy()
#         r_curr[self.is_int] = np.clip(r_curr[self.is_int], lower_rate, upper_rate)

#         best_obj = -np.inf
#         best_sol = None
#         prob     = None

#         S_int_sqrt_c = cp.Constant(self.Sigma_sqrt)

#         for it in range(max_iter):
#             # features at current linearisation point
#             y0_np   = r_curr[self.dest_map]
#             dr_np   = y0_np - self.prev_rates[self.dest_map]
#             xg_np   = y0_np - y0_np.mean()
#             mu_d    = mu[self.dest_map]
#             grad    = 2.0 * (y0_np - mu_d)  # (N,)

#             # linearised (r_d - mu_d)^2
#             const_t   = (y0_np - mu_d)**2 - grad*(y0_np - mu_d)
#             r_d_cp    = r[self.dest_map]
#             rcsq_aff  = const_t + grad*(r_d_cp - mu_d)

#             # predictor
#             Pflat = (
#                 self.intercept_adj_flat
#                 + cp.multiply(self.coef_rcsq, rcsq_aff)
#                 + cp.multiply(self.coef_xgap,  xg_np)
#                 + cp.multiply(self.coef_drate, dr_np)
#                 + cp.multiply(self.coef_promo, BPO)
#                 + cp.multiply(self.coef_logT,  t_log)
#                 + cp.multiply(self.coef_sqrt,  m_sqrt)
#             )
#             P = cp.reshape(Pflat, (p, p), order="C")

#             # balances and deltas
#             delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)   # (p,)
#             delta_std    = delta_scaled / self.scale_factor
#             delta_real   = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
#             FB0          = self.prev_balances + delta_real

#             # constraints for this iterate
#             cons_iter = list(cons_fixed)
#             if mass_guard is not None and idx_int.size > 0:
#                 cons_iter.append(cp.norm(delta_std[idx_int], 1) <= mass_guard)

#             # --- risk CAP on *true* delta_std (internal block) ---
#             if tau is not None and idx_int.size > 0:
#                 cons_iter.append(cp.norm(S_int_sqrt_c @ delta_std[idx_int], 2) <= float(tau))

#             # objective (deterministic)
#             obj = cp.Maximize(cp.sum(FB0[idx_int]))

#             prob = cp.Problem(obj, cons_iter)
#             val  = prob.solve(solver=solver, verbose=verbose, **solver_params)

#             if prob.status not in ("optimal", "optimal_inaccurate"):
#                 if verbose:
#                     print(f"[iter {it}] status={prob.status}")
#                 break

#             r_next = r.value.copy()

#             # stash best feasible
#             if val is not None and val > best_obj + tol:
#                 best_obj = val
#                 best_sol = (
#                     r_next,
#                     FB0.value.copy(),
#                     delta_real.value.copy(),
#                     BPO.value.copy(),
#                     T.value.copy(),
#                     M.value.copy()
#                 )
#                 r_curr = r_next
#             else:
#                 if verbose:
#                     print(f"[iter {it}] converged (no improvement)")
#                 break

#         # fallback
#         if best_sol is None:
#             best_sol = (r_curr, None, None, None, None, None)

#         r_star, FB0_star, delta_real_star, BPO_star, T_star, M_star = best_sol
#         return r_star, FB0_star, prob, delta_real_star, self.prev_rates, BPO_star, T_star, M_star
# BasePricingOptimiser.register(DistributionallyRobustPricingOptimiserEC, "dro_tau")