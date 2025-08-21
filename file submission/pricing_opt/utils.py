"""
Light‑weight helpers used by *all* optimisers:
  • _precompute(...)      – tensor flattening & scaling
  • nearest_psd(...)      – eigenvalue clip
"""

from __future__ import annotations
import numpy as np
# pricing_opt/utils.py  (or pricing_opt/base.py if you prefer)
import cvxpy as cp


# ---------------------------------------------------------------------------
# Precompute helper
# ---------------------------------------------------------------------------
def _precompute(
    product_index: Dict[str, int],
    reverse_index: Dict[int, str],
    F: np.ndarray,
    CF: np.ndarray,
    CR: np.ndarray,
    C0: np.ndarray,
    prev_balances: np.ndarray,
    prev_rates: np.ndarray,
    scale_factor: float,
    rate_mean: float = 0.0,
    size: int = None
):
    """
    Build broadcast maps & reduced flow coefficients for the *product‑level* rate model.
    This is a helper function to prepare the data for the optimiser.
    
    Assumptions
    -----------
    - Shapes: F, CF, CR are (p, p, K) with matching p. 
    - F and CF has already computed
    - R is stand for the decisions variables.The features are as follows:
    - CR[...,0] = origin product rate term
      CR[...,1] = destination product rate term
      CR[...,2] = origin delta (r_i - r_i_prev)
      CR[...,3] = dest   delta (r_j - r_j_prev)
      (extra slices ignored)
    - Product IDs starting with 'INT' are treated as internal.

    Returns
    -------
    dict containing all fields needed by the optimiser instance.
    """
    # --- basic shape checks -------------------------------------------------
    F  = np.asarray(F,  dtype=float)
    CF = np.asarray(CF, dtype=float)
    CR = np.asarray(CR, dtype=float)
    C0 = np.asarray(C0, dtype=float)

    # Checking the dimensions of the F and CF again!
    if F.shape != CF.shape:
        raise ValueError(f"F{F.shape} and CF{CF.shape} must match.")
    if F.shape[:2] != CR.shape[:2] or F.shape[:2] != C0.shape:
        raise ValueError("Leading (p,p) dims of F/CF/CR/C0 must agree.")

    p = F.shape[0]          # number of products
    m = CR.shape[2]
    if m < 4:     # CR must have at least 4 slices (in this example)
        raise ValueError("CR must have >=4 slices (origin, dest, o-delta, d-delta).")
    k= CR.shape[2]  # number of features (slices in CR)
    p2 = p * p

    # --- identify internal products ----------------------------------------
    products = [reverse_index[i] for i in range(p)]            
    is_int = np.array([pid.startswith("INT") for pid in products], dtype=bool)
    idx_int = np.where(is_int)[0]       # indices of internal products

    # --- origin/dest maps for row-major flatten ----------------------------
    origin_map = np.repeat(np.arange(p), p)  # flow idx -> origin i  - repeat for each row
    dest_map   = np.tile(np.arange(p), p)    # flow idx -> dest j - the destination of the map

    # --- constant block from fixed (non-optimised) features -----------------
    const_block = C0 + np.einsum("ijk,ijk->ij", CF, F)  # (p,p)

    # --- previous rates -----------------------------------------------------
    prev_rates = np.asarray(prev_rates, dtype=float).reshape(-1)        # check shape to be really 1 dimension
    if prev_rates.shape[0] != p:
        raise ValueError(f"prev_rates length {prev_rates.shape[0]} != p={p}.")
    prev_origin = prev_rates[origin_map]  # (p2,)
    prev_dest   = prev_rates[dest_map]    # (p2,)

    a = {}
    for i in range(m):
        a[f'a{i}'] = CR[..., i]
    coef_ratec = a['a0'].reshape(p2)
    coef_xgap = a['a1'].reshape(p2)
    coef_drate = a['a2'].reshape(p2)
    coef_logT = a['a3'].reshape(p2)
    coef_sqrt = a['a4'].reshape(p2)
    coef_promo = a['a5'].reshape(p2)
    coef_dest = a['a1'].reshape(p2)  # destination rate term
    coef_dest2 = a['a2'].reshape(p2)  # destination delta term

    # as the features also have the spread between the rate last month
    # intercept_adj_flat = const_block.reshape(p2) - coef_origin * prev_origin - coef_dest * prev_dest
    intercept_adj_flat = const_block.reshape(p2)

    # --- scale consistently -------------------------------------------------
    # Gurobi and CVXPY expect the objective to be scaled by a factor
    intercept_adj_flat *= scale_factor
    # Scale all coefficients a0, a1, a2, a3 by scale_factor
    # coef_rcsq  *= scale_factor
    # coef_xgap  *= scale_factor
    # coef_drate *= scale_factor
    # coef_logT  *= scale_factor
    # coef_sqrt  *= scale_factor
    # coef_promo *= scale_factor
    # --- pack results ---------------------------------------------
    return {
        "p": p,
        "p2": p2,
        "is_int": is_int,
        "idx_int": idx_int,
        "origin_map": origin_map,
        "dest_map": dest_map,
        "prev_balances": np.asarray(prev_balances, dtype=float).reshape(p),
        "prev_rates": prev_rates,
        "intercept_adj_flat": intercept_adj_flat,
        "scale_factor": float(scale_factor),
        "coef_rcsq": coef_ratec,
        "coef_xgap": coef_xgap,
        "coef_drate": coef_drate,
        "coef_logT": coef_logT,
        "coef_sqrt": coef_sqrt,
        "coef_promo": coef_promo,
        'rate_mean': rate_mean
    }

#-----------------------------------------------------------------------------
# PSD helper
#-----------------------------------------------------------------------------
def nearest_psd(M: np.ndarray) -> np.ndarray:
    """Return the closest (in Frobenius norm) PSD matrix to M."""
    #‑‑ symmetrise first
    M = 0.5 * (M + M.T)
    #‑‑ eigen‑decompose
    w, V = np.linalg.eigh(M)
    w_clipped = np.clip(w, 0, None)      # set negative λ to 0
    return (V * w_clipped) @ V.T         # V diag(w⁺) Vᵀ



def build_flow_model(self, r: cp.Variable,
                     *, lower_rate: float, upper_rate: float,
                     mass_guard: float | None):

    p, is_int, idx_int = self.p, self.is_int, self.idx_int
    cons = [
        r[~is_int] == self.prev_rates[~is_int],
        r[is_int]  >= lower_rate,
        r[is_int]  <= upper_rate,
        # … your business‑rule constraints here …
    ]

    # --- flows -------------------------------------------------
    r_o   = r[self.origin_map]
    r_d   = r[self.dest_map]

  # M is the destination product's M


    Pflat = (self.intercept_adj_flat +
             cp.multiply(self.coef_dest, r_o) +
             cp.multiply(self.coef_dest2,  r_d))
    P = cp.reshape(Pflat, (p, p), order="C")

    delta_scaled  = cp.sum(P, axis=0) - cp.sum(P, axis=1)
    delta_std     = delta_scaled / self.scale_factor
    delta_real    = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
    FB0           = self.prev_balances + delta_real            # nominal £

    # optional mass guard
    if mass_guard is not None:
        cons.append(cp.norm(delta_std[idx_int], 1) <= mass_guard)

    return FB0, delta_real, cons


def build_deter_flow_model(self, r: cp.Variable,  BPO: cp.Variable, T: cp.Variable, M: cp.Variable,
                           *, lower_rate: float, upper_rate: float,
                           mass_guard: float | None):

    p, is_int, idx_int = self.p, self.is_int, self.idx_int
    cons = [
        r[~is_int] == self.prev_rates[~is_int],
        r[is_int]  >= lower_rate,
        r[is_int]  <= upper_rate,
    ]

    # Extract destination-side variables
    BPO_d = BPO[self.dest_map]
    T_d   = T[self.dest_map]
    M_d   = M[self.dest_map]
    r_d = r[self.dest_map]  # destination product rates

    # Compute drate (change in rate) from previous period
    prev_r_d = self.prev_rates[self.dest_map]
    drate = r_d - prev_r_d
    rcsq_affine = (y0 - mu)**2 + 2.0*(y0 - mu)*(r_d - mu)   # affine in r

    xgap  = r_d - r_d.mean()

    # Hypographs for log(1+T) and sqrt(M)
    t_log  = cp.Variable(BPO_d.shape, name="t_log")
    m_sqrt = cp.Variable(BPO_d.shape, name="m_sqrt")
    cons += [t_log <= cp.log(1 + T_d),  t_log >= 0,
            m_sqrt <= cp.sqrt(M_d),    m_sqrt >= 0]


        

#['rate_c_sq','xgap','drate','log1p_T','sqrt_M','promo_adstock']
#"coef_rcsq": coef_rcsq,
    #     "coef_xgap": coef_xgap,
    #     "coef_drate": coef_drate,
    #     "coef_logT": coef_logT,
    #     "coef_sqrt": coef_sqrt,
    #     "coef_promo": coef_promo
    # # Flow expression using reduced features

    Pflat = (self.intercept_adj_flat +
            #  cp.multiply(self.coef_rcsq, rcsq) +
             cp.multiply(self.coef_xgap, xgap) +
             cp.multiply(self.coef_drate, drate) +
             cp.multiply(self.coef_promo, BPO_d) +
             cp.multiply(self.coef_logT, t_log) +
             cp.multiply(self.coef_sqrt, m_sqrt))

    # Reshape flat flow into matrix
    P = cp.reshape(Pflat, (p, p), order="C")

    # Balance equations
    inflow_int = cp.sum(P[:, self.idx_int], axis=0)
    FB0_int = self.prev_balances[self.idx_int] + inflow_int
    delta_scaled = cp.sum(P, axis=0) - cp.sum(P, axis=1)
    delta_std = delta_scaled / self.scale_factor
    delta_real = cp.multiply(delta_std, self.std_deltas) + self.mean_deltas
    FB0 = self.prev_balances + delta_real

    cons += [BPO >= 0.0, BPO <= 0.01,
         T   >= 0.0, T   <= 1.0,
         M   >= 0.0, M   <= 1.0]
    
    # # SDP constraints to enforce q_j ≥ r_j²
    # for j in range(self.p):
    #     cons.append(cp.bmat([[1, r[j]], [r[j], qwz[j]]]) >> 0)

    # Optional mass guard (limits total balance change)
    if mass_guard is not None:
        cons.append(cp.norm(delta_std[idx_int], 1) <= mass_guard)



    return FB0, delta_real, cons
