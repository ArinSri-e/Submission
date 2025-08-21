# ga_objective_from_det.py
import numpy as np

def build_ga_objective(det_opt,
                       *,
                       BPO=None,   # shape (p*p,) or scalar -> broadcast
                       T=None,     # shape (p*p,) or scalar
                       M=None,     # optional, not used in your sqrt term
                       mass_guard=None,
                       w_mass=1e5):
    """
    det_opt: an initialised DeterministicPricingOptimiser (after __init__)
    Returns: objective_fn(r: (p,)) -> float to MAXIMISE
    Notes:
      - Uses SAME formulae as your CVXPY model but in NumPy.
      - Holds non-rate features fixed (BPO, T, M). If None, uses zeros.
      - Applies an L1 mass-guard penalty if provided.
    """
    p         = det_opt.p
    dest_map  = det_opt.dest_map              # (p*p,) mapping
    idx_int   = det_opt.idx_int               # bool mask of internal products length p
    scale_f   = det_opt.scale_factor
    prev_bal  = det_opt.prev_balances         # (p,)
    prev_rates= det_opt.prev_rates            # (p,)
    mu        = det_opt.rate_mean             # (p,)
    std_d     = det_opt.std_deltas            # (p,)
    mean_d    = det_opt.mean_deltas           # (p,)

    # Coefficients (all (p*p,) flattened, row-major like you used)
    intercept = det_opt.intercept_adj_flat
    c_rcsq    = det_opt.coef_rcsq
    c_xgap    = det_opt.coef_xgap
    c_drate   = det_opt.coef_drate
    c_promo   = det_opt.coef_promo
    c_logT    = det_opt.coef_logT
    c_sqrt    = det_opt.coef_sqrt

    # Fixed non-rate features
    if BPO is None: BPO = 0.0
    if T   is None: T   = 0.0
    if M   is None: M   = 0.0  # not used directly in your sqrt-term
    BPO = np.broadcast_to(np.asarray(BPO, float).reshape(-1), (p*p,))
    T   = np.broadcast_to(np.asarray(T,   float).reshape(-1), (p*p,))
    # Your model uses m_sqrt <= sqrt(T); when fixed we take m_sqrt = sqrt(T)
    t_log  = np.log1p(T)
    m_sqrt = np.sqrt(T)

    mu_d = mu[dest_map]  # (p*p,)

    def objective_fn(r: np.ndarray) -> float:
        r = np.asarray(r, float).reshape(p)

        # Flow-level features from current r
        y0   = r[dest_map]                            # (p*p,)
        dr   = r[dest_map] - prev_rates[dest_map]     # (p*p,)
        # xgap: each flow uses dest rate minus mean of dests (your code used mean(y0))
        xgap = y0 - y0.mean()

        # Square term (deterministic, not linearised for GA eval)
        rcsq = (y0 - mu_d)**2

        # Flow prediction (flattened) exactly as in your CP model
        Pflat = (
            intercept
            + c_rcsq * rcsq
            + c_xgap * xgap
            + c_drate * dr
            + c_promo * BPO
            + c_logT * t_log
            + c_sqrt * m_sqrt
        )
        P = Pflat.reshape(p, p)  # row-major (order="C")

        # Balance deltas (scaled -> std -> real) and FB0
        delta_scaled = P.sum(axis=0) - P.sum(axis=1)           # (p,)
        delta_std    = delta_scaled / float(scale_f)
        delta_real   = delta_std * std_d + mean_d              # (p,)
        FB0          = prev_bal + delta_real                   # (p,)

        # Base objective: sum over internal products
        obj = float(FB0[idx_int].sum())

        # Optional mass-guard penalty: ||delta_std[idx_int]||_1 <= mass_guard
        if mass_guard is not None:
            viol = max(0.0, np.abs(delta_std[idx_int]).sum() - float(mass_guard))
            obj -= w_mass * viol

        return obj

    return objective_fn
