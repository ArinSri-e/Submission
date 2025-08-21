

import cvxpy as cp
import numpy as np
from typing import Optional


def quadratic(p: int, r: np.ndarray, solver: str , Q_raw: np.ndarray,sdp:bool) -> cp.Expression:
    if sdp:
    # SDP Shor lift
        W     = cp.Variable((p, p), PSD=True)
        r_col = cp.reshape(r, (p, 1))
        r_row = cp.reshape(r, (1, p))
        cons += [
            cp.bmat([
                [cp.Constant([[1.0]]), r_row],
                [r_col,                W    ],
            ]) >> 0
        ]
        quad_pen = cp.trace(Q_raw @ W)
        # switch to an SDP‐capable solver
        if solver in {"ECOS_BB", "OSQP"}:
            solver = "MOSEK"
        # tiny regularizer
    else:
        # QP path
        quad_pen = cp.sum_squares(Q_raw @ r)
    quad_pen = quad_pen + 1e-6 * cp.sum_squares(r)
    return quad_pen

