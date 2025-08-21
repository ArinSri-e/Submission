from typing import Tuple
import numpy as np
import cvxpy as cp

# Re‑exported so every optimiser can `from ...data_types import OptimiserReturn`
OptimiserReturn = Tuple[
    np.ndarray,  # optimal rates
    np.ndarray,  # final balances
    cp.Problem,  # cvxpy Problem (for diagnostics)
    np.ndarray,  # delta vector (real units)
    np.ndarray,  # previous rates
]
