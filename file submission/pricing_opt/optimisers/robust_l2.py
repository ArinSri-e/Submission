# pricing_opt/optimisers/robust_l2.py
from __future__ import annotations

import cvxpy as cp
import numpy as np
from typing import Optional

from pricing_opt.base import BasePricingOptimiser
from pricing_opt.optimisers.linear import LinearPricingOptimiser
from pricing_opt.utils import build_flow_model
from pricing_opt.data_types import OptimiserReturn


class RobustL2PricingOptimiser(LinearPricingOptimiser):
    """
    Linear optimiser with an ℓ₂‑penalty on the internal standardised
    balance shifts Δz_int.

    If robust_lambda="auto", the class chooses λ so that

        λ · ‖Δz_int‖₂   ≈   lambda_ratio × (baseline objective)

    at the *non‑robust* optimum.
    """
    name = "robust-l2"

    # ------------------------------------------------------------------ #
    # construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, *args,
                 lambda_ratio: float = 0.01,   # 1 % of baseline profit by default
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ratio = float(lambda_ratio)

    # ------------------------------------------------------------------ #
    # private helper: rule‑of‑thumb λ                                     #
    # ------------------------------------------------------------------ #
    def _auto_lambda(self, base_obj: float, dz_norm2: float) -> float:
        if dz_norm2 == 0:
            return 0.0
        return self.lambda_ratio * base_obj / dz_norm2

    # ------------------------------------------------------------------ #
    # main solve                                                          #
    # ------------------------------------------------------------------ #
    def solve(
        self,
        *,
        lower_rate: float = 0.05,
        upper_rate: float = 5.0,
        mass_guard: Optional[float] = None,
        robust_lambda: float | str = "auto",
        solver: str = "ECOS",
        verbose: bool = True,
        **solver_params,
    ) -> OptimiserReturn:

        p, idx_int = self.p, self.idx_int

        # -------------------------------------------------------------- #
        # 1) choose λ                                                   #
        # -------------------------------------------------------------- #
        if robust_lambda == "auto":
            # — baseline linear (no λ) —
            r0 = cp.Variable(p, name="rate0")
            FB0, dz0, cons0 = build_flow_model(
                self, r0,
                lower_rate=lower_rate,
                upper_rate=upper_rate,
                mass_guard=mass_guard,
            )
            prob0 = cp.Problem(cp.Maximize(cp.sum(FB0[idx_int])), cons0)
            prob0.solve(solver=solver, verbose=False, **solver_params)
            if prob0.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"[robust‑l2] baseline solve failed ({prob0.status})")

            base_obj = prob0.value
            dz_norm2 = np.linalg.norm(dz0.value[idx_int], 2)
            lam = self._auto_lambda(base_obj, dz_norm2)

            if verbose:
                print(f"[λ=auto] baseline={base_obj:.2f}, ‖Δz‖₂={dz_norm2:.2f} → λ={lam:.4g}")
        else:
            lam = float(robust_lambda)

        # -------------------------------------------------------------- #
        # 2) build robust model                                         #
        # -------------------------------------------------------------- #
        r = cp.Variable(p, name="rate")
        FB, dz, cons = build_flow_model(
            self, r,
            lower_rate=lower_rate,
            upper_rate=upper_rate,
            mass_guard=mass_guard,
        )

        penalty = lam * cp.norm(dz[idx_int], 2)
        objective = cp.Maximize(cp.sum(FB[idx_int]) - penalty)

        prob = cp.Problem(objective, cons)
        prob.solve(solver=solver, verbose=verbose, **solver_params)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"[robust‑l2] solver failed ({prob.status})")

        return (
            r.value,            # optimal rates
            FB.value,           # final balances
            prob,               # cvxpy Problem
            dz.value,           # standardised Δz
            self.prev_rates,    # previous rates
        )


# ------------------------------------------------------------------ #
# register with the global factory                                   #
# ------------------------------------------------------------------ #
BasePricingOptimiser.register(RobustL2PricingOptimiser, "robust-l2", "l2-robust")
