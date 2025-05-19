"""Solver strategy implementations for optimization pipelines.

This module provides strategies for adapting solver parameters during optimization
stages, particularly for continuation and progressive-smoothing pipelines.
"""

from sklearn.base import clone
import copy


DEFAULT_CONFIG = {
    "base_tol": 1e-5,
    "tol_delta_factor": 1e-3,
    "max_iter_start": 150,
    "max_iter_step": 50,
    "max_iter_cap": 1000,
    "large_problem_threshold": 1000,
    "small_problem_threshold": 100,
    "p0_frac_large": 0.1,
    "p0_frac_small": 0.5,
    "p0_min": 10,
}


class StageBasedSolverStrategy:
    """Stage-wise tuning of a base solver for continuation and progressive-smoothing.

    This class adapts solver parameters based on the stage of optimization.
    """

    def __init__(self, config=None):
        """Initialize the solver strategy with configuration."""
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        if not 0 < self.config["tol_delta_factor"] < 1:
            raise ValueError("tol_delta_factor must be in (0, 1)")

        small_thresh = self.config["small_problem_threshold"]
        large_thresh = self.config["large_problem_threshold"]
        if small_thresh > large_thresh:
            raise ValueError(
                "small_problem_threshold must not exceed large_problem_threshold")

        self._growth_factor = (
            1 + self.config["max_iter_step"] / self.config["max_iter_start"]
        )

    def create_solver_for_stage(self, base_solver, delta, stage, n_features):
        """Clone base_solver and adapt tol, max_iter and p0 for the given stage."""
        solver = self._clone(base_solver)
        self._set_tol(solver, delta, stage)
        self._set_max_iter(solver, stage)
        self._set_working_set(solver, n_features)
        return solver

    @staticmethod
    def _clone(est):
        """Try sklearn.clone first; fall back to deepcopy."""
        try:
            return clone(est)
        except Exception:
            return copy.deepcopy(est)

    def _set_tol(self, solver, delta, stage):
        """Set tolerance based on stage and delta value."""
        if hasattr(solver, "tol"):
            base = self.config["base_tol"]
            solver.tol = base if stage == 0 else max(
                base, self.config["tol_delta_factor"] * delta)

    def _set_max_iter(self, solver, stage):
        """Set maximum iterations based on stage number."""
        if hasattr(solver, "max_iter"):
            start = self.config["max_iter_start"]
            solver.max_iter = min(
                self.config["max_iter_cap"],
                int(start * self._growth_factor ** stage)
            )

    def _set_working_set(self, solver, n_features):
        """Set working set size based on number of features."""
        if not hasattr(solver, "p0"):
            return

        cfg = self.config
        if n_features > cfg["large_problem_threshold"]:
            frac = cfg["p0_frac_large"]
        elif n_features < cfg["small_problem_threshold"]:
            frac = cfg["p0_frac_small"]
        else:
            frac = 0.5 * (cfg["p0_frac_large"] + cfg["p0_frac_small"])

        solver.p0 = max(cfg["p0_min"], int(n_features * frac))
