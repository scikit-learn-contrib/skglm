from skglm.penalties import L0_5, L2_3
from skglm.solvers import ProxNewton


def validate_solver(solver, datafit, penalty):
    """Ensure the solver is suited for the `datafit` + `penalty` problem.

    Parameters
    ----------
    solver : instance of BaseSolver
        Solver.

    datafit : instance of BaseDatafit
        Datafit.

    penalty : instance of BasePenalty
        Penalty.
    """
    if (isinstance(solver, ProxNewton)
        and not set(("raw_grad", "raw_hessian")) <= set(dir(datafit))):
        raise Exception(
            f"ProwNewton cannot optimize {datafit.__class__.__name__}, since `raw_grad`"
            " and `raw_hessian` are not implemented.")
    if ("ws_strategy" in dir(solver) and solver.ws_strategy == "subdiff"
        and isinstance(penalty, (L0_5, L2_3))):
        raise Exception(
            "ws_strategy=`subdiff` is not available for Lp penalties (p < 1). "
            "Set ws_strategy to `fixpoint`.")
