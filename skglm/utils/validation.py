import re


def check_group_compatible(obj):
    """Check whether ``obj`` is compatible with ``bcd_solver``.

    Parameters
    ----------
    obj : instance of BaseDatafit or BasePenalty
        Object to check.

    Raises
    ------
    ValueError
        if the ``obj`` doesn't have a ``grp_ptr`` and ``grp_indices``
        attributes.
    """
    obj_name = obj.__class__.__name__
    group_attrs = ('grp_ptr', 'grp_indices')

    for attr in group_attrs:
        if not hasattr(obj, attr):
            raise ValueError(
                f"datafit and penalty must be compatible with 'bcd_solver'.\n"
                f"'{obj_name}' is not block-separable. "
                f"Missing '{attr}' attribute."
            )


def check_obj_solver_attr(obj, solver, required_attr):
    """Check whether datafit or penalty is compatible with solver.

    Parameters
    ----------
    obj : Instance of Datafit or Penalty
        The instance Datafit (or Penalty) to check.

    solver : Instance of Solver
        The instance of Solver to check.

    required_attr : List or tuple of strings
        The attributes that ``obj`` must have.

    Raises
    ------
        AttributeError
            if any of the attribute in ``required_attr`` is missing
            from ``obj`` attributes.
    """
    missing_attrs = []
    for attr in required_attr:
        attributes = attr if not isinstance(attr, str) else (attr,)

        for a in attributes:
            if hasattr(obj, a):
                break
        else:
            missing_attrs.append(_join_attrs_with_or(attributes))

    if len(missing_attrs):
        required_attr = [_join_attrs_with_or(attrs) for attrs in required_attr]

        # get name obj and solver
        name_matcher = re.compile(r"\.(\w+)'>")

        obj_name = name_matcher.search(str(obj.__class__)).group(1)
        solver_name = name_matcher.search(str(solver.__class__)).group(1)

        raise AttributeError(
            f"{obj_name} is not compatible with {solver_name}. "
            f"It must implement {' and '.join(required_attr)}\n"
            f"Missing {' and '.join(missing_attrs)}."
        )


def _join_attrs_with_or(attrs):
    #
    if isinstance(attrs, str):
        return f"`{attrs}`"

    if len(attrs) == 1:
        return f"`{attrs[0]}`"

    out = " or ".join([f"`{a}`" for a in attrs])
    return f'"{out}"'
