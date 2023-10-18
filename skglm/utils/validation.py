
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


def check_obj_solver_compatibility(obj, required_attr):
    """Check whether datafit or penalty is compatible with a solver.

    Parameters
    ----------
    obj : Instance of Datafit or Penalty
        The instance Datafit (or Penalty) to check.

    required_attr : List or tuple of strings
        The attributes that ``obj`` must have.

    Raises
    ------
        AttributeError
            if any of the attribute in ``required_attr`` is missing
            from ``obj`` attributes.
    """
    missing_attrs = [f"`{attr}`" for attr in required_attr if not hasattr(obj, attr)]

    if len(missing_attrs):
        required_attr = ' and '.join(f"`{attr}`" for attr in required_attr)

        raise AttributeError(
            "Object not compatible with solver. "
            f"It must implement {' and '.join(required_attr)} \n"
            f"Missing {' and '.join(missing_attrs)}."
        )
