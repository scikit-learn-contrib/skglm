from functools import lru_cache

import numba
from numba import float32, float64
from numba.experimental import jitclass


def spec_to_float32(spec):
    """Convert a numba specification to an equivalent float32 one.

    Parameters
    ----------
    spec : list
        A list of (name, dtype) for every attribute of a jitclass.

    Returns
    -------
    spec32 : list
        A list of (name, dtype) for every attribute of a jitclass, where float64
        have been replaced by float32.
    """
    spec32 = []
    for name, dtype in spec:
        if dtype == float64:
            dtype32 = float32
        elif isinstance(dtype, numba.core.types.npytypes.Array):
            if dtype.dtype == float64:
                dtype32 = dtype.copy(dtype=float32)
            else:
                dtype32 = dtype
        else:
            raise ValueError(f"Unknown spec type {dtype}")
        spec32.append((name, dtype32))
    return spec32


@lru_cache()
def jit_cached_compile(klass, spec, to_float32=False):
    """Jit compile class and cache compilation.

    Parameters
    ----------
    klass : class
        Un instantiated Datafit or Penalty.

    spec : tuple
        A tuple of (name, dtype) for every attribute of a jitclass.

    to_float32 : bool, optional
        If ``True``converts float64 types to float32, by default False.

    Returns
    -------
    Instance of Datafit or penalty
        Return a jitclass.
    """
    if to_float32:
        spec = spec_to_float32(spec)

    return jitclass(spec)(klass)


def compiled_clone(instance, to_float32=False):
    """Compile instance to a jitclass.

    Parameters
    ----------
    instance : Instance of Datafit or Penalty
        Datafit or Penalty object.

    to_float32 : bool, optional
        If ``True``converts float64 types to float32, by default False.

    Returns
    -------
    Instance of Datafit or penalty
        Return a jitclass.
    """
    return jit_cached_compile(
        instance.__class__,
        instance.get_spec(),
        to_float32,
    )(**instance.params_to_dict())


if __name__ == '__main__':
    pass
