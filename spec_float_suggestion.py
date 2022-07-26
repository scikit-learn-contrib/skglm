import numba
from numba import float32, float64, int32


def new_spec_to_float32(spec):
    spec32 = []

    for name, dtype in spec:
        if not isinstance(dtype, numba.types.Type):
            raise ValueError(
                f"spec items must be numba types\n"
                f"'{name}' has type '{dtype}'"
            )

        if not isinstance(dtype, numba.types.Float):
            attr_type = dtype
        elif isinstance(dtype, numba.types.Array):
            attr_type = dtype.copy(dtype=float32)
        else:
            attr_type = float32

        spec32.append((name, attr_type))

    return spec32


def spec_to_float32(spec):

    spec32 = []
    for name, dtype in spec:
        if dtype == float64:
            dtype32 = float32
        elif isinstance(dtype, numba.types.Array):
            if dtype.dtype == float64:
                dtype32 = dtype.copy(dtype=float32)
            else:
                dtype32 = dtype
        else:
            raise ValueError(f"Unknown spec type {dtype}")
        spec32.append((name, dtype32))
    return spec32


if __name__ == '__main__':
    spec_quadratic = (
        ('i', int32),
        ("alpha", float64),
        ('Xty', float64[:]),
        ('lipschitz', float64[:]),
    )

    spec_QuadraticGroup = (
        ('grp_ptr', int32[:]),
        ('grp_indices', int32[:]),
        ('lipschitz', float64[:])
    )

    print(new_spec_to_float32(spec_quadratic))
    print(new_spec_to_float32(spec_QuadraticGroup))

    print("=================")

    print(spec_to_float32(spec_QuadraticGroup))
    # this raise exception
    print(spec_to_float32(spec_quadratic))
