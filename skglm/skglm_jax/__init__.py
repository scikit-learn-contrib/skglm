# if not set, raises an error related to CUDA linking API.
# as recommended, setting the 'XLA_FLAGS' to bypass it.
# side-effect: (perhaps) slow compilation time.
# import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'  # noqa

# set flag to resolve bug with `jax.linalg.norm`
# ref: https://github.com/google/jax/issues/8916#issuecomment-1101113497
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"  # noqa

import jax
jax.config.update("jax_enable_x64", True)
