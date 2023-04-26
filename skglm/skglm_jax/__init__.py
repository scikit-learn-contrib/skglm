# if not set, raises an error related to CUDA linking API.
# as recommended, setting the 'XLA_FLAGS' to bypass it.
# side-effect: (perhaps) slow compilation time.
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'  # noqa
