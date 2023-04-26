import jax
from functools import partial


jax_jit_method = partial(jax.jit, static_argnums=(0,))
