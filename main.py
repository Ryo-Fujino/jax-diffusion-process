import jax.numpy as jnp
import jax
from functools import partial

@partial(jax.jit, static_argnums=(2, 3))
def kernel_fn(key, x, drift_fn, diffusion_fn, dt):
    key, subkey = jax.random.split(key)
    x = x + drift_fn(x) * dt + diffusion_fn(x) * jax.random.normal(key=subkey, shape=(x.shape)) * jnp.sqrt(dt)
    return key, x

@partial(jax.jit, static_argnums=(1, 2, 3))
def diffusion_sampler_fn(key, drift_fn, diffusion_fn, num, dt, x0):

    def step_fn(carry, _x):
        key, x = carry
        key, x = kernel_fn(key, x, drift_fn, diffusion_fn, dt)
        return (key, x), x

    carry = (key, x0)
    _, states = jax.lax.scan(step_fn, carry, None, num)
    return states