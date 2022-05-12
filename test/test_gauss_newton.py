import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.gauss_newton import gauss_newton, levenberg_marquardt
from jax.config import config

config.update("jax_enable_x64", True)


class TestOptimisers:

    @pytest.mark.parametrize("optimiser", [gauss_newton, levenberg_marquardt])
    @pytest.mark.parametrize("true_vals", [(-2., 2., -1.),
                                           (1., -5., 3.),
                                           (10., -10., 2.)])
    def test_optimiser(self, optimiser, true_vals):
        dt = 0.01
        T = 150
        ts = jnp.linspace(dt, T * dt, T)

        def test_func(params):
            c0, c1, c2 = params
            return c0 + c1 * ts + c2 * ts ** 2

        true_params = jnp.array(true_vals)
        signal = test_func(true_params)

        Xi = 0.01
        key = jax.random.PRNGKey(666)
        ys = signal + jnp.sqrt(Xi) * jax.random.normal(key, (T,))

        init_params = jnp.zeros((3,))

        estimated_params, obj_vals = optimiser(test_func, init_params, ys, Xi, lr=0.1, stop_tolerance=1e-12)

        npt.assert_allclose(estimated_params, true_params, atol=1e-1)
