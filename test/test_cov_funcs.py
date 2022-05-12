import math
import pytest
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.cov_funcs import vmap_cov_harmonic_sde, vmap_marginal_cov_harmonic_sde
from jax.config import config

config.update("jax_enable_x64", True)


class TestCovFuncs:

    @pytest.mark.parametrize("elem", [0, 1])
    @pytest.mark.parametrize("f, lam, b", [(0.1, 1., 0.1), (0.1, 0., 1.), (1., 1., 0.)])
    def test_cov_func_of_harmonic_sde(self, elem: int, f: float, lam: float, b: float):
        dt = 0.01
        T = 30
        ts = jnp.linspace(dt, T * dt, T)

        init_cov = 2 * b * lam * jnp.eye(2)

        covs = vmap_cov_harmonic_sde(ts, ts, init_cov, f, lam, b)

        # Test if the diagonal coincides with the marginal
        diag = jnp.diag(covs[:, :, elem, elem])
        marginal = vmap_marginal_cov_harmonic_sde(ts, 0., init_cov, lam, b, 2 * math.pi * f)[:, elem, elem]
        npt.assert_allclose(diag, marginal)

        # The marginal should be increasing in t if lam=0
        if lam == 0:
            npt.assert_array_less(jnp.zeros((T - 1,)), jnp.diff(diag))

    # TODO: test approx_cond_cov_chirp_sde
