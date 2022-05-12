import math
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.quadratures import SigmaPoints
from jax import vmap
from jax.config import config

config.update("jax_enable_x64", True)

d = 1
gh_order = 5

cub = SigmaPoints.cubature(d=d)
gh = SigmaPoints.gauss_hermite(d=d, order=gh_order)


class TestQuadratures:

    def test_normalise(self):
        # Test if weights normalise
        npt.assert_almost_equal(jnp.sum(cub.w), 1.)
        npt.assert_almost_equal(jnp.sum(gh.w), 1.)

    def test_integrating_polynomial(self):
        c1 = 0.1
        c2 = 2.

        def f(x: jnp.ndarray):
            return c1 * x + c2 * x ** 2

        vf = vmap(f, [0])

        m = 0.5 * jnp.ones(shape=(d,))
        P = 0.2 * jnp.eye(d)

        true_integral_val = jnp.reshape(c1 * m + c2 * (P + jnp.outer(m, m)), (-1,))

        integral_cub = cub.expectation_from_nodes(vf, cub.gen_sigma_points(m, jnp.sqrt(P)))
        integral_gh = gh.expectation_from_nodes(vf, gh.gen_sigma_points(m, jnp.sqrt(P)))

        npt.assert_almost_equal(true_integral_val, integral_cub)
        npt.assert_almost_equal(true_integral_val, integral_gh)

    def test_integrating_sine(self):
        def f(x: jnp.ndarray):
            return jnp.sin(x)

        vf = vmap(f, [0])

        m = math.pi / 2 * jnp.ones(shape=(d,))
        P = jnp.eye(d)

        true_integral_val = jnp.reshape(jnp.sin(m) * jnp.exp(-P / 2), (-1,))

        integral_cub = cub.expectation_from_nodes(vf, cub.gen_sigma_points(m, jnp.sqrt(P)))
        integral_gh = gh.expectation_from_nodes(vf, gh.gen_sigma_points(m, jnp.sqrt(P)))

        npt.assert_allclose(true_integral_val, integral_cub, rtol=2e-1)
        npt.assert_almost_equal(true_integral_val, integral_gh, decimal=4)
