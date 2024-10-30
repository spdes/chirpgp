import pytest
import math
import jax
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.tools import lti_sde_to_disc
from chirpgp.models import _m32_solution

jax.config.update("jax_enable_x64", True)


class TestM32:

    @pytest.mark.parametrize("ell, sigma, dt", [(0.1, 1., 0.1),
                                                (1., 2., 0.1),
                                                (0.456, 1.234, 0.789),
                                                (0.1789, 11.234, 0.0789)])
    def test_m32(self, ell: float, sigma: float, dt: float):
        lam = math.sqrt(3) / ell
        q = 4 * sigma ** 2 * lam ** 3
        A = jnp.array([[0., 1.],
                       [-lam ** 2, -2 * lam]])
        B = jnp.array([0., math.sqrt(q)])

        correct_mean, correct_cov = lti_sde_to_disc(A, B, dt)

        mean, cov = _m32_solution(ell, sigma, dt)

        npt.assert_allclose(mean, correct_mean, atol=1e-12)
        npt.assert_allclose(cov, correct_cov, atol=1e-12)
