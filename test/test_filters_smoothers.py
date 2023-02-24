import pytest
import math
import jax
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt

from chirpgp.filters_smoothers import kf, rts, ekf, eks, cd_ekf, cd_eks, \
    sgp_filter, sgp_smoother, cd_sgp_filter, cd_sgp_smoother
from chirpgp.quadratures import SigmaPoints
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)


class TestFiltersSmoothers:

    @pytest.mark.parametrize('a, b', ([1., 1.],
                                      [2.1, 0.4]))
    def test_equivalence_on_linear_models(self, a, b):

        dim_x = 3
        dt = 0.01

        # dx = A x dt + B dW
        A = -a * jnp.eye(dim_x)
        B = b * jnp.eye(dim_x)

        drift = lambda u: A @ u
        dispersion = lambda _: B

        # x_k = F x_{k-1} + Q
        F = math.exp(-a * dt) * jnp.eye(dim_x)
        Sigma = b ** 2 / (2 * a) * (1 - math.exp(-2 * a * dt)) * jnp.eye(dim_x)

        Xi = 0.1
        H = jnp.ones((dim_x,))

        m0 = jnp.zeros((dim_x,))
        P0 = 0.1 * jnp.eye(dim_x)

        def simulate():
            num_measurements = 1000
            xx = np.zeros((num_measurements, dim_x))
            yy = np.zeros((num_measurements,))
            x = np.array(m0).copy()
            for i in range(num_measurements):
                x = F @ x + np.sqrt(Sigma) @ np.random.randn(dim_x)
                y = H @ x + np.sqrt(Xi) * np.random.randn()
                xx[i] = x
                yy[i] = y

            return jnp.asarray(xx), jnp.asarray(yy)

        @jax.jit
        def m_and_cov(u, _):
            return F @ u, Sigma

        xs, ys = simulate()

        kf_results = kf(F, Sigma, H, Xi, m0, P0, ys)
        ekf_results = ekf(m_and_cov, H, Xi, m0, P0, dt, ys)
        cd_ekf_results = cd_ekf(drift, dispersion, H, Xi, m0, P0, dt, ys)
        sgps = SigmaPoints.gauss_hermite(d=dim_x, order=4)
        ghkf_results = sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, ys)
        cd_ghkf_results = cd_sgp_filter(drift, B, sgps, H, Xi, m0, P0, dt, ys)

        for i in range(3):
            npt.assert_allclose(kf_results[i], ekf_results[i])
            npt.assert_allclose(kf_results[i], ghkf_results[i])
            npt.assert_allclose(kf_results[i], cd_ekf_results[i], rtol=1e-5)
            npt.assert_allclose(kf_results[i], cd_ghkf_results[i], rtol=1e-5)

        rts_results = rts(F, Sigma, kf_results[0], kf_results[1])
        eks_results = eks(m_and_cov, ekf_results[0], ekf_results[1], dt)
        cd_eks_results = cd_eks(drift, dispersion, cd_ekf_results[0], cd_ekf_results[1], dt)
        ghks_results = sgp_smoother(m_and_cov, sgps, ghkf_results[0], ghkf_results[1], dt)
        cd_ghks_results = cd_sgp_smoother(drift, B, sgps, cd_ghkf_results[0], cd_ghkf_results[1], dt)

        for i in range(2):
            npt.assert_allclose(rts_results[i], eks_results[i])
            npt.assert_allclose(rts_results[i], ghks_results[i])
            npt.assert_allclose(rts_results[i], cd_eks_results[i], atol=1e-1)
            npt.assert_allclose(cd_eks_results[i], cd_ghks_results[i])
