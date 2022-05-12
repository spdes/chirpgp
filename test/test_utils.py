import pytest
import math
import jax
import jax.numpy as jnp
import jax.scipy
import numpy.testing as npt
from chirpgp.tools import simulate_lgssm, lti_sde_to_disc, fwd_transformed_pdf, chol_partial_const_diag, rmse
from chirpgp.quadratures import gaussian_expectation
from jax.config import config

config.update("jax_enable_x64", True)


class TestUtils:

    @pytest.mark.parametrize("lam, f, dt", [(0.1, 1., 0.1), (1., 2., 0.1),
                                            (0., 0.1, 0.1), (0., 0.5, 0.1),
                                            (0., 0.1, 1.), (0., 0.1, 2.)])
    def test_lti_disc(self, lam: float, f: float, dt: float):
        A = jnp.array([[-lam, -2 * math.pi * f],
                       [2 * math.pi * f, -lam]])

        B = jnp.eye(2)
        F, Q = lti_sde_to_disc(A, B, dt)

        z = 2 * math.pi * dt * f
        expected_F = jnp.array([[jnp.cos(z), -jnp.sin(z)],
                                [jnp.sin(z), jnp.cos(z)]]) * jnp.exp(-dt * lam)
        if lam == 0:
            expected_Q = jnp.eye(2) * dt
        else:
            expected_Q = jnp.eye(2) * (1 - jnp.exp(-2 * dt * lam)) / (2 * lam)

        npt.assert_allclose(F, expected_F)
        npt.assert_allclose(Q, expected_Q, atol=1e-12)

    def test_simulate(self):
        lam = 0.1
        F = lambda dt: jnp.exp(-dt * lam) * jnp.eye(2)
        Q = lambda dt: 1 / (2 * lam) * (1 - jnp.exp(-2 * lam * dt)) * jnp.eye(2)

        x0 = jnp.ones((2,))
        test_dt = 0.001
        T = 100

        @jax.jit
        def jitted_simulator(subkey: jnp.ndarray):
            return simulate_lgssm(F(test_dt), Q(test_dt), x0, T, subkey)

        num_mc = 100000

        keys = jax.random.PRNGKey(999)
        keys = jax.random.split(keys, num_mc + 1)

        _, trajs = jax.lax.scan(lambda carry, elem: (carry, jitted_simulator(elem)), keys[0], keys[1:])

        npt.assert_allclose(jnp.mean(trajs[:, -1, :], axis=0), F(T * test_dt) @ x0, rtol=1e-2, atol=1e-3)
        npt.assert_allclose(jnp.cov(trajs[:, -1, :], rowvar=False), Q(T * test_dt), rtol=1e-2, atol=1e-3)

    def test_change_of_v_formula(self):
        def pdf_x(x):
            return jnp.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

        def g(x): return x ** 2

        def g_inv(y): return jnp.sqrt(y)  # Note that this only has half.

        def true_pdf_y(y):
            """Chi-square"""
            return jnp.exp(-0.5 * y) / jnp.sqrt(2 * math.pi * y)

        pdf_y = fwd_transformed_pdf(pdf_x, g_inv)

        ys = jnp.linspace(0.1, 10, 1000)

        npt.assert_allclose(pdf_y(ys), 0.5 * true_pdf_y(ys))

    def test_chol_partial_const_diag(self):
        key = jax.random.PRNGKey(999)
        x = jax.random.normal(key, (3,))
        a = jax.scipy.linalg.block_diag(4., 9., jnp.outer(x, x) + jnp.eye(3))
        l = chol_partial_const_diag(a, 2, lower=True)
        npt.assert_allclose(l @ l.T, a)

    def test_g_expectation(self):
        def g(x): return jnp.exp(x)

        key = jax.random.PRNGKey(111)
        ms = jax.random.normal(key, (100, 1))
        key, _ = jax.random.split(key)
        Ps = jax.random.uniform(key, (100, 1, 1), minval=0.1, maxval=1.)

        approx_expec = gaussian_expectation(ms, jnp.sqrt(Ps), func=g, d=1, order=10)
        true_expec = g(ms + Ps[:, 0] / 2)

        npt.assert_allclose(approx_expec, true_expec)

    @pytest.mark.parametrize('reduce_sum', [True, False])
    def test_rmse(self, reduce_sum: bool):
        x1 = jnp.array([[1., 2., 3.],
                        [4., 5., 6.]])
        x2 = jnp.array([[0., 1., 2.],
                        [3., 4., 5.]])
        if reduce_sum:
            npt.assert_allclose(rmse(x1, x2, reduce_sum), 3.)
        else:
            npt.assert_allclose(rmse(x1, x2, reduce_sum), jnp.array([1., 1., 1.]))
