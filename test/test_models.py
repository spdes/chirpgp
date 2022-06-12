import pytest
import jax
import jax.scipy
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import scipy.linalg
import tme.base_jax as tme
from chirpgp.models import g, g_inv, model_chirp, disc_chirp_lcd, disc_chirp_lcd_cond_v, disc_chirp_tme, \
    disc_chirp_euler_maruyama, disc_m32, disc_model_lascala_lcd
from chirpgp.tools import lti_sde_to_disc
from jax.config import config

config.update("jax_enable_x64", True)


class TestModels:

    def test_g(self):
        """Test bijection.
        """
        x = jax.random.normal(jax.random.PRNGKey(666), (20,))
        npt.assert_allclose(x, g_inv(g(x)), rtol=1e-14, atol=0)

    @pytest.mark.parametrize('lam', [0.1, 1.])
    @pytest.mark.parametrize('b', [0.1, 1.])
    @pytest.mark.parametrize('ell', [0.1, 1.])
    def test_chirp_models(self, lam, b, ell):
        """Test model_chirp against lcd_chirp.
        """
        sigma, delta = 0.1, 0.1
        drift, dispersion, m0, P0, H = model_chirp(lam, b, ell, sigma, delta)
        m_and_cov = disc_chirp_lcd(lam, b, ell, sigma)

        dt = 0.1
        drift_matrix = np.zeros((4, 4))
        lcd_matrix = np.zeros((4, 4))
        for i in range(4):
            u = np.zeros((4,))
            u[i] = 1.
            drift_matrix[:, i] = drift(u)
            lcd_matrix[:, i] = m_and_cov(u, dt)[0]

        # Test mean
        npt.assert_allclose(scipy.linalg.expm(drift_matrix * dt), lcd_matrix)

        # Test cov
        u = jnp.array([0., 0., 1., 0.])
        F, Sigma = lti_sde_to_disc(drift_matrix, dispersion(u), dt)
        npt.assert_allclose(Sigma, m_and_cov(u, dt)[1], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('lam', [0.1, 1.])
    @pytest.mark.parametrize('b', [0.1, 1.])
    def test_lcd_chirp_cond_v(self, lam, b):
        u = jax.random.normal(jax.random.PRNGKey(666), (4,))
        dt = 0.1

        m_and_cov = disc_chirp_lcd(lam, b, 1., 1.)(u, dt)
        m_and_cov_cond_v = disc_chirp_lcd_cond_v(lam, b)(u[:2], u[2], dt)

        npt.assert_allclose(m_and_cov[0][:2], m_and_cov_cond_v[0])
        npt.assert_allclose(m_and_cov[1][:2, :2], m_and_cov_cond_v[1])

    def test_chirp_tme_against_lcd(self):
        lam, b, ell, sigma = 0.1, 0.1, 0.1, 0.1
        m_and_cov_lcd = disc_chirp_lcd(lam, b, ell, sigma)
        m_and_cov_tme = disc_chirp_tme(lam, b, ell, sigma, order=3)

        u = jax.random.normal(jax.random.PRNGKey(666), (4,))
        dt = 1e-3
        for i in range(2):
            npt.assert_allclose(m_and_cov_lcd(u, dt)[i], m_and_cov_tme(u, dt)[i], atol=1e-5)

    def test_chirp_euler(self):
        """dummy test.
        """
        disc_chirp_euler_maruyama()

    def test_disc_m32(self):
        ell, sigma = 1.1, 2.2
        m_and_cov = disc_m32(ell, sigma)
        m_and_cov2 = disc_chirp_lcd(1., 1., ell, sigma)

        u = jax.random.normal(jax.random.PRNGKey(666), (4,))
        dt = 1e-2
        npt.assert_allclose(m_and_cov(u[2:], dt)[0], m_and_cov2(u, dt)[0][2:])
        npt.assert_allclose(m_and_cov(u[2:], dt)[1], m_and_cov2(u, dt)[1][2:, 2:])

    @pytest.mark.parametrize('ell', [0.2, 1.])
    @pytest.mark.parametrize('sigma', [0.2, 1.])
    def test_disc_model_old_lcd(self, ell, sigma):
        """their mean coincides when using LCD.
        """
        m_and_cov = disc_model_lascala_lcd(ell, sigma)
        m_and_cov2 = disc_chirp_lcd(lam=jnp.array(0.), b=1., ell=ell, sigma=sigma)

        u = jax.random.normal(jax.random.PRNGKey(666), (4,))
        dt = 1e-2
        npt.assert_allclose(m_and_cov(u, dt)[0], m_and_cov2(u, dt)[0])

    def test_lcd_against_tme(self):
        """LCD and TME should give similar results.
        """
        lam, b, ell, sigma = 0.5, 0.1, 0.5, 1.
        drift, dispersion, _, _, _ = model_chirp(lam, b, ell, sigma, 0.1)

        m_and_cov_lcd = jax.jit(disc_chirp_lcd(lam, b, ell, sigma))
        m_and_cov_tme = jax.jit(lambda u, dt: tme.mean_and_cov(u, dt, drift, dispersion, order=3))

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num=5)
        for key in keys:
            m = jax.random.normal(key, (4,))
            lcd_m, lcd_cov = m_and_cov_lcd(m, 0.01)
            tme_m, tme_cov = m_and_cov_tme(m, 0.01)

            npt.assert_allclose(lcd_m, tme_m, rtol=4e-3)
            npt.assert_allclose(lcd_cov, tme_cov, atol=1e-3)
