import jax.random
import pytest
import math
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.classical_methods import hilbert_method, mean_power_spectrum, adaptive_notch_filter, mle_polynomial
from chirpgp.toymodels import gen_chirp, gen_chirp_envelope, affine_freq, polynomial_freq, constant_mag
from jax.config import config

config.update("jax_enable_x64", True)


class TestClassicalMethods:

    @pytest.mark.parametrize('true_freq', [2., 30., 60., 100.])
    def test_hilbert(self, true_freq: float):
        """Test Hilbert method
        """
        dt = 0.001
        T = 1000

        ts = np.linspace(dt, dt * T, T)
        ys = np.sin(2 * math.pi * true_freq * ts)

        est_freq = hilbert_method(ts, ys)

        npt.assert_allclose(est_freq, true_freq)

    def test_psd(self):
        """Test power spectral density method
        """
        fs = 7200
        T = 4
        ts = np.arange(0, int(T * fs)) / fs

        freq_func, phase_func = affine_freq(200, 400.)

        chirp = gen_chirp(ts, constant_mag(1.), phase_func, 0.)

        new_ts, est_freq = mean_power_spectrum(ts, chirp)

        npt.assert_allclose(est_freq, freq_func(new_ts), rtol=1e-3)

    def test_mle_polynomial(self):
        dt = 0.01
        T = 300
        ts = jnp.linspace(dt, dt * T, T)

        true_poly_coeffs = [0.1, 0.2, 5.1, -1.3]
        freq_func, phase_func = polynomial_freq(true_poly_coeffs)
        true_chirp = gen_chirp(ts, constant_mag(1.), phase_func, 0.)

        key = jax.random.PRNGKey(666)
        Xi = 0.1
        ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key, shape=(ts.size,))

        key, _ = jax.random.split(key)
        init_params = jnp.array([1.] + true_poly_coeffs) + 0.01 * jax.random.normal(key, (5,))

        optimised_poly_coeffs, obj_vals = mle_polynomial(ts, ys, Xi, init_params, method='levenberg_marquardt',
                                                         lr=0.3, nu=0.2)
        npt.assert_allclose(optimised_poly_coeffs, jnp.array([1.] + true_poly_coeffs), rtol=2e-2)

    def test_anf(self):
        """Test adaptive notch filter method
        """
        dt = 0.001
        T = 5000
        ts = jnp.linspace(dt, dt * T, T)

        freq_func, phase_func = polynomial_freq([0.1, 0., 1.])
        chirp = gen_chirp_envelope(ts, constant_mag(1.), phase_func, 0.)

        freq_est, alpha_est, magnitude_est = adaptive_notch_filter(ts, chirp,
                                                                   alpha0=0., w0=freq_func(dt), s0=1 + 0.j,
                                                                   mu=1e-1, gamma_alpha=5e-6, gamma_w=8e-5)

        npt.assert_allclose(freq_est, freq_func(ts), atol=1.2e-2)
