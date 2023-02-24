import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.toymodels import affine_freq, polynomial_freq, meow_freq, random_ou_mag, gen_chirp, gen_harmonic_chirp, \
    constant_mag
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


class TestToyModels:

    def test_polynomials(self):
        ts = jnp.linspace(0., 5., 1000)

        coeffs = [1.9, 3.6, -2.1]

        desired_freq_func = lambda t: coeffs[0] + coeffs[1] * t + coeffs[2] * t ** 2
        desired_phase_func = lambda t: coeffs[0] * t + 0.5 * coeffs[1] * t ** 2 + 1 / 3 * coeffs[2] * t ** 3

        freq_func, phase_func = polynomial_freq(coeffs)

        npt.assert_allclose(freq_func(ts), desired_freq_func(ts))
        npt.assert_allclose(phase_func(ts), desired_phase_func(ts))

    def test_random_ou_mag(self):
        ell, sigma = 0.1, 0.1

        ts = jnp.linspace(0.01, 1, 100)

        @partial(jax.vmap, in_axes=[0])
        def gen_traj(key):
            return random_ou_mag(ell, sigma, key)(ts)

        num_mcs = 100000
        key = jax.random.PRNGKey(666)
        trajectories = gen_traj(jax.random.split(key, num_mcs))
        npt.assert_allclose(jnp.mean(trajectories, axis=0), jnp.zeros_like(ts), atol=1e-3, rtol=0)
        npt.assert_allclose(jnp.var(trajectories, axis=0), jnp.ones_like(ts) * sigma ** 2, atol=1e-3, rtol=0)

    @pytest.mark.parametrize(['freq_phase_pair', 'argument_pair'], [(affine_freq, {'a': 1., 'b': 2.}),
                                                                    (polynomial_freq, {'coeffs': [1., -0.2, 0.1]}),
                                                                    (meow_freq, {})])
    def test_freq_vs_phase(self, freq_phase_pair, argument_pair):
        freq_func, phase_func = freq_phase_pair(**argument_pair)

        dt = 0.001
        T = 100
        ts = jnp.linspace(dt, dt * T, T)

        freqs = freq_func(ts)[1:]
        finite_difference_freqs_from_phase = jnp.diff(phase_func(ts)) / dt

        npt.assert_allclose(finite_difference_freqs_from_phase, freqs, atol=1e-3, rtol=1e-3)

    def test_harmonic_chirp(self):
        dt = 0.001
        T = 100
        ts = jnp.linspace(dt, dt * T, T)

        freq, phase = meow_freq(offset=8.)
        npt.assert_array_equal(gen_chirp(ts, constant_mag(1.), phase),
                               gen_harmonic_chirp(ts, [constant_mag(1.)], phase))
