import math
import jax
import jax.numpy as jnp
import jaxopt
import time
from chirpgp.models import g, g_inv, build_chirp_model
from chirpgp.filters_smoothers import ekf, eks
from chirpgp.toymodels import gen_chirp, meow_freq, constant_mag

jax.config.update("jax_enable_x64", True)

# Times
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

# Frequency
true_freq_func, true_phase_func = meow_freq(offset=8.)

key = jax.random.PRNGKey(666)
key_for_measurements, key_for_ou = jax.random.split(key)

true_chirp = gen_chirp(ts, constant_mag(1.), true_phase_func)

Xi = 0.1
ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

init_theta = g_inv(jnp.array([0.1, 0.1, 0.1, 1., 1., 7.]))


def obj_func(theta: jnp.ndarray):
    _, _, m_and_cov, m0, P0, H = build_chirp_model(g(theta))
    return ekf(m_and_cov, H, Xi, m0, P0, dt, ys)[-1][-1]


opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
opt_vals, opt_state = opt_solver.run(init_theta)
opt_params = g(opt_vals)
print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

_, _, m_and_cov, m0, P0, H = build_chirp_model(opt_params)


@jax.jit
def filtering(measurements):
    return ekf(m_and_cov, H, Xi, m0, P0, dt, measurements)


@jax.jit
def smoothing(mfs, Pfs):
    return eks(m_and_cov, mfs, Pfs, dt)


# Trigger jit
_dummy = filtering(ys)
smoothing(_dummy[0], _dummy[1])

tic = time.time()
filtering_results = filtering(ys)
smoothing_results = smoothing(filtering_results[0], filtering_results[1])
print(f'Elapsed {time.time() - tic} seconds.')
