import pytest
import math
import jax
import jax.scipy
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from chirpgp.filters_smoothers import kf, rts
from chirpgp.models import posterior_cramer_rao
from chirpgp.tools import lti_sde_to_disc
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)


class TestFiltersSmoothers:

    def test_crlb_lgssm(self):

        ell, sigma = 1., 1.
        dt = 0.1
        T = 10

        A = jnp.array([[0., 1.],
                       [-3 / ell ** 2, -2 * math.sqrt(3) / ell]])
        B = jnp.array([0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5])

        F, Sigma = lti_sde_to_disc(A, B, dt)
        chol_sigma = jnp.linalg.cholesky(Sigma)

        Xi = 1.
        H = jnp.array([1., 0.])

        m0 = jnp.zeros((2,))
        P0 = jnp.asarray(np.diag([sigma ** 2, 3 / ell ** 2 * sigma ** 2]))
        chol_P0 = jnp.sqrt(P0)

        num_mcs = 1000000

        def simulate(key):
            def scan_body(carry, elem):
                x = carry
                rnd_x, rnd_y = elem
                x = F @ x + chol_sigma @ rnd_x
                y = jnp.dot(H, x) + math.sqrt(Xi) * rnd_y
                return x, (x, y)

            rnds_x = jax.random.normal(key, shape=(T, 2))
            key, _ = jax.random.split(key)
            rnds_y = jax.random.normal(key, shape=(T, ))
            key, _ = jax.random.split(key)
            x0 = m0 + chol_P0 @ jax.random.normal(key, shape=(2, ))
            _, (xs, ys) = jax.lax.scan(scan_body, x0, (rnds_x, rnds_y))
            return jnp.concatenate([x0[None, :], xs], axis=0), ys

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num_mcs)
        xss, yss = jax.vmap(simulate, in_axes=[0])(keys)

        kf_results = jax.vmap(kf, in_axes=[None, None, None, None, None, None, 0])(F, Sigma, H, Xi, m0, P0, yss)
        mfs, Pfs, _ = kf_results

        # Test if Pfs are the same for all MC runs
        i, k = np.random.randint(0, T - 1, 2)
        npt.assert_array_equal(Pfs[i], Pfs[k])

        Pfs = Pfs[0]

        # Test if E[(mfs - x)(mfs - x)^T] is all close to Pfs
        res = (mfs - xss[:, 1:, :])
        E = jnp.mean(jnp.einsum('...i,...j->...ij', res, res), axis=0)
        npt.assert_allclose(E, Pfs, atol=1e-1)

        # Test if CRLB is all close to Pfs
        def logpdf_transition(xt, xs):
            return jax.scipy.stats.multivariate_normal.logpdf(xt, F @ xs, Sigma)

        def logpdf_likelihood(yt, xt):
            return jax.scipy.stats.norm.logpdf(yt, H @ xt, math.sqrt(Xi))

        xss = jnp.transpose(xss, [1, 0, 2])
        yss = jnp.transpose(yss, [1, 0])

        j0 = jnp.linalg.inv(P0)
        js = posterior_cramer_rao(xss, yss, j0, logpdf_transition, logpdf_likelihood)
        npt.assert_allclose(jnp.linalg.inv(js), Pfs, atol=1e-12)
