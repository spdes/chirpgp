"""
Generate 1000 independent random keys
"""

import jax
import numpy as np

key = jax.random.PRNGKey(999)

keys = jax.random.split(key, 1000)

np.save('./rnd_keys.npy', np.asarray(keys))
