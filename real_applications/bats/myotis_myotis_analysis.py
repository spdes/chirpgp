"""
Analysis the social call of a bat Myotis myotis.

References
----------
Guido Pfalzer, Inter- und intraspezifische Variabilität der Soziallaute heimischer Fledermausarten, 2002.

See Also
--------
See also the English translation "Inter- and intra-specific variability of social calls from native bat species"
by Yann Gager.

Guido Pfalzer and Jurgen Kusch. Structure and variability of bat social calls: implications for specificity and
individual recognition, 2003.

Data download
-------------
http://www.batcalls.com/.
"""
import jax
import time
import numpy as np
import scipy.signal
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
from chirpgp.models import g, build_harmonic_chirp_model
from chirpgp.filters_smoothers import sgp_filter, sgp_smoother
from chirpgp.quadratures import gaussian_expectation, SigmaPoints

jax.config.update("jax_enable_x64", True)

# Load data
fs, sound = scipy.io.wavfile.read('Myotis_myotis_2_o.wav')

# Times
dt = 1 / fs
T = sound.shape[0]
ts = jnp.linspace(dt, dt * T, T)

# Crop the interval of interests
ys = sound[209210:234544]
ts = ts[209210:234544]

# Standardise the data
ys = ys - np.mean(ys)
ys = ys / np.max(np.abs(ys))

# Number of harmonics
num_harmonics = 4

# Scale the frequency state for numerical stability
freq_scale = 10000

# Measurements and measurement noise variance
ys = jnp.asarray(ys)
Xi = 0.0001

# Create the model and the filters and smoothers
params = jnp.array([0.1, 1., 1., 0.2, 10., 2.])
_, _, m_and_cov, m0, P0, H = build_harmonic_chirp_model(params, num_harmonics=num_harmonics, freq_scale=freq_scale)

sgps = SigmaPoints.cubature(d=2 * num_harmonics + 2)


@jax.jit
def filtering(measurements):
    return sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, measurements)


@jax.jit
def smoothing(mfs, Pfs):
    return sgp_smoother(m_and_cov, sgps, mfs, Pfs, dt)


# Trigger jit compilation
_dummy = filtering(ys)
smoothing(_dummy[0], _dummy[1])

# Do the estimation
tic = time.time()
filtering_results = filtering(ys)
smoothing_results = smoothing(filtering_results[0], filtering_results[1])
toc = time.time()
print(f'Our method takes {toc - tic} seconds.')
estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, -2],
                                            chol_Ps=jnp.sqrt(smoothing_results[1][:, -2, -2]),
                                            func=g, force_shape=True)[:, 0] * freq_scale

# Plot
plt.rcParams.update({
    'figure.figsize': (7.32, 5.76),
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})
yscale = 1000

# Plot the signal
plt.plot(ts, ys)
plt.xlabel('Time')
plt.ylabel('Signal of the bat call')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.show()

# Plot the spectrogram. This might take a while
nperseg = 400
noverlap = nperseg - 1
tic = time.time()
freqs, sp_ts, Sxx = scipy.signal.spectrogram(ys, fs, nperseg=nperseg, noverlap=noverlap, mode='psd', nfft=1024)
toc = time.time()
print(f'The spectrogram takes {toc - tic} seconds.')
sp_ts = sp_ts + ts[0]
spectrogram = plt.pcolormesh(sp_ts, freqs / yscale, np.log10(Sxx), cmap=plt.cm.ocean, shading='gouraud')
plt.colorbar(spectrogram, pad=0.02)

# Plot the estimated frequencies
f, = plt.plot(ts, estimated_freqs_mean / yscale, c='black', linewidth=1)
for k in range(2, num_harmonics + 1):
    h, = plt.plot(ts, k * estimated_freqs_mean / yscale, c='black', linewidth=1, linestyle='--')

plt.xlim(sp_ts.min(), sp_ts.max())
plt.ylim(0, 100)
plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (second)')
plt.legend(handles=[f, h], labels=['Fundamental frequency', 'harmonics'], edgecolor=None, framealpha=0.5)
plt.tight_layout(pad=0.1)
plt.savefig(f'myotis.png', transparent=True)
plt.show()
