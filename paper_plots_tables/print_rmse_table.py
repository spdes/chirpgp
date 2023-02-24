"""
This generated Table 1 in the paper.
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

path = '../tetralith/results/'

num_mcs = 100

# Print and plot for single chirp
methods = ['hilbert', 'mean_spectrogram', 'mle_polynomial', 'anf',
           'lascala_ekfs_mle', 'lascala_ghfs_mle', 'fastf0nls', 'fhc', 'kpt_mle',
           'ekfs_mle', 'ghfs_mle', 'ckfs_mle', 'cd_ekfs_mle', 'cd_ghfs_mle']
method_labels = ['Hilbert', 'Spectrogram', 'Polynomial', 'ANF',
                 'EKFS on (2)', 'GHFS on (2)', 'FastF0NLS', 'FHC', 'KPT',
                 r'\textbf{EKFS}', r'\textbf{GHFS}', r'\textbf{CKFS}', r'\textbf{CD-EKFS}', r'\textbf{CD-GHFS}']
mags = ['const', 'damped', 'ou']


plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axs = plt.subplots(nrows=3, ncols=1, sharex='col', sharey='row', figsize=(17, 6))

for row, mag in enumerate(mags):
    axs[row].set_ylabel('RMSE')
    all_rmses = []
    table = PrettyTable()
    table.field_names = ['Method', 'Mean', 'Median', 'Min', 'NaNs']

    for col, method in enumerate(methods):
        rmses = np.array([])
        num_nans = 0
        for mc in range(num_mcs):
            if method in ['fhc']:
                file_name = path + f'{method}_{mag}_{mc}.mat'
                rmse = np.squeeze(scipy.io.loadmat(file_name)['rmse'])
            else:
                file_name = path + f'{method}_{mag}_{mc}.npz'
                rmse = np.load(file_name)['rmse']
            if np.isfinite(rmse):
                rmses = np.append(rmses, rmse)
            else:
                num_nans += 1

        table.add_row([method_labels[col],
                       f'{np.mean(rmses) * 10:.5f} ± {np.std(rmses) * 10:.5f}',
                       f'{np.median(rmses) * 10:.5f}',
                       f'{np.min(rmses) * 10:.5f}',
                       f'{num_nans}'])
        all_rmses.append(rmses)

        # Scatter rmse points
        np.random.default_rng(seed=666)
        _dummy = np.random.uniform(col - 0.1, col + 0.1, rmses.size)
        axs[row].scatter(_dummy, rmses, s=3, c='tab:purple', edgecolors='none')

    print(f'RMSEs with mag {mag}')
    print(table)

    axs[row].boxplot(all_rmses, positions=np.arange(len(methods)), sym='', vert=True, whis=1.5, widths=0.4,
                     meanline=True, showmeans=True,
                     labels=method_labels,
                     whiskerprops={'linewidth': 1.5},
                     capprops={'linewidth': 1.5},
                     boxprops={'linewidth': 1.5},
                     medianprops={'c': 'tab:blue', 'linewidth': 1.5},
                     meanprops={'c': 'tab:orange', 'linewidth': 1.5, 'linestyle': ':'})
    axs[row].grid(linestyle='--', alpha=0.3, which='both', axis='y')

# Log scale
for ax in axs:
    ax.set_yscale('log')

# Textbox
axs[0].annotate(r'$\alpha(t)=1$', xy=(8.5, 1), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))
axs[1].annotate(r'$\alpha(t) = \mathrm{e}^{-0.3 \, t}$', xy=(8.5, 3), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))
axs[2].annotate(r'$\alpha(t)$ random', xy=(8.5, 7), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))

plt.tight_layout(pad=0.1)
# plt.savefig('/home/zgbkdlm/Papers/chirp-estimation-paper/figs/rmse-statistics.pdf')
plt.show()

# Print and plot for harmonic chirp
methods = ['harmonic_fastf0nls', 'harmonic_fhc', 'harmonic_kpt_mle',
           'harmonic_ekfs_mle', 'harmonic_ckfs_mle']
method_labels = ['FastF0NLS', 'FHC', 'KPT',
                 r'\textbf{EKFS}', r'\textbf{CKFS}']

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axs = plt.subplots(nrows=3, ncols=1, sharex='col', sharey='row', figsize=(17, 6))

for row, mag in enumerate(mags):
    axs[row].set_ylabel('RMSE')
    all_rmses = []
    table = PrettyTable()
    table.field_names = ['Method', 'Mean', 'Median', 'Min', 'NaNs']

    for col, method in enumerate(methods):
        rmses = np.array([])
        num_nans = 0
        for mc in range(num_mcs):
            if method in ['harmonic_fhc', 'harmonic_kpt']:
                file_name = path + f'{method}_{mag}_{mc}.mat'
                rmse = scipy.io.loadmat(file_name)['rmse']
            else:
                file_name = path + f'{method}_{mag}_{mc}.npz'
                rmse = np.load(file_name)['rmse']
            if np.isfinite(rmse):
                rmses = np.append(rmses, rmse)
            else:
                num_nans += 1

        table.add_row([method_labels[col],
                       f'{np.mean(rmses) * 10:.5f} ± {np.std(rmses) * 10:.4f}',
                       f'{np.median(rmses) * 10:.5f}',
                       f'{np.min(rmses) * 10:.5f}',
                       f'{num_nans}'])
        all_rmses.append(rmses)

        # Scatter rmse points
        np.random.default_rng(seed=666)
        _dummy = np.random.uniform(col - 0.1, col + 0.1, rmses.size)
        axs[row].scatter(_dummy, rmses, s=3, c='tab:purple', edgecolors='none')

    print(f'Harmonic RMSEs with mag {mag}')
    print(table)

    axs[row].boxplot(all_rmses, positions=np.arange(len(methods)), sym='', vert=True, whis=1.5, widths=0.4,
                     meanline=True, showmeans=True,
                     labels=method_labels,
                     whiskerprops={'linewidth': 1.5},
                     capprops={'linewidth': 1.5},
                     boxprops={'linewidth': 1.5},
                     medianprops={'c': 'tab:blue', 'linewidth': 1.5},
                     meanprops={'c': 'tab:orange', 'linewidth': 1.5, 'linestyle': ':'})
    axs[row].grid(linestyle='--', alpha=0.3, which='both', axis='y')

# Log scale
for ax in axs:
    ax.set_yscale('log')

# Textbox
axs[0].annotate(r'$\alpha(t)=1$', xy=(8.5, 1), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))
axs[1].annotate(r'$\alpha(t) = \mathrm{e}^{-0.3 \, t}$', xy=(8.5, 3), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))
axs[2].annotate(r'$\alpha(t)$ random', xy=(8.5, 7), xycoords='data',
                fontsize=18, bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))

plt.tight_layout(pad=0.1)
plt.show()
