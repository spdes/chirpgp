from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

with open('LICENSE', 'r') as f:
    lic = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='chirpgp',
    version='0.1.0',
    author='Zheng Zhao',
    author_email='zz@zabemon.com',
    description='Chirp instantaneous frequency estimation using SDE-GPs',
    keywords=['stochastic differential equations',
              'statistics',
              'signal processing',
              'Gaussian processes',
              'frequency estimation'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license=lic,
    url='https://github.com/spdes/chirpgp',
    download_url='https://github.com/spdes/chirpgp',
    packages=['chirpgp'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    python_requires='>=3.7',
    install_requires=requirements
)
