# GW150914 frequency estimation

`gw_freq_estimate.py` shows how to apply our method to estimate the frequency of a gravitation wave signal GW150914 (the
very famous GW event in 2016). Please follow the instructions in what follows to run it.

1. Download the data in https://www.gw-openscience.org/events/GW150914/. Open this link, then save the data by clicking
   all the `click for DATA` for L1 and H1. I do not upload the data here because I am not certain about its license.
2. Save the data in folder `./data`. This folder should contain at least two files: `fig1-observed-H.txt`
   and `fig1-waveform-H.txt`.
3. Run `./gw_freq_estimate.py`.
