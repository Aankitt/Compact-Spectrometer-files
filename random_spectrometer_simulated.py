import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from Transmission_Matrix import transmission_matrix
from SVD import SVD
from data_acquire import gauss
import seaborn as sns
from numpy.linalg import norm
from scipy.signal import chirp, find_peaks, peak_widths
from sklearn import metrics
from matplotlib.lines import Line2D
from data_acquire import lorentzian


bin_number = 500
no_of_channels = 16
Transmission_matrix, Intensity, spectra_matrix, label, average_value = transmission_matrix(bin_number, no_of_channels)
diagonal, T_inv = SVD(Transmission_matrix, tolerance=1e-3)

Transmission_matrix = pd.DataFrame(Transmission_matrix, columns=label)

sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)

print(np.max(average_value), np.min(average_value))
x = np.linspace(np.min(average_value), np.max(average_value), bin_number, endpoint=False)
amplitude = 1
mean = 1312
sigma = [0.1, 1, 4]
spectra_values = []
probe_values = []
for i in np.arange(len(sigma)):
    probe = lorentzian(x, amplitude, mean, sigma[i])

    probe_value = probe / probe.max()

    probe_intensity = Transmission_matrix @ probe

    Reconstructed_spectra = T_inv @ probe_intensity

    Reconstructed_spectra = Reconstructed_spectra / np.max(Reconstructed_spectra)

    Reconstructed_spectra = np.array(Reconstructed_spectra)

    # peaks, _ = find_peaks(probe_value)
    # results_half = peak_widths(probe_value, peaks, rel_height=0.5)
    # width = results_half[0]

    # mse = metrics.mean_squared_error(probe_value, Reconstructed_spectra_truncated)
    # rmse = np.sqrt(mse)
    # print(rmse)

    probe_value = pd.DataFrame(probe_value, columns=['val'], index=label)

    refWL = probe_value.index.to_numpy().astype(float)

    ref_spectra = probe_value['val'].to_numpy()

    if i <= 1:
        most_negative = np.min(Reconstructed_spectra)
        Reconstructed_spectra = Reconstructed_spectra - most_negative
        Reconstructed_spectra = Reconstructed_spectra / np.max(Reconstructed_spectra)
    else:
        pass

    ax.plot(refWL, probe_value, linestyle='solid', linewidth=1)
    ax.plot(refWL, Reconstructed_spectra, linestyle='dashed', linewidth=1)

labelfontsize = 6
axisfontsize = 7

plt.xlabel('Wavelength (nm)', fontsize=axisfontsize)
plt.ylabel('Normalized Intensity', fontsize=axisfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
plt.gcf().subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.17)
custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='solid'),
                Line2D([0], [0], color='black', lw=1, linestyle='dashed')]
plt.legend(custom_lines, ['Probe', 'Reconstructed'], fontsize=labelfontsize)
plt.show()




# peaks, _ = find_peaks(Reconstructed_spectra_truncated)
# results_half = peak_widths(Reconstructed_spectra_truncated, peaks, rel_height=0.5)
# width = results_half[0]
# print(np.max(width))
# plt.plot(Reconstructed_spectra_truncated)
# plt.plot(peaks, Reconstructed_spectra_truncated[peaks], "x")
# plt.hlines(*results_half[1:], color="C2")
# # plt.hlines(*results_full[1:], color="C3")
# plt.xticks(ticks=np.arange(0, bin_number, step=1), labels=average_value)
# ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# plt.show()
