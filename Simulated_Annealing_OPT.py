import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from Transmission_Matrix import transmission_matrix
from SVD import SVD
from data_acquire import gauss
from data_acquire import lorentzian
import seaborn as sns
from numpy.linalg import norm
from sklearn import metrics
from scipy.signal import chirp, find_peaks, peak_widths
from matplotlib.lines import Line2D


bin_number = 500
no_of_channels = 16
Transmission_matrix, Intensity, spectra_matrix, label, average_value = transmission_matrix(bin_number, no_of_channels)
diagonal, T_inv = SVD(Transmission_matrix, tolerance=1e-3)


Transmission_matrix = pd.DataFrame(Transmission_matrix, columns=label)
# Intensity = np.array(Intensity)

sns.set_style("ticks")

labelfontsize = 6
axisfontsize = 7

# fig, ax = plt.subplots(figsize=(4, 2), dpi=150)
# row_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
# x_label = ['1280', '1290', '1300', '1310', '1320', '1330', '1340']
# Plotting the speckle pattern
# fg = sns.heatmap(Intensity, yticklabels=row_labels, xticklabels=50)
# # ax.set_xticks([1280, 1290, 1300, 1310, 1320, 1330, 1340])
# plt.xlabel('Wavelength (nm)', fontsize=axisfontsize)
# plt.ylabel('Channel no.', fontsize=axisfontsize)
# plt.xticks(rotation=0, fontsize=labelfontsize)
# plt.yticks(fontsize=labelfontsize)
# plt.gcf().subplots_adjust(left=0.1, right=1, top=0.97, bottom=0.19)
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=labelfontsize)
# plt.show()

print(np.min(average_value), np.max(average_value))
# Creating a Gaussian Probe
x = np.linspace(np.min(average_value), np.max(average_value), bin_number, endpoint=False)
# x = np.array(average_value)
# mean = np.linspace(average_value[0], average_value[-1], 200)

probe_1 = lorentzian(x, 0.7, 1293, 0.1)
probe_2 = lorentzian(x, 1, 1300, 0.1)
probe_3 = lorentzian(x, 0.8, 1310, 0.1)
probe_4 = lorentzian(x, 0.5, 1319, 0.1)
probe_5 = lorentzian(x, 0.5, 1330, 0.1)


probe = probe_1 + probe_2 + probe_3 + probe_4 + probe_5

probe_value = probe / np.max(probe)

probe_intensity = Transmission_matrix @ probe

Reconstructed_spectra_1 = T_inv @ probe_intensity

Reconstructed_spectra_1 = Reconstructed_spectra_1 / np.max(Reconstructed_spectra_1)
# Reconstructed_spectra_2 = Reconstructed_spectra_2 / np.max(Reconstructed_spectra_2)

Reconstructed_spectra_1 = np.array(Reconstructed_spectra_1)
# Reconstructed_spectra_2 = np.array(Reconstructed_spectra_2)

probe_value = pd.DataFrame(probe_value, columns=['val'], index=label)
refWL_1 = probe_value.index.to_numpy().astype(float)

print(refWL_1)

# Creating non-negative value
most_negative = np.min(Reconstructed_spectra_1)
Reconstructed_spectra_1 = Reconstructed_spectra_1 - most_negative
Reconstructed_spectra_1 = Reconstructed_spectra_1 / np.max(Reconstructed_spectra_1)

fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)
ax.plot(refWL_1, probe_value, linestyle='solid', linewidth=1, color='green', label='Probe')
ax.plot(refWL_1, Reconstructed_spectra_1, linestyle='dashed', linewidth=1, color='red', label='Reconstructed')
# ax.plot(refWL_2, Reconstructed_spectra_2, linestyle='dashed', linewidth=1, color='blue', label='Reconstructed')
ax.set_xticks([1280, 1290, 1300, 1310, 1320, 1330, 1340])
# ax.set_xticks([1300, 1305, 1310, 1315, 1320, 1325])
plt.xlabel('Wavelength (nm)', fontsize=axisfontsize)
plt.ylabel('Normalized Intensity', fontsize=axisfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
plt.gcf().subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.17)
# plt.xlim(1300, 1330)
plt.legend(fontsize=labelfontsize,
           #loc='lower right'
           )
# plt.savefig(('sweep' + str(i) + '.png'))
plt.show()

mse = metrics.mean_squared_error(probe_value, Reconstructed_spectra_1)
rmse = np.sqrt(mse)
print(rmse)

# # # # # initializing simulated annealing algorithm
# # # # initial_spectra = np.array(Reconstructed_spectra)
# # # #
# # # # temperature_range = np.linspace(1000, 0, 100)
# # # #
# # # # for i in temperature_range:
# # # #     energy_i = np.copy(initial_spectra)
# # # #     # print(norm(probe_intensity - Transmission_matrix @ energy_i))
# # # #     for j in np.arange(len(initial_spectra)):
# # # #         scalar_value = random.uniform(0.5, 2)
# # # #         initial_energy = norm(probe_intensity - Transmission_matrix @ energy_i)
# # # #         amenable_spectra = np.copy(energy_i)
# # # #         amenable_spectra[j] = scalar_value * amenable_spectra[j]
# # # #         new_energy = norm(probe_intensity - Transmission_matrix @ amenable_spectra)
# # # #         delta_energy = new_energy - initial_energy
# # # #         prob = np.exp(-delta_energy / i)
# # # #
# # # #         # if delta_energy > 0:
# # # #             # print(new_energy, initial_energy, delta_energy, i, delta_energy / i, prob)
# # # #
# # # #         if delta_energy < 0 or prob > random.uniform(0, 1):
# # # #             initial_spectra[j] = amenable_spectra[j]
# # # #
# # # # initial_spectra = initial_spectra / np.max(initial_spectra)
# # # #
# # # # initial_spectra[initial_spectra < 0] = 0
# #

