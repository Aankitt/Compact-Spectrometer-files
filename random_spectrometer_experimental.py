import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy.linalg import svd
from scipy.linalg import inv
from scipy.linalg import norm
from scipy.integrate import simps
from data_acquire import data_acquire
from data_acquire import data_acquire_raw
from Transmission_Matrix import transmission_matrix
import seaborn as sns
from Simpson import find_area
from Simpson import create_intensity_matrix
from SVD import SVD
from scipy.stats import multivariate_normal
from data_acquire import gauss
from numpy import random
from sklearn import metrics


bin_number = 1100
number_of_channels = 16
Transmission_matrix, Intensity, spectra_matrix, label, average_value = transmission_matrix(bin_number, number_of_channels)
diagonal, T_inv = SVD(Transmission_matrix, tolerance=1e-1)

fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)
# Plotting the speckle pattern
fg = sns.heatmap(Transmission_matrix, cbar=True, xticklabels=150, yticklabels=5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Channel no.')
plt.xticks(rotation=0)
plt.gcf().subplots_adjust(bottom=0.18)
plt.show()

print(average_value[2] - average_value[3], average_value[5] - average_value[6])

# sns.set_theme(context='notebook', font='sans-serif', style="white", font_scale=1, rc=None)
sns.set_style("ticks")

# Extracting the experimental data taken from next identical chip.
average_bin_2 = []
for i in np.arange(1, 17, 1):
    data_2 = pd.read_csv('file:///D:/Spectrometer_data/chip_2/sample_1/channel%20' + str(i) + '_1.1.1.1.B.txt',
                         delimiter=',', skiprows=45)

# Finding the average value of each bin
    wavelength_bin, volts_bin = data_acquire(data_2, bin_number)
    wavelength, volts = data_acquire_raw(data_2)
    average_value_per_bin = []
    for j in np.arange(bin_number):
        average = np.average(volts_bin[j])
        average_value_per_bin.append(average)
    average_bin_2.append(average_value_per_bin)

# Creating the intensity matrix
average_matrix_2 = np.zeros((number_of_channels, bin_number))
for i in np.arange(number_of_channels):
    for j in np.arange(bin_number):
        average_matrix_2[i][j] = average_bin_2[i][j]

Intensity_matrix_2 = pd.DataFrame(average_matrix_2, columns=label)

# Reconstructing the original laser signal
Reconstructed = T_inv @ Intensity_matrix_2
Reconstructed = np.array(Reconstructed)
Reconstructed = np.diagonal(Reconstructed)
Reconstructed = Reconstructed / np.max(Reconstructed)
# Reconstructed[Reconstructed < 0] = 0
Reconstructed = pd.DataFrame(Reconstructed, columns=['val'], index=label)

# normalizing
average_value_per_bin_ref = spectra_matrix / np.max(spectra_matrix)
ref_spectra = pd.DataFrame(average_value_per_bin_ref, columns=['val'], index=label)
refWL = ref_spectra.index.to_numpy().astype(float)
reconWL = Reconstructed.index.to_numpy().astype(float)

ref_spectra = ref_spectra['val'].to_numpy()
Reconstructed = Reconstructed['val'].to_numpy()
# print(ref_spectra)
labelfontsize = 6
axisfontsize = 7

# plotting the graphs
fig, ax = plt.subplots(figsize=(4, 2), dpi=150)
ax.plot(refWL, ref_spectra, linewidth=1.0, color='green', label='Reference', linestyle='solid')
ax.plot(reconWL, Reconstructed, linestyle='dashed', linewidth=1, color='red', label='Reconstructed')
ax.set_xticks([1280, 1290, 1300, 1310, 1320, 1330, 1340])

plt.xlabel('Wavelength (nm)', fontsize=axisfontsize)
plt.ylabel('Normalized Intensity', fontsize=axisfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
plt.gcf().subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.17)
plt.legend(fontsize = labelfontsize)
plt.show()

mse = metrics.mean_squared_error(ref_spectra, Reconstructed)
rmse = np.sqrt(mse)
print(rmse)

# confusion_matrix = metrics.confusion_matrix(ref_spectra, Reconstructed)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# cm_display.plot()
# plt.show()