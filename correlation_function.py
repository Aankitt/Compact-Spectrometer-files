import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Transmission_Matrix import transmission_matrix
from SVD import SVD
from data_acquire import gauss
import seaborn as sns
from numpy.linalg import norm


bin_number = 1350
no_of_channels = 16
Transmission_matrix, Intensity_matrix, spectra_matrix, label, average_value = transmission_matrix(bin_number, no_of_channels)

column_sum = Intensity_matrix.sum()
column_sum = np.array(column_sum)

norm_matrix = np.zeros((len(Intensity_matrix.axes[0]), len(Intensity_matrix.axes[1])))

for i in np.arange(len(Intensity_matrix.axes[0])):
    for j in np.arange(len(Intensity_matrix.axes[1])):
        norm_matrix[i][j] = Intensity_matrix.iloc[i, j] / column_sum[j]

norm_matrix = pd.DataFrame(norm_matrix, columns=label)
norm_matrix = Intensity_matrix

sns.set_style("ticks")

labelfontsize = 6
axisfontsize = 7
#
# fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)
# # Plotting the speckle pattern
# fg = sns.heatmap(norm_matrix, cbar=False, xticklabels=150, yticklabels=5)
# plt.xlabel('Wavelength (nm)', fontsize=axisfontsize)
# plt.ylabel('Channel no.', fontsize=axisfontsize)
# ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
# plt.gcf().subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.17)
# plt.xticks(rotation=0)
# # plt.gcf().subplots_adjust(bottom=0.18)
# plt.show()

store1 = []
store1_average = []
store2 = []
store2_average = []
store3 = []
store3_average = []

wavelength_shift = 300
delta_lambda = np.arange(0, wavelength_shift, 1)

Transmission_matrix = pd.DataFrame(Transmission_matrix)
sum_column = Transmission_matrix.sum()
norm_matrix = Transmission_matrix

for k in delta_lambda:
    for i in np.arange(no_of_channels):
        for j in np.arange(len(Intensity_matrix.columns)):
            if j <= 1050:
                numerator = norm_matrix.iloc[i, j] * norm_matrix.iloc[i, j + k]
                denominator_1 = norm_matrix.iloc[i, j]
                denominator_2 = norm_matrix.iloc[i, j + k]
                store1.append(numerator)
                store2.append(denominator_1)
                store3.append(denominator_2)
            else:
                break
            numerator = 0
            denominator_1 = 0
            denominator_2 = 0
    store1_avg = np.average(store1)
    store1_average.append(store1_avg)
    store2_avg = np.average(store2)
    store2_average.append(store2_avg)
    store3_avg = np.average(store3)
    store3_average.append(store3_avg)
    store1 = []
    store2 = []
    store3 = []

correlation = []
correlation_norm = []

for i in np.arange(len(store1_average)):
    if i == 0:
        corr_norm = store1_average[i] / (store2_average[i] * store3_average[i]) - 1
        correlation_norm.append(corr_norm)
    else:
        corr = store1_average[i] / (store2_average[i] * store3_average[i]) - 1
        correlation.append(corr)

correlation = correlation/correlation_norm[0]
print(correlation)




fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)
delta = np.linspace(0, 15.4, 299)
# plt.plot(delta_lambda[1:], correlation)
ax.plot(delta, correlation, linestyle='solid', linewidth=0.9, color='black')
plt.xlabel('$\lambda$ (nm)', fontsize=axisfontsize)
plt.ylabel('Normalized C($\lambda$)', fontsize=axisfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
plt.text(10, 0.5, '$\delta\lambda$ = 3.2 nm', fontsize=labelfontsize)
plt.gcf().subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.17)
plt.show()

