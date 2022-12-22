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
import seaborn as sns
from Simpson import find_area
from Simpson import create_intensity_matrix
from SVD import SVD
from scipy.stats import multivariate_normal
from data_acquire import gauss
from numpy import random
from sklearn import metrics

# creating a transmission matrix by extracting experimental data on at a time.


def transmission_matrix(bin_number, number_of_channels):
    average_bin = []
    for i in np.arange(1, 17, 1):
        data_1 = pd.read_csv('file:///D:/Spectrometer_data/chip_1/sample_1/channel%20' + str(i) + '_1.1.1.1.B.txt',
                             delimiter=',', skiprows=45)

    # Finding the average value of each bin
        wavelength_bin, volts_bin = data_acquire(data_1, bin_number)
        wavelength, volts = data_acquire_raw(data_1)
        average_value_per_bin = []
        for j in np.arange(bin_number):
            average = np.average(volts_bin[j])
            average_value_per_bin.append(average)
        average_bin.append(average_value_per_bin)
    # Creating the intensity matrix
    area_matrix = np.zeros((number_of_channels, bin_number))
    for i in np.arange(number_of_channels):
        for j in np.arange(bin_number):
            area_matrix[i][j] = average_bin[i][j]

# Creating a string value and float values for x-labeling for plots
    average_value_str = []
    average_value = []

    for i in np.arange(bin_number):

        y = np.average(wavelength_bin[i])
        x = str((y))

        average_value.append(y)
        average_value_str.append(x)

    labels = average_value_str

    intensity_matrix = pd.DataFrame(area_matrix, columns=labels)

    column_sum = intensity_matrix.sum()
    column_sum = np.array(column_sum)

    norm_matrix = np.zeros((len(intensity_matrix.axes[0]), len(intensity_matrix.axes[1])))

    for i in np.arange(len(intensity_matrix.axes[0])):
        for j in np.arange(len(intensity_matrix.axes[1])):
            norm_matrix[i][j] = intensity_matrix.iloc[i, j] / column_sum[j]

    norm_matrix = pd.DataFrame(norm_matrix, columns=labels)

    # reference waveguide data call and creating spectra matrix
    data_ref1_wg = pd.read_csv('file:///D:/Spectrometer_data/chip_1/ref_sample/reference%20wg%20channel_2.1.1.1.B.txt',
                               delimiter=',', skiprows=45)
    wavelength_ref_bin, volts_ref_bin = data_acquire(data_ref1_wg, bin_number)

    # Compute the area using the composite Simpson's rule.
    average_value_per_bin_ref = []
    for j in np.arange(bin_number):
        average_ref = np.average(volts_ref_bin[j])
        average_value_per_bin_ref.append(average_ref)

    spectral_intensities = np.array(average_value_per_bin_ref)
    spectral_intensities_raw = np.array(average_value_per_bin_ref)
    spectral_intensities = np.array(spectral_intensities)
    spectral_intensities = np.float64(spectral_intensities)
    spectral_intensities = np.reciprocal(spectral_intensities)
    spectra_matrix = np.diag(spectral_intensities)

    # Creating transmission matrix with SVD or pseudo inverse
    t_matrix = area_matrix @ spectra_matrix

    return t_matrix, intensity_matrix, spectral_intensities_raw, labels, average_value