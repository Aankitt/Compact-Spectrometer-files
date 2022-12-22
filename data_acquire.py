import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
import seaborn as sns


# simulated probe range
# data_range_start = 975
# data_range_stop = 1775

# experimental probe range
# data_range_start = 600
# data_range_stop = 2048

# ## correlation range
data_range_start = 650
data_range_stop = 2000

def data_acquire(data, bin_no):
    # Selecting only the central range for the intensity and wavelength value.
    time = data.iloc[data_range_start:data_range_stop, 1]
    volts = data.iloc[data_range_start:data_range_stop, 3]

    time = time * 1e6
    # Changing the time values into the wavelength values
    wavelength_model = (1256 + 15.8138 * time +
                        5.5467 * time ** 2 -
                        1.0695 * time ** 3 +
                        0.02803 * time ** 4)

    wavelength = np.array(wavelength_model)
    # discretising the spectral channel in-terms of wavelength and volts
    wavelength_bin = np.array_split(wavelength, bin_no)
    volts = np.array(volts)
    volts_bin = np.array_split(volts, bin_no)
    return wavelength_bin, volts_bin


def data_acquire_raw(data):
    time = data.iloc[data_range_start:data_range_stop, 1]
    volts = data.iloc[data_range_start:data_range_stop, 3]

    time = time * 1e6
    wavelength_model = (1256 + 15.8138 * time +
                        5.5467 * time ** 2 -
                        1.0695 * time ** 3 +
                        0.02803 * time ** 4)

    wavelength = np.array(wavelength_model)
    volts = np.array(volts)
    return wavelength, volts


def gauss(x, a, mu, sig):
    # creating a gaussian function
    return a*np.exp(-(x-mu)**2/(2*sig**2))


def lorentzian(x, amp, center, width):
    return (amp*width**2)/((x-center)**2+width**2)
