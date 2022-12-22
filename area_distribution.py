
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import variation
from scipy.optimize import curve_fit


fig, ax = plt.subplots(figsize=(2.62, 2), dpi=150)

labelfontsize = 6
axisfontsize = 7


area = pd.read_csv('file:///D:/Spectrometer_data/Revised/area.csv', delimiter=',')
distribution = pd.read_csv('file:///D:/Spectrometer_data/Revised/distribution.csv', delimiter=',')


location_point = area.iloc[:, 0]
area_px = area.iloc[:, 1]
radius_px = area.iloc[:, 2]
radius_um = area.iloc[:, 3]


first_closest = distribution.iloc[:, 2]
second_closest = distribution.iloc[:, 3]

print(first_closest)
sns.set_style("ticks")

# ax.hist([first_closest, second_closest], bins=12,
#          label=['First closest', 'Second closest'])


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2*standard_deviation ** 2))


bin_heights, bin_borders, _ = ax.hist(radius_um, bins=9, label='First closest')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
ax.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), linestyle='solid', linewidth=0.5, color='black')

# bin_heights, bin_borders, _ = ax.hist(second_closest, bins=20, label='Second closest', alpha=0.6)
# bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
# popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
#
# x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
# ax.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), linestyle='solid', linewidth=0.5, color='black')

plt.xlabel('Radius ($\u03BC$m)', fontsize=axisfontsize)
plt.ylabel('Counts', fontsize=axisfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
# ax.set_xticks([1, 1.5, 2, 2.5, 3])
plt.gcf().subplots_adjust(left=0.14, right=0.9, bottom=0.19)
# plt.legend(loc='upper right', fontsize=labelfontsize)
plt.show()
