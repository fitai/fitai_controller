from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from functions import method1, calc_vel2
from load_data import load_file
from processing.filters import simple_highpass
from test import find_threshold

# Should use known weight for demo
weight = 50  # kg
fs = 100.  # Sampling rate (default = 100Hz)
bin_size = 1.  # Number of samples to collect before processing the signal (float)

#: Step 1 - load in data
data_folder = '/Users/kyle/PycharmProjects/FitAI/data_files/'
#: NOTE: File 1 has 50 reps
# file1 = 'AccData_12_02_2015-15_06_19.csv'
#: File 2 has 30 reps
# file2 = ''
# NEW DATA
file1 = 'Tim_OHP_1.csv'

#: Switch between data files here
filename = file1

thresh = find_threshold()

col_names = ['timestamp', 'x', 'y', 'z']
data = load_file(join(data_folder, filename), names=col_names, skiprows=1)
# Extract data from file contents
dat = data[['x', 'y', 'z']]
rms_raw = [np.sqrt(np.mean(dat.iloc[i].x**2 + dat.iloc[i].y**2 + dat.iloc[i].z**2)) for i, _ in enumerate(tuple(dat.x))]
a_rms = list()
#: Implement the simple high pass filter
for i in range(len(rms_raw)):
    if i > 0:
        a = simple_highpass(rms_raw[i], rms_raw[i-1], a_rms[-1], fc=1., fs=fs)
    else:
        a = 0
    a_rms.append(a)

a_rms = np.array(a_rms)
v_euler = calc_vel2(a_rms, scale=bin_size, fs=fs)
pwr_rms = a_rms * weight * v_euler

pwr_imp, pwr_thresh, pwr_reps = method1(pwr_rms, calc_offset=0, scale=1.)

plt.figure()
plt.plot(pwr_rms, color='black')
# plt.axhline(y=pwr_thresh, color='blue', linestyle='--')
plt.axhline(y=thresh, color='red', linestyle='--')
plt.title('Tim DL 45lbs')
