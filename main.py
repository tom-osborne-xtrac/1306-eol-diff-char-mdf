from datetime import time
import tkinter as tk
from tkinter import filedialog
from types import NoneType
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll
import numpy as np
import os
import pandas as pd
import mdfreader
from scipy.ndimage import uniform_filter1d

# Set-ExecutionPolicy Unrestricted -Scope CurrentUser

Config = {
    "Debug": True,
}

def smooth(x, window_len, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    Source: SciPy Cookbook -  https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len:0:-1], x, x[-2:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int((window_len/2-1)):-int((window_len/2)+1)]


def set_axis(plots, axis, label, start, end, major, minor=None):
    """
    Function for setting plot label, axis major
    and minor ticks and formats the gridlines
    """
    for plot in plots:
        if major:
            major_ticks = np.arange(start, end + 1, major)
            if axis == 'x':
                plot.set_xlabel(label)
                plot.set_xlim([start, end])
                plot.set_xticks(major_ticks)
            elif axis == 'y':
                plot.set_ylabel(label)
                plot.set_ylim([start, end])
                plot.set_yticks(major_ticks)

        if minor:
            minor_ticks = np.arange(start, end + 1, minor)
            if axis == 'x':
                plot.set_xticks(minor_ticks, minor=True)
            elif axis == 'y':
                plot.set_yticks(minor_ticks, minor=True)

        if major and minor:
            plot.grid(which='both')
        plot.grid(which='minor', alpha=0.4)
        plot.grid(which='major', alpha=0.8)


# Select file for analysis
filePath = './data/17082022_1306-019-DEV_Loaded_EOL_004_PHASE 150.mdf'
if filePath == '':
    filePath = tk.filedialog.askopenfilename()
fileName = os.path.basename(filePath)

# Define channels
channels = [
    # Speed Channels
    'Cadet_IP_Speed',
    'WhlRPM_RL',
    'WhlRPM_RR',
    'InshaftN',
    'ClushaftN',
    'MaishaftN',
    'OutshaftN',
    
    # Torque Channels
    'Cadet_IP_Torque',
    'Cadet_OP_Torque_1',
    'Cadet_OP_Torque_2',

    # Oil system channels
    'Cadet_Oil_flow',
    'Cadet_Oil_Pres',
    'Cadet_Oil_Temp',

    # Misc channels
    'CadetPhase',
    'GearEngd'
]

# Load data
mdf_data = mdfreader.Mdf(
    filePath,
    channel_list = channels
)
mdf_data.resample(master_channel="t_71")

data = pd.DataFrame()

for channel in channels:
    chan_info = mdf_data.get_channel(channel)
    if chan_info is not None:
        time_chan = chan_info['master']
        data[time_chan] = mdf_data.get_channel_data(time_chan) - mdf_data.get_channel_data(time_chan).min()
        data[channel] = mdf_data.get_channel_data(channel)
    else:
        print("ERROR: No data found for channel:", channel)

data = data.rename(columns={'t_71': 'time'})

# Calculate sample rate
deltas = np.diff(data['time'], n=1)
sr = int(1 / (sum(deltas) / len(deltas)))

# Gear used
gears_hr = {
    1: "1st",
    2: "2nd",
    3: "3rd",
    4: "4th",
    5: "5th",
    6: "6th",
    7: "7th"
}
gear_ratios = {
    1: 12.803,
    2: 9.267,
    3: 7.058,
    4: 5.581,
    5: 4.562,
    6: 3.878,
    7: 3.435
}

actualGear = np.argmax(np.bincount(data['GearEngd']))
actualGear_hr = gears_hr[actualGear]

# Calculated channels
calc_IPTrqGradient = np.gradient(uniform_filter1d(data['Cadet_IP_Torque'] , size=int(sr)), edge_order=2) * 100
data['calc_IPTrqGradient'] = calc_IPTrqGradient.tolist()
data['calc_IPTrqGradient_smoothed'] = np.append(smooth(data['calc_IPTrqGradient'], sr + 1), 0.0)
data['calc_AxleTrqFromOutput'] = data['Cadet_OP_Torque_1'] + data['Cadet_OP_Torque_2']
data['calc_AxleTrqFromInput'] = data['Cadet_IP_Torque'] * gear_ratios[actualGear]
data['calc_LockTrq'] = data['Cadet_OP_Torque_1'] - data['Cadet_OP_Torque_2']
data['calc_OPSpeedDelta'] = np.append(smooth(data['WhlRPM_RL'] - data['WhlRPM_RR'], sr + 1), 0.0)

# TODO: Filter data
# TODO: Filter conditions

# LH Data
dataLH = data[data['calc_OPSpeedDelta'] > 0]
dataLH_filtered = dataLH[(abs(dataLH['calc_IPTrqGradient_smoothed']) < 1) & (abs(dataLH['calc_AxleTrqFromInput']) > 50)]

# RH Data
dataRH = data[data['calc_OPSpeedDelta'] < 0]
dataRH_filtered = dataRH[(abs(dataRH['calc_IPTrqGradient_smoothed']) < 1) & (abs(dataRH['calc_AxleTrqFromInput']) > 50)]

# Set points for torque analysis graphs
set_points_x = [-800, -400, -200, -100, 0, 100, 200, 400, 800]
set_points = []
for v in set_points_x:
    pair = [(v, 0), (v, 1000)]
    set_points.append(pair)
plot_set_points = matcoll.LineCollection(set_points)

# Plot raw data
fig, ax = plt.subplots(3)
axSecondary0 = ax[0].twinx()
axSecondary1 = ax[1].twinx()
axSecondary0.plot(
    data['time'],
    data['calc_IPTrqGradient_smoothed'],
    color='orange',
    label='IP Torque Gradient Smoothed',
    marker=None    
)
ax[0].plot(
    data['time'],
    data['Cadet_IP_Torque'],
    color='green',
    label='IP Torque',
    marker=None    
)

ax[1].plot(
    data['time'],
    data['calc_AxleTrqFromInput'],
    color='blue',
    label='Axle Torque',
    marker=None    
)
ax[1].plot(
    data['time'],
    data['Cadet_OP_Torque_1'],
    color='red',
    label='LH OP Torque',
    marker=None    
)
ax[1].plot(
    data['time'],
    data['Cadet_OP_Torque_2'],
    color='orange',
    label='RH OP Torque',
    marker=None    
)
axSecondary1.plot(
    data['time'],
    data['WhlRPM_RL'],
    color='grey',
    label='RH OP Torque',
    marker=None    
)
axSecondary1.plot(
    data['time'],
    data['WhlRPM_RR'],
    color='darkgrey',
    label='RH OP Torque',
    marker=None    
)
axSecondary1.plot(
    data['time'],
    data['calc_OPSpeedDelta'],
    color='black',
    label='OP Speed Delta [rpm]',
    marker=None
)

ax[2].plot(
    dataLH['time'],
    dataLH['calc_AxleTrqFromInput'],
    color='blue',
    label='Axle Torque',
    marker=None    
)
ax[2].plot(
    dataRH['time'],
    dataRH['calc_AxleTrqFromInput'],
    color='green',
    label='Axle Torque',
    marker=None    
)
ax[2].scatter(
    dataLH_filtered['time'],
    dataLH_filtered['calc_AxleTrqFromInput'],
    color='magenta',
    label='Axle Torque',
    marker=".",
    zorder=1
)
ax[2].scatter(
    dataRH_filtered['time'],
    dataRH_filtered['calc_AxleTrqFromInput'],
    color='yellow',
    label='Axle Torque',
    marker=".",
    zorder=1
)
set_axis(ax, 'x', 'Time [s]', 0, data['time'].max(), 50, 5)
set_axis([ax[0]], 'y', 'Torque [Nm]', -200, 200, 50, 10)
set_axis([ax[1], ax[2]], 'y', 'Torque [Nm]', -1000, 1000, 250, 50)
axSecondary0.set_ylim([-10, 10])
axSecondary1.set_ylim([-100, 100])
ax[0].set_title("Input Torque & Input Torque Delta", loc='left')
fig.suptitle(f'Diff Test Overview - 3rd Gear', fontsize=16)

if Config["Debug"]:
    print(
        """
        =============
        DEBUG
        =============
        """
    )
    print(filePath, "\n\n")
    print(data.head())
    print(
        """
        =============
        END DEBUG
        =============
        """
    )
    
plt.subplots_adjust(
    left=0.05,
    bottom=0.07,
    right=0.955,
    top=0.9,
    wspace=0.2,
    hspace=0.4
)
plt.show()
