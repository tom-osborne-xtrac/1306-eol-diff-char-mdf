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


# Select file for analysis
filePath = 'N:/Projects/1306/XT REPORTS/XT-13944 - Loaded Test Rig - EoL Testing/SAMPLE DATA/Development data/2022-08-17 - EoL Development/17082022_1306-019-DEV_Loaded_EOL_004_PHASE 150.mdf'
if filePath == '':
    filePath = tk.filedialog.askopenfilename()
fileName = os.path.basename(filePath)

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
    'Cadet_Oil_Temp'

    # Misc channels
    'CadetPhase',
    'GearEngd'
]

# Load data
mdf_data = mdfreader.Mdf(
    filePath,
    channel_list = channels
)

channel_list = []
for channel in channels:
    chan_info = mdf_data.get_channel(channel)
    if chan_info is not None:
        time_chan = chan_info['master']
        time_data = mdf_data.get_channel_data(time_chan) - mdf_data.get_channel_data(time_chan).min()
        channel_dict = { 
            'name': channel,
            'time': time_data,
            'data': mdf_data.get_channel_data(channel)
        }        
        channel_list.append(channel_dict)
    else:
        print("ERROR: No data found for channel:", channel)

print(type(channel_list), channel_list)
data = pd.DataFrame(channel_list)
print(type(data), data)

# Calculate sample rate
deltas = np.diff(data['Cadet_IP_Speed']['time'], n=1)
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

actualGear = np.argmax(np.bincount(data['GearEngd']['data']))
actualGear_hr = gears_hr[actualGear]

data['calc_IPTrqGradient'] = {
    'time': data['Cadet_IP_Torque']['time'],
    'data': np.gradient(uniform_filter1d(data['Cadet_IP_Torque']['data'] , size=int(sr)), edge_order=2) * 10
}
print("(data['calc_IPTrqGradient']", type(data['calc_IPTrqGradient']))
print(data['calc_IPTrqGradient'])

data['calc_IPTrqGradient_smoothed'] = smooth(data['calc_IPTrqGradient'][1], sr + 1)

# Axle Torque Calculations
data['calc_AxleTrqFromOutput'] = data['Cadet_OP_Torque_1']['data'] + data['Cadet_OP_Torque_2']['data']
data['calc_AxleTrqFromInput'] = data['Cadet_IP_Torque']['data'] * gear_ratios[actualGear]
data['calc_LockTrq'] = data['Cadet_OP_Torque_1']['data'] - data['Cadet_OP_Torque_2']['data']
data['calc_OPSpeedDelta'] = smooth(data['WhlRPM_RL']['data'] - data['WhlRPM_RR']['data'], sr + 1)

# Filter data
# Filter conditions

# Set points for torque analysis graphs
set_points_x = [-800, -400, -200, -100, 0, 100, 200, 400, 800]
set_points = []
for v in set_points_x:
    pair = [(v, 0), (v, 1000)]
    set_points.append(pair)
plot_set_points = matcoll.LineCollection(set_points)
print(plot_set_points)

# Plot raw data
fig, ax = plt.subplots(3)
axSecondary = ax[0].twinx()
axSecondary.plot(
    data['calc_IPTrqGradient'][0][:len(time_data) - 1],
    data['calc_IPTrqGradient_smoothed'][1],
    color='orange',
    label='IP Torque Gradient Smoothed',
    marker=None    
)
ax[0].plot(
    data['Cadet_IP_Torque'][0],
    data['Cadet_IP_Torque'][1],
    color='green',
    label='IP Torque',
    marker=None    
)
ax[0].set_title("Input Torque & Input Torque Delta", loc='left')
ax[0].grid()
ax[0].legend(loc=2)
ax[0].set_xlim([0, time_data.max()])
ax[0].set_xlabel("Time [s]")
ax[0].set_ylim([-200, 200])
ax[0].set_ylabel("Torque [Nm]")
axSecondary.set_ylim([-10, 10])

ax[1].plot(
    time_data,
    mdf_data['calc_AxleTrqFromInput']['data'],
    color='blue',
    label='IP Torque',
    marker=None    
)
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
    print('Time Offset:', type(time_offset), time_offset)
    print("Sample rate:", sr)
    print("Gear:", actualGear, actualGear_hr)
    print("Timedata Type: ", type(time_data))
    print(
        """
        =============
        END DEBUG
        =============
        """
    )
    
plt.show()
