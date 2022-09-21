from datetime import time
import tkinter as tk
from tkinter import filedialog
from types import NoneType
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll
import numpy as np
import os
import pandas as pd
import mdfreader
from scipy.ndimage import uniform_filter1d
import config
from utils import *

# Set-ExecutionPolicy Unrestricted -Scope CurrentUser

# Select file for analysis
filePath = './data/17082022_1306-019-DEV_Loaded_EOL_004_PHASE 150.mdf'
if filePath == '':
    filePath = tk.filedialog.askopenfilename()
fileName = os.path.basename(filePath)

# Load data
mdf_data = mdfreader.Mdf(
    filePath,
    channel_list = config.Channels
)
mdf_data.resample(master_channel="t_71")

# Build a dataframe
data = pd.DataFrame()
for channel in config.Channels:
    chan_info = mdf_data.get_channel(channel)
    if chan_info is not None:
        time_chan = chan_info['master']
        data[time_chan] = mdf_data.get_channel_data(time_chan) - mdf_data.get_channel_data(time_chan).min()
        data[channel] = mdf_data.get_channel_data(channel)
    else:
        print("ERROR: No data found for channel:", channel)
data = data.rename(columns={'t_71': 'time'})

# Misc information
sr = get_sample_rate(data['time'])
gear, gear_hr, ratio = get_gear(data['GearEngd'])

# Calculated channels
calc_IPTrqGradient = np.gradient(uniform_filter1d(data['Cadet_IP_Torque'] , size=int(sr)), edge_order=2) * 100
data['calc_IPTrqGradient'] = calc_IPTrqGradient.tolist()
data['calc_IPTrqGradient_smoothed'] = np.append(smooth(data['calc_IPTrqGradient'], sr + 1), 0.0)
data['calc_AxleTrqFromOutput'] = data['Cadet_OP_Torque_1'] + data['Cadet_OP_Torque_2']
data['calc_AxleTrqFromInput'] = data['Cadet_IP_Torque'] * ratio
data['calc_LockTrq'] = data['Cadet_OP_Torque_1'] - data['Cadet_OP_Torque_2']
data['calc_OPSpeedDelta'] = np.append(smooth(data['WhlRPM_RL'] - data['WhlRPM_RR'], sr + 1), 0.0)

# TODO: Add zero points for filtered data
# TODO: Group data sets into single data points
# TODO: Plot main differential characterisation graph
# TODO: Make executable

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

if config.Debug:
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
