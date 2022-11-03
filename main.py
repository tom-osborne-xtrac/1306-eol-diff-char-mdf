import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import mdfreader
import numpy as np
import pandas as pd
from matplotlib import collections as matcoll
from scipy.ndimage import uniform_filter1d

import config as config
from functions import *

# The below command can be used in PowerShell if you are having permission issues with starting the virtual environment
# Set-ExecutionPolicy Unrestricted -Scope CurrentUser

# TODO: Add zero points for filtered data
# TODO: Make executable

# Select file for analysis
filePath = ''
if config.Debug and config.UseSampleData:
    filePath = './data/17082022_1306-019-DEV_Loaded_EOL_004_PHASE 150.mdf'

if filePath == '':
    filePath = tk.filedialog.askopenfilename()
fileName = os.path.basename(filePath)

# Load data
mdf_data = mdfreader.Mdf(filePath, channel_list = config.Channels)
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
sr = get_sample_rate(data['time']) # Sample Rate
gear, gear_hr, ratio = get_gear(data['GearEngd']) # Gear, Gear (Human readable), Ratio

# Calculated channels
calc_IPTrqGradient = np.gradient(uniform_filter1d(data['Cadet_IP_Torque'] , size=int(sr)), edge_order=2) * 100
data['calc_IPTrqGradient'] = calc_IPTrqGradient.tolist()
data['calc_IPTrqGradient_smoothed'] = np.append(smooth(data['calc_IPTrqGradient'], sr + 1), 0.0)
data['calc_AxleTrqFromOutput'] = data['Cadet_OP_Torque_1'] + data['Cadet_OP_Torque_2']
data['calc_AxleTrqFromInput'] = data['Cadet_IP_Torque'] * ratio
data['calc_LockTrq'] = data['Cadet_OP_Torque_1'] - data['Cadet_OP_Torque_2']
data['calc_OPSpeedDelta'] = np.append(smooth(data['WhlRPM_RL'] - data['WhlRPM_RR'], sr + 1), 0.0)

# Filter Data
# Here we are filtering the raw data to pick out the static loaded sections
# This is the data we need to produce the graph and should ignore all ramps and unloaded sections
data_f = data[
    (abs(data['calc_OPSpeedDelta']) > 10.0) & 
    (abs(data['calc_IPTrqGradient_smoothed']) < 1) & 
    (abs(data['calc_AxleTrqFromInput']) > 50)
].reset_index()

# Attempt at adding data points for zero load
# Previously I was able to do this by taking 50 points of data while unloaded and cornering
# This method is trying to filter based on the following conditions, but seems to pick up some erroneous data
data_zero = data[
    (abs(data['calc_OPSpeedDelta']) > 15) & 
    (abs(data['Cadet_IP_Torque']) < 5) &
    (abs(data['calc_IPTrqGradient_smoothed']) < 1)
]

# Here we are splitting the data into two data series, to be distinguished on the plot as LH and RH corners
# This allows us to see any "asymmetry" in the diff, which seems particularly prominent in the Coast direction

# LH Data
data_f_L = data_f[data_f['calc_OPSpeedDelta'] > 15].reset_index()
data_f_L_grouped = SplitData(data_f_L)

# RH Data
data_f_R = data_f[data_f['calc_OPSpeedDelta'] < -15].reset_index()
data_f_R_grouped = SplitData(data_f_R)

# Just some colours to see each dataframe plot more easily, remove at later stage.
plot_colors = [
    "darkred", "red", "darkorange", "orange",
    "gold", "yellow", "lime", "green", "darkgreen",
    "turquoise", "cyan", "blue", "navy",
    "purple", "pink", "magenta", "darkred"
]

# Set points for torque analysis graphs
# This can be used to format the axis
set_points_x = [-800, -400, -200, -100, 0, 100, 200, 400, 800]
set_points = []
for v in set_points_x:
    pair = [(v, 0), (v, 1000)]
    set_points.append(pair)
plot_set_points = matcoll.LineCollection(set_points)


# Plot Data
# Fig 1: Raw data plotted as time series - Useful for debugging the analysis and checking the test 
# was completed correctly
# Fig 2: Diff Characterisation Plot: This is x-axis = Input  Torque; y-axis = Locking Torque
if config.Debug:
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
        data['time'],
        data['calc_AxleTrqFromInput'],
        color='black',
        label='Axle Torque',
        marker=None,
        linewidth=0.5
    )

    # This can be uncommented if you want to see each data group in its own colour (check for anomolies)
    # for idx, dfgroup in enumerate(list_of_dfs):
    #     ax[2].scatter(
    #         dfgroup['time'],
    #         dfgroup['calc_AxleTrqFromInput'],
    #         color=plot_colors[idx],
    #         label='Axle Torque',
    #         marker=".",
    #         s=50,
    #         zorder=1
    #     )
    for idx, dfgroup in enumerate(data_f_L_grouped):
        ax[2].scatter(
            dfgroup['time'],
            dfgroup['calc_AxleTrqFromInput'],
            color="red",
            label='Axle Torque',
            marker=".",
            zorder=1
        )

    for idx, dfgroup in enumerate(data_f_R_grouped):
        ax[2].scatter(
            dfgroup['time'],
            dfgroup['calc_AxleTrqFromInput'],
            color="blue",
            label='Axle Torque',
            marker=".",
            s=50,
            zorder=1
        )

    set_axis(ax, 'x', 'Time [s]', 0, data['time'].max(), 50, 5)
    set_axis([ax[0]], 'y', 'Torque [Nm]', -200, 200, 50, 10)
    set_axis([ax[1], ax[2]], 'y', 'Torque [Nm]', -1000, 1000, 250, 50)
    axSecondary0.set_ylim([-10, 10])
    axSecondary1.set_ylim([-100, 100])
    ax[0].set_title("Input Torque & Input Torque Delta", loc='left')
    fig.suptitle(f'Diff Test Overview - 3rd Gear', fontsize=16)

# Diff Characterisaion Plot
fig2, ax2 = plt.subplots(1)

for idx, dfgroup in enumerate(data_f_L_grouped):
    ax2.scatter(
        dfgroup['calc_AxleTrqFromInput'].mean(),
        abs(dfgroup['calc_LockTrq'].mean()),
        color="red",
        label='Axle Torque',
        marker=".",
        s=100,
        zorder=1
    )

for idx, dfgroup in enumerate(data_f_R_grouped):
    ax2.scatter(
        dfgroup['calc_AxleTrqFromInput'].mean(),
        abs(dfgroup['calc_LockTrq'].mean()),
        color="blue",
        label='Axle Torque',
        marker=".",
        s=100,
        zorder=1
    )

set_axis([ax2], 'x', ' Input Torque [Nm]', -1000, 1000, 100, 50)
set_axis([ax2], 'y', ' Locking Torque [Nm]', 0, 1000, 100, 50)
ax2.axvline(0, color='black')
ax2.spines['bottom'].set_linewidth(2)

plt.subplots_adjust(
    left=0.05,
    bottom=0.07,
    right=0.955,
    top=0.9,
    wspace=0.2,
    hspace=0.4
)
plt.show()
