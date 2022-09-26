import tkinter as tk
from tkinter import filedialog
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
# TODO: Plot main differential characterisation graph
# TODO: Make executable

# Filter Data
data_f = data[(abs(data['calc_OPSpeedDelta']) > 10.0) & (abs(data['calc_IPTrqGradient_smoothed']) < 1) & (abs(data['calc_AxleTrqFromInput']) > 50)]
data_f = data_f.reset_index()

# Group Data
# First we build a list of indices which we use to split to dataframe up
group_ids = [len(data_f)]
for idx, row in data_f.iterrows():
    if idx == len(data_f.index)-1:
        break

    cur = data_f.iloc[idx]
    nxt = data_f.iloc[idx + 1]
    diff = nxt['time'] - cur['time']

    if diff > 0.5:
        group_ids.append(idx + 1)

# Split dataframe using the id's created above
l_mod = [0] + group_ids + [max(group_ids) + 1]
list_of_dfs = [data_f.iloc[l_mod[n]:l_mod[n + 1]] for n in range(len(l_mod) - 1)]

# Just some colours to see each dataframe plot more easily, remove at later stage.
plot_colors = [
            "darkred", "red", "darkorange", "orange",
            "gold", "yellow", "lime", "green", "darkgreen",
            "turquoise", "cyan", "blue", "navy",
            "purple", "pink", "magenta", "darkred"
        ]

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

# ax[2].plot(
#     data['time'],
#     data['calc_AxleTrqFromInput'],
#     color='black',
#     label='Axle Torque',
#     marker=None    
# )

for idx, dfgroup in enumerate(list_of_dfs):
    ax[2].scatter(
        dfgroup['time'],
        dfgroup['calc_AxleTrqFromInput'],
        color=plot_colors[idx],
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
    
plt.subplots_adjust(
    left=0.05,
    bottom=0.07,
    right=0.955,
    top=0.9,
    wspace=0.2,
    hspace=0.4
)
plt.show()
