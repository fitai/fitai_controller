# To learn how to identify rep start/stop points
from __future__ import division

import pandas as pd
import numpy as np
from copy import copy

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from ml.utils import load_events, proximity_dist, prox_ordered_dists, rigid_ordered_dist, calc_rep_times
from processing.util import process_data
from processing.filters import filter_signal

conn = create_engine(db_conn_string)

events, lift_ids = load_events()

lift_id = 395  # baseline lift
# lift_id  = 396
# lift_id = 510

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(lift_id)

# Establish expectations on time between reps and within reps
irts = events.loc[events['lift_type'].eq(header['lift_type'])].groupby('lift_id').apply(lambda df: calc_rep_times(df))
inters = irts.xs([lift_id, 'inter_rep'], level=[0, 1])
min_irt = max(inters.min().values[0], 1.)
intras = irts.xs([lift_id, 'intra_rep'], level=[0, 1])
min_intra = max(intras.min().values[0], 1.)

# ### offline ###

# # process acceleration and calculate a/v/pwr/pos/f
# a, v, _, _, _ = process_data(header, data, inits={'a_z': {'x': 0, 'y': 0}}, RMS=False, highpass=True)
# events = events.loc[events['lift_id'].eq(lift_id)]
#
# r = range(data.shape[0])
# sig = v['v_z'].iloc[r].copy()
# vel_ = pd.Series(filter_signal(sig.values, {'x': 0, 'y': 0}, 'highpass', freqs=[.1, None], fs=header['sampling_rate'], filter_order=1), name='v_z')
#
# x = vel_.rolling(window=10, min_periods=0, center=False).apply(np.mean).fillna(0.)
# v_ = vel_.rolling(window=10, min_periods=0, center=False).apply(lambda y: np.abs(np.mean(sorted(y)[3:-3]))).fillna(0.)
# var_ = vel_.rolling(window=10, min_periods=0, center=False).apply(lambda y: np.var(sorted(y)[2:-2])).fillna(0.)
# t = data['timepoint'].iloc[r]
#
# var_max = max(var_)
# mask_var = (var_ > 0.02*var_max) * 1
# v_m = mask_var.rolling(window=10, min_periods=0, center=False).apply(np.mean)
#
# plt.figure()
# plt.plot(t, vel_, color='black')
# plt.plot(t, var_, color='green', alpha=0.5)
# plt.plot(t, mask_var, color='purple', alpha=0.5)
#
# for _, e in events.loc[events['timepoint'] < t.iloc[-1]].iterrows():
#     c = 'g' if 'start' in e['event'] else 'r'
#     plt.axvline(e['timepoint'], color=c, linestyle='dashed')

# ### REAL-TIME EMULATOR ###

t_min = 0
t_max = data.shape[0]

# build time series of 30 samples at a time; one packet = 30 samples
packet_size = header['sampling_rate']
sampling_rate = header['sampling_rate']

n = int(np.ceil((t_max-t_min)/packet_size))

sig_track = pd.Series()
a_track = pd.Series()
p_track = pd.Series()
prev_dat = []

# initial_conditions
prev_vz = 0.
prev_filt_vz = 0.
prev_az = 0.
prev_filt_az = 0.
t_prev = -1 * sampling_rate

hold = False
min_irt_samples = 10
min_intra_samples = 60

cross_track = []
starts = []
stops = []
ts = []

for i in range(1, n+1):
    t0 = t_min+(i-1)*packet_size
    t1 = t_min+i*packet_size   # right-exclude

    # establish initial conditions
    y0 = {'x': prev_az, 'y': prev_filt_az}

    # bring in data packet
    packet = data.iloc[t0:t1].copy()
    accel = packet[['timepoint', 'a_x', 'a_y', 'a_z']]
    acc, vel, _, _, _ = process_data(header, accel, inits={'v_z': prev_vz, 'a_z': y0}, RMS=False, highpass=True)
    sig = vel['v_z']
    v0 = {'x': prev_vz, 'y': prev_filt_vz}
    sig = pd.Series(filter_signal(sig.values, v0, 'highpass', [.1, None], header['sampling_rate'], 1), index=vel['v_z'].index, name='v_z')

    if len(prev_dat) < 1:
        var_ = sig.rolling(window=10, min_periods=0, center=False).apply(lambda y: np.var(sorted(y)[2:-2])).fillna(0.)
        p_ = pd.Series(np.zeros(packet_size), name=sig.name)
        prev_dat.append(sig)
    else:
        sig_ = pd.concat(prev_dat + [sig], axis=0)
        var_ = sig_.rolling(window=10, min_periods=0, center=False).apply(lambda y: np.var(sorted(y)[2:-2])).fillna(0.)
        mask_var = (var_ > 0.02 * var_max) * 1
        v_m = mask_var.rolling(window=10, min_periods=0, center=False).apply(np.mean)

        p_ = ((v_m > 0.05) * 1).diff().fillna(0.).iloc[packet_size:]
        var_ = var_.iloc[packet_size:]
        prev_dat[0] = sig  # overwrite old packet

    sig_track = sig_track.append(sig)
    p_track = p_track.append(p_)
    a_track = a_track.append(acc['a_z'])

    # initial conditions for next loop
    prev_vz = vel['v_z'].iloc[-1]
    prev_filt_vz = sig.iloc[-1]
    prev_az = accel['a_z'].iloc[-1]
    prev_filt_az = acc['a_z'].iloc[-1]

    # look for crossings
    # any positive values represent crossings from 0 to 1
    # any negative values represent crossings from 1 to 0
    # check that there were any crossings, and that
    # crossings are far enough away from most recent prior stop
    # NOTE: result of p_ > 0 is a Series, result of p_.index - t_prev is an array. These two cannot be
    #       combined directly via & ; have to convert the Series to an array via Series.values
    crossings = p_.loc[(abs(p_) > 0).values & ((p_.index - t_prev) >= min_irt_samples)]

    # handle any crossings
    if crossings.shape[0] > 0:
        # before moving forward, if no positive crossings have been registered (i.e. len(cross_track) == 0),
        #  the next crossing must be positive. Enforce this here
        if len(cross_track) == 0 and any(crossings == -1) and (crossings.shape[0] > 1):
            # eliminate any negative crossing before a positive crossing
            tc = crossings.loc[crossings == 1].index[0]  # id index of first positive crossing
            crossings = crossings.loc[tc:]  # slice out anything before first positive crossing
        elif len(cross_track) == 0 and any(crossings == -1) and (crossings.shape[0] == 1):
            # only one crossing, and it's negative; do not log this
            continue

        # update tracking variables
        if len(cross_track) == 0:
            cross_track = list(crossings.values)
            ts = list(crossings.index)
        else:
            cross_track += list(crossings.values)
            ts += list(crossings.index)

        if len(cross_track) < 2:
            continue
        elif len(cross_track) > 2:
            if ts[2] - ts[1] > min_irt_samples:
                hold_c = cross_track[2:]  # preserve future crossings
                hold_t = ts[2:]
                hold = True
            cross_track = cross_track[:2]  # move forward with current crossings
            ts = ts[:2]

        # ensure the rep lasted at least sampling_rate samples - cut out jitter
        if ts[1] - ts[0] > min_intra_samples:
            print('logging start/stop pair')
            starts.append(ts[0] - t_min)
            stops.append(ts[1] - t_min)
            t_prev = ts[1]

        # triggers when len(cross_track) >= 4
        # clear tracking variables
        if hold and (ts[0] - t_prev >= min_irt_samples):
            cross_track = copy(hold_c)
            ts = copy(hold_t)
        else:
            cross_track = []
            ts = []

        # because of the "and" in the if condition, hard-reset hold each time this triggers, just in case
        hold = False

plt.figure(2)
plt.title('Detector Output')
plt.plot(data['timepoint'], sig_track, 'blue', alpha=0.75, label='online_vz')
plt.legend()

start_ts = [x/sampling_rate for x in starts]
stop_ts = [x/sampling_rate for x in stops]

# plot detected points
for i, start in enumerate(start_ts):
    plt.axvline(start, color='g', linestyle='dashed')
    plt.axvline(stop_ts[i], color='r', linestyle='dashed')

# plot labeled points
for _, e in events.loc[(events['timepoint'] < t_max) & (events['lift_id'].eq(lift_id))].iterrows():
    c = 'g' if 'start' in e['event'] else 'r'
    plt.axvline(e['timepoint'], color=c, linestyle='solid')


calc_reps = len(starts)
n_reps = header['final_num_reps'] if header['final_num_reps'] is not None else header['init_num_reps']
print('calculated reps: {r1}, actual reps: {r2}'.format(r1=calc_reps, r2=n_reps))

# start1 = copy(starts)
# start2 = copy((events.loc[events['event'].eq('rep_start')]['timepoint']*30.).values)
#
# m1, sd1 = proximity_dist(start1, start2)
# m1 = m1 / 30.  # in seconds
#
# m1, sd1, rem = prox_ordered_dists(start1, list(start2))
#
# stop1 = copy(stops)
# stop2 = copy((events.loc[events['event'].eq('rep_stop')]['timepoint']*30.).values)
#
# m1, sd1 = proximity_dist(stop1, stop2)
