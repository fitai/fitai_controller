# To learn how to identify rep start/stop points
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from copy import copy

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from ml.utils import load_events, proximity_dist, prox_ordered_dists, rigid_ordered_dist, calc_rep_times
from processing.util import process_data

conn = create_engine(db_conn_string)

events, lift_ids = load_events()

id = 395  # baseline lift
# id = 396

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(id)

# loads in time between reps
irts = events.loc[events['lift_type'].eq(header['lift_type'])].groupby('lift_id').apply(lambda df: calc_rep_times(df))
inters = irts.xs([id, 'inter_rep'], level=[0, 1])
min_irt = max(inters.min().values[0], 1.)
intras = irts.xs([id, 'intra_rep'], level=[0, 1])
min_intra = max(intras.min().values[0], 1.)

# process acceleration and calculate a/v/pwr/pos/f
a, v, pwr, pos, force = process_data(header, data, RMS=False, highpass=True, verbose=True)
events = events.loc[events['lift_id'].eq(id)]
# build entire input data set; join in all data signals
d = a.join(v).join(pwr).join(pos).join(force)
d['timepoint'] = data['timepoint']
d['lift_id'] = header['lift_id']

r = range(0, 1500)
t = d['timepoint'].iloc[r]
sig = d['v_z'].iloc[r]
# sig = d['v_z'].iloc[r]
sig = sig.rolling(window=10, min_periods=0, center=False).mean().fillna(0.)
sig_ = sig.copy()
x = sig.rolling(window=20, min_periods=0, center=False).apply(lambda y: np.mean(sorted(y)[3:-3])).fillna(0.)
# v_ = sig.rolling(window=20, min_periods=0, center=False).apply(lambda y: np.var(sorted(y)[3:-3])).fillna(0.)

mask = (sig > x) * 1
p_ = mask.diff()

plt.figure()
plt.plot(t, sig, color='black')
plt.plot(t, x, color='blue', alpha=0.5)
plt.plot(t, p_*1, color='purple', alpha=0.5)
for _, e in events.loc[events['timepoint'] < t.iloc[-1]].iterrows():
    c = 'g' if 'start' in e['event'] else 'r'
    plt.axvline(e['timepoint'], color=c, linestyle='dashed')

# ### REAL-TIME EMULATOR ###

t_min = 0
t_max = 1500

# sig = d['a_z'].iloc[t_min:]
# sig = d['v_z'].iloc[t_min:]
sig = d['v_z'].iloc[t_min:t_max+1]
prev_val = sig.iloc[t_min-1] if t_min > 0 else 0.

# build time series of 30 samples at a time; one packet = 30 samples
packet_size = header['sampling_rate']
sampling_rate = header['sampling_rate']

n = int(np.ceil((sig.shape[0]) / float(packet_size)))

sig_track = pd.Series()
mean_track = pd.Series()
prev_dat = []
cross_track = []
ts = []
starts = []
stops = []
t_track = 0.
min_irt_samples = min_irt * sampling_rate  # minimum number of samples between rep stop and next rep start
min_intra_samples = 0.9*min_intra * sampling_rate  # min number samples within a rep
# idx_max = t_min
hold = False
t_prev = -1 * sampling_rate
for i in range(1, n):
    t0 = t_min+(i-1)*packet_size
    t1 = t_min+i*packet_size   # right-exclude

    # split off current packet
    # packet = sig.loc[t0:t1].copy()
    # accel = packet
    packet = data.iloc[t0:t1].copy()
    accel = packet[['timepoint', 'a_x', 'a_y', 'a_z']]

    # calculate signals of interest
    if len(prev_dat) < 2:  # no preceding packets. can't process
        acc, vel, pwr, pos, force = process_data(header, accel, RMS=False, highpass=True)
        s_ = vel['v_z']
        # s_ = accel
        # smooth data
        d_ = s_.rolling(window=10, min_periods=0, center=False).apply(
            lambda y: np.mean(y)).fillna(s_.mean())
        # calculate truncated rolling mean
        m_ = s_.rolling(window=20, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).fillna(s_.mean())
        prev_dat.append(accel)
    else:
        # bring in previous data to apply rolling window over, then slice out most recent data via iloc
        dat = pd.concat(prev_dat + [accel], axis=0)
        acc, vel, pwr, pos, force = process_data(header, dat, RMS=False, highpass=True)
        s_ = vel['v_z']
        # s_ = dat
        d_ = s_.rolling(window=10, min_periods=0, center=False).apply(
            lambda y: np.mean(y)).iloc[2*packet_size-1:]
        m_ = s_.rolling(window=20, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).iloc[2*packet_size-1:]
        prev_dat.pop(0)  # remove oldest packet
        prev_dat.append(accel)

    # look for crossings
    mask = (d_ > m_) * 1
    # any positive values represent crossings from 1 to 0
    # any negative values represent crossings from 0 to 1
    p_ = mask.diff().fillna(0.)
    # check that there was a crossings, and
    # crossings are far enough away from most recent prior stop
    # NOTE: result of p_ > 0 is a Series, result of p_.index - t_prev is an array. These two cannot be
    #       combined directly via & ; have to convert the Series to an array via Series.values
    crossings = p_.loc[(abs(p_) > 0).values & ((p_.index - t_prev) >= min_irt_samples)]

    # for viz later
    sig_track = sig_track.append(d_.iloc[1:], ignore_index=True)
    mean_track = mean_track.append(m_.iloc[1:], ignore_index=True)

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

        if len(cross_track) < 4:
            continue
        elif len(cross_track) > 4:
            if ts[4] - ts[3] > min_irt_samples:
                hold_c = cross_track[4:]  # preserve future crossings
                hold_t = ts[4:]
                hold = True
            cross_track = cross_track[:4]  # move forward with current crossings
            ts = ts[:4]

        # ensure the rep lasted at least sampling_rate samples - cut out jitter
        if ts[3] - ts[0] > min_intra_samples:
            print('logging start/stop pair')
            starts.append(ts[0] - t_min)
            stops.append(ts[3] - t_min)
            t_prev = ts[3]

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

mask = (sig_track > mean_track) * 1
p_ = mask.diff()
plt.figure()
plt.plot(sig_track.reset_index(drop=True), 'black', alpha=0.5)
plt.plot(sig_, 'black', linestyle='dashed')
# plt.plot(mean_track.reset_index(drop=True), 'blue', alpha=0.5)
# plt.plot(p_, 'purple', alpha=0.5)

for i, start in enumerate(starts):
    plt.axvline(start, color='g', linestyle='dashed')
    plt.axvline(stops[i], color='r', linestyle='dashed')

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
