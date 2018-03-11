# To learn how to identify rep start/stop points
from __future__ import division

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from ml.utils import load_events, proximity_dist, prox_ordered_dists, rigid_ordered_dist, calc_rep_times
from processing.util import process_data
from rep_learn import run_detector

conn = create_engine(db_conn_string)

events, lift_ids = load_events()

lift_id = 395  # baseline lift
# lift_id = 396
# lift_id = 510

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(lift_id)

# ### offline ###

# process acceleration and calculate a/v/pwr/pos/f
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

# ### Values required by the algorithm: ###

# Establish expectations on time between reps and within reps
irts = events.loc[events['lift_type'].eq(header['lift_type'])].groupby('lift_id').apply(lambda df: calc_rep_times(df))
inters = irts.xs([lift_id, 'inter_rep'], level=[0, 1])
min_irt = max(inters.min().values[0], 1.)
intras = irts.xs([lift_id, 'intra_rep'], level=[0, 1])
min_intra = max(intras.min().values[0], 1.)

# Initial conditions/variables, specific to detector
t_min = 0
t_max = data.shape[0]

# build time series of 30 samples at a time; one packet = 30 samples
packet_size = header['sampling_rate']
sampling_rate = header['sampling_rate']

n = int(np.ceil((t_max-t_min)/packet_size))

# signal tracking
sig_track = pd.Series()
p_track = pd.Series()
prev_dat = []

# result tracking
starts = []
stops = []
cross_track = []
ts = []

# initial_conditions
prev_vz = 0.
prev_filt_vz = 0.
prev_az = 0.
prev_filt_az = 0.
t_prev = 0.
n_reps = 0.
hold = False

# detector parameters
var_max = 0.05
min_irt_samples = 10
min_intra_samples = 60
# min_irt_samples = min_irt * header['sampling_rate']
# min_intra_samples = min_intra * header['sampling_rate']

# ### Real-Time Emulator
for i in range(1, n+1):
    t0 = t_min+(i-1)*packet_size
    t1 = t_min+i*packet_size   # right-exclude

    # establish initial conditions
    y0 = {'x': prev_az, 'y': prev_filt_az}

    # bring in data packet
    packet = data.iloc[t0:t1].copy()
    accel = packet[['timepoint', 'a_x', 'a_y', 'a_z']]
    acc, vel, _, _, _ = process_data(header, accel, inits={'v_z': prev_vz, 'a_z': y0}, RMS=False, highpass=True)

    # these will likely change a lot, so use kwargs input to keep things easy
    detector_input = {
        'vel': vel
        , 'sampling_rate': sampling_rate
        , 'prev_dat': prev_dat
        , 'prev_vz': prev_vz
        , 'prev_filt_vz': prev_filt_vz
        , 'packet_size': packet_size
        , 't_prev': t_prev
        , 'hold': hold
        , 'cross_track': cross_track
        , 'ts': ts
        , 'var_max': var_max
        , 'min_irt_samples': min_irt_samples
        , 'min_intra_samples': min_intra_samples
        , 'starts': starts
        , 'stops': stops
        , 'n_reps': n_reps
        , 't_min': t_min
        # extra, just for plotting purposes:
        , 'sig_track': sig_track
        , 'p_track': p_track
    }
    sig, prev_dat, hold, cross_track, ts, n_reps, sig_track, p_track = run_detector(**detector_input)

    # initial conditions for next loop
    prev_vz = vel['v_z'].iloc[-1]
    prev_filt_vz = sig.iloc[-1]
    prev_az = accel['a_z'].iloc[-1]
    prev_filt_az = acc['a_z'].iloc[-1]

plt.figure(2)
plt.title('Detector Output - lift {}'.format(lift_id))
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
print('Lift {l} - calculated reps: {r1}, actual reps: {r2}'.format(l=lift_id, r1=calc_reps, r2=n_reps))

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
