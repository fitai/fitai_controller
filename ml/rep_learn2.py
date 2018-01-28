# To learn how to identify rep start/stop points
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from copy import copy

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data
from ml.utils import load_events, build_rep_prob_signal, build_centered_rep_prob_signal, clean_timepoints, \
    max_min_norm, proximity_dist, prox_ordered_dists, rigid_ordered_dist, calc_rep_times
# from ml.modeling import run_model

#: Step 1 - Pull all rep_start and rep_stop events
conn = create_engine(db_conn_string)

# events_sql = '''
# SELECT
#     lift_id
#     , ROUND(timepoint::NUMERIC, 2) AS timepoint
#     , event
# FROM lift_event
# WHERE event IN ('rep_start', 'rep_stop')
# '''
# events = pd.read_sql(events_sql, conn)
#
#
# # make sure that all even timepoints occur on an even point (multiple of 0.02)
# # if the point is odd, then subtract 0.01 from it to make it even
# def clean_event_times(events):
#     events['timepoint'] = [x if round(x*100, 2) % 2 == 0 else round(x, 2)-0.01 for x in events['timepoint']]
#     return events
#
# events = clean_event_times(events)
#
# # pull data and metadata for all lifts in events df
# lift_ids = events['lift_id'].unique()

events, lift_ids = load_events()

# metadata, dat = pd.DataFrame(), pd.DataFrame()

# for id in lift_ids:
# id = lift_ids[0]
id = 395  # baseline lift
# id = 396

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(id)
# data = clean_timepoints(data)  # ensure everything downstream has rounded timepoints

# loads in time between reps
irts = events.loc[events['lift_type'].eq(header['lift_type'])].groupby('lift_id').apply(lambda df: calc_rep_times(df))
inters = irts.xs([id, 'inter_rep'], level=[0, 1])
min_irt = max(inters.min().values[0], 1.)
intras = irts.xs([id, 'intra_rep'], level=[0, 1])
min_intra = max(intras.min().values[0], 1.)

# process acceleration and calculate a/v/pwr/pos/f
a, v, pwr, pos, force = process_data(header, data, RMS=False, highpass=True, verbose=True)

# build entire input data set; join in all data signals
d = a.join(v).join(pwr).join(pos).join(force)
d['timepoint'] = data['timepoint']
d['lift_id'] = header['lift_id']

dat = d  # for single lift processing
# dat = dat.append(d, ignore_index=True)  # for multi-lift processing
# metadata = metadata.append(pd.Series(header), ignore_index=True)
metadata = pd.Series(header).to_frame().T

# Temp filter for events from single lift
events = events.loc[events['lift_id'].eq(id)]
#: Join in relevant data
# events = pd.merge(events, metadata, on='lift_id', how='left')

#: Split into starts and stops - for use later
starts = events.loc[events['event'].eq('rep_start')].reset_index(drop=True)
stops = events.loc[events['event'].eq('rep_stop')].reset_index(drop=True)

# Prep split into test/train
# NOTE: the "timepoint" field will be used as index
N = len(dat)
N1 = int(float(N)*3/5)

tr_idx, te_idx = dat.loc[:N1, 'timepoint'], dat.loc[N1:, 'timepoint']

start_ps = pd.DataFrame()
start_ts = pd.DataFrame()
for i, row in starts.iterrows():
    ps, ts = build_centered_rep_prob_signal(row['timepoint'], header['sampling_rate'], t_window=1.)
    start_ps = start_ps.append(pd.Series(ps, name=i))  # stack set of gaussian probabilities with known mean and std
    start_ts = start_ts.append(pd.Series(ts, name=i))  # stack set of timepoints associated with probabilities

#: Build probability signals
# everything point outside of +/- 1/2 t_window is set to probability 0
#: NOTE: p0 = prob(stop), p1 = prob(start)
# zeros = pd.Series(0., index=dat['timepoint'])  # signal of zeros

# p1 = (zeros + probs).fillna(0.)
# p1.name = 'p1'

# probs = pd.Series()
# for i, row in stops.iterrows():
#     probs = probs.append(build_rep_prob_signal(row['timepoint'], float(header['sampling_rate']), t_window=1.))
#
# p0 = (zeros + probs).fillna(0.)
# p0.name = 'p0'

#: Prep inputs
X = dat.drop(['lift_id', 'millis'], axis=1).copy().set_index('timepoint')

# _, _ = run_model(X, tr_idx, te_idx, p0, p1, 'ridge')

# from sklearn.neural_network import MLPRegressor
#
# mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='tanh')
# mlp.fit(X, p1.reset_index(drop=True))
# y_hat = mlp.predict(X)
#
# plt.plot(p1.reset_index(drop=True), 'black')
# plt.plot(y_hat, 'blue')

plt.figure()
plt.plot(d['timepoint'], max_min_norm(d['a_x']), color='black', alpha=0.75)
plt.plot(d['timepoint'], max_min_norm(d['v_x']), color='blue', alpha=0.5)
# plt.plot(max_min_norm(d['pos_x']), color='black', linestyle='dashed')
for _, e in events.iterrows():
    c = 'green' if 'start' in e['event'] else 'red'
    plt.axvline(e['timepoint'], color=c, linestyle='dashed')


r = range(0, 1500)
t = d['timepoint'].iloc[r]
# sig = d['pos_z'].iloc[r]
sig = d['v_z'].iloc[r]
sig = sig.rolling(window=10, min_periods=0, center=False).mean().fillna(0.)
x = sig.rolling(window=20, min_periods=0, center=False).apply(lambda y: np.mean(sorted(y)[3:-3])).fillna(0.)
v_ = sig.rolling(window=20, min_periods=0, center=False).apply(lambda y: np.var(sorted(y)[3:-3])).fillna(0.)

mask = (sig > x) * 1
p_ = mask.diff()

# plt.figure()
# plt.plot(t, sig, color='black')
# plt.plot(t, x, color='blue', alpha=0.5)
# plt.plot(t, p_*.3, color='purple', alpha=0.5)
# plt.plot(t, v_, color='red', alpha=0.5)
# plt.plot(t, x_.iloc[r], color='red', alpha=0.5)
# plt.axhline(y=0, color='black', linestyle='dashed')
for _, e in events.loc[events['timepoint'] < t.iloc[-1]].iterrows():
    c = 'g' if 'start' in e['event'] else 'r'
    plt.axvline(e['timepoint'], color=c, linestyle='dashed')


# ### REAL-TIME EMULATOR ###

t_min = 0

# sig = d['a_z'].iloc[t_min:]
# sig = d['v_z'].iloc[t_min:]
sig = d['pos_z'].iloc[t_min:]
prev_val = sig.iloc[t_min-1]

# build time series of 30 samples at a time; one packet = 30 samples
packet_size = 30
sampling_rate = 30

n = int(np.ceil((sig.shape[0]) / float(packet_size)))

sig_track = pd.Series()
mean_track = pd.Series()
prev_dat = None
prev_mean = None
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
    t1 = t_min+i*packet_size - 1  # right-exclude

    # split off current packet
    packet = sig.loc[t0:t1].copy()
    # acc, vel, pwr, pos, force = process_data(header, packet, RMS=False, highpass=True)

    # packet = pos  # signal choice here

    # calculate signals of interest
    if prev_dat is None:  # no preceding packets. can't process
        d_ = packet.rolling(window=10, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y))).fillna(packet.mean())
        # calculate truncated rolling mean
        m_ = packet.rolling(window=20, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).fillna(prev_val)
        prev_dat = packet
        prev_mean = m_
    else:
        # bring in previous data to apply rolling window over, then slice out most recent data via iloc
        dat = prev_dat.append(packet)
        # smooth data
        d_ = dat.rolling(window=10, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y))).iloc[packet_size-1:]
        # calculate truncated rolling mean
        m_ = dat.rolling(window=20, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).iloc[packet_size-1:]
        prev_dat = packet
        prev_mean = m_

    # look for crossings
    mask = (d_ < m_) * 1
    # any positive values represent crossings from 0 to 1
    # any negative values represent crossings from 1 to 0
    p_ = mask.diff().fillna(0.)
    # check that there was a crossings, and
    # crossings are far enough away from most recent prior stop
    # NOTE: result of p_ > 0 is a Series, result of p_.index - t_prev is an array. These two cannot be
    #       combined directly via & ; have to convert the Series to an array via Series.values
    crossings = p_.loc[(abs(p_) > 0).values & ((p_.index - t_prev) >= min_irt_samples)]

    # for viz later
    sig_track = sig_track.append(d_.iloc[1:])
    mean_track = mean_track.append(m_.iloc[1:])

    # if 320 < t1 < 380:
    #     print i
    #     plt.figure()
    #     plt.plot(d_, 'black')
    #     plt.plot(m_, 'blue')
    #     if crossings.shape[0] > 0:
    #         print crossings
    #         print cross_track

    # handle any crossings
    if crossings.shape[0] > 0:
        # before moving forward, if no positive crossings have been registered (i.e. len(cross_track) == 0),
        #  the next crossing must be positive. Enforce this here
        if len(cross_track) == 0 and any(crossings == 1) and (crossings.shape[0] > 1):
            # eliminate any negative crossing before a positive crossing
            tc = crossings.loc[crossings == -1].index[0]  # id index of first positive crossing
            crossings = crossings.loc[tc:]  # slice out anything before first positive crossing
        elif len(cross_track) == 0 and any(crossings == 1) and (crossings.shape[0] == 1):
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
            starts.append(ts[0])
            stops.append(ts[3])
            t_prev = ts[3]
        # else:  # registered 4 crossings
        #     starts.append(ts[0])
        #     stops.append(ts[3])

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

plt.figure()
plt.plot(sig_track, 'black', alpha=0.5)
# plt.plot(sig_track, 'blue', linestyle='dashed', alpha=0.5)
plt.plot(mean_track, 'blue', alpha=0.5)

for i, start in enumerate(starts):
    plt.axvline(start, color='g', linestyle='dashed')
    plt.axvline(stops[i], color='r', linestyle='dashed')

calc_reps = len(starts)
n_reps = metadata['final_num_reps'].iloc[0] if metadata['final_num_reps'].iloc[0] is not None else metadata['init_num_reps'].iloc[0]
print('calculated reps: {r1}, actual reps: {r2}'.format(r1=calc_reps, r2=n_reps))

start1 = copy(starts)
start2 = copy((events.loc[events['event'].eq('rep_start')]['timepoint']*30.).values)

m1, sd1 = proximity_dist(start1, start2)
m1 = m1 / 30.  # in seconds

m1, sd1, rem = prox_ordered_dists(start1, list(start2))

stop1 = copy(stops)
stop2 = copy((events.loc[events['event'].eq('rep_stop')]['timepoint']*30.).values)

m1, sd1 = proximity_dist(stop1, stop2)

