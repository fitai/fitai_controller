# To learn how to identify rep start/stop points
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from copy import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data
from ml.utils import load_events, build_rep_prob_signal, build_centered_rep_prob_signal

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
id = 395

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(id)

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

#: Build probability signals
#: NOTE: p0 = prob(stop), p1 = prob(start)
start_ps = pd.DataFrame()
start_ts = pd.DataFrame()
for i, row in starts.iterrows():
    ps, ts = build_centered_rep_prob_signal(row['timepoint'], header['sampling_rate'], t_window=1.)
    start_ps = start_ps.append(pd.Series(ps, name=i))  # stack set of gaussian probabilities with known mean and std
    start_ts = start_ts.append(pd.Series(ts, name=i))  # stack set of timepoints associated with probabilities

# everything point outside of +/- 1/2 t_window is set to probability 0
zeros = pd.Series(0., index=dat['timepoint'])  # signal of zeros

# p1 = (zeros + probs).fillna(0.)
# p1.name = 'p1'

probs = pd.Series()
for i, row in stops.iterrows():
    probs = probs.append(build_rep_prob_signal(row['timepoint'], header['sampling_rate'], t_window=1.))

p0 = (zeros + probs).fillna(0.)
p0.name = 'p0'

#: Prep inputs
X = dat.drop(['lift_id', 'millis'], axis=1).copy().set_index('timepoint')


def run_model(X, tr, te, p0, p1, model_type='rf'):
    if model_type == 'rf':
        m0, m1 = RandomForestRegressor(n_estimators=200), RandomForestRegressor(n_estimators=200)
    elif model_type == 'ridge':
        m0, m1 = Ridge(), Ridge()
    elif model_type == 'lasso':
        m0, m1 = Lasso(), Lasso()

    # predict starts
    m1.fit(X.ix[tr], p1.ix[tr])
    p1_pred = pd.Series(m1.predict(X.ix[te]), index=te)

    fig = plt.figure(0)
    plt.plot(p1.ix[te], 'black', label='testing')
    plt.plot(p1_pred, 'blue', label='prediction')
    fig.legend()

    m0.fit(X.ix[tr], p0.ix[tr])
    p0_pred = pd.Series(m0.predict(X.ix[te]), index=te)

    # fig = plt.figure(1)
    # plt.plot(p0.ix[te], 'black', label='testing')
    # plt.plot(p0_pred, 'blue', label='prediction')
    # fig.legend()

    return m0, m1

_, _ = run_model(X, tr_idx, te_idx, p0, p1, 'ridge')

# from sklearn.neural_network import MLPRegressor
#
# mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='tanh')
# mlp.fit(X, p1.reset_index(drop=True))
# y_hat = mlp.predict(X)
#
# plt.plot(p1.reset_index(drop=True), 'black')
# plt.plot(y_hat, 'blue')

def max_min_norm(s):
    return (s - min(s))/(max(s) - min(s))

plt.figure()
plt.plot(d['timepoint'], max_min_norm(d['a_x']), color='black', alpha=0.75)
plt.plot(d['timepoint'], max_min_norm(d['v_x']), color='blue', alpha=0.5)
# plt.plot(max_min_norm(d['pos_x']), color='black', linestyle='dashed')
for _, e in events.iterrows():
    c = 'green' if 'start' in e['event'] else 'red'
    plt.axvline(e['timepoint'], color=c, linestyle='dashed')


r = range(650, 1500)
t = d['timepoint'].iloc[r]
sig = d['pos_z'].iloc[r]
sig = sig.rolling(window=5, min_periods=0, center=False).mean().fillna(0.)
x = sig.rolling(window=15, min_periods=0, center=False).apply(lambda y: np.mean(sorted(y)[3:-3])).fillna(0.)
# x_ = sig.rolling(window=15, min_periods=10, center=False).mean().fillna(0.)

pos = (sig > x) * 1
p_ = pos.diff()

plt.figure()
plt.plot(t, sig, color='black')
plt.plot(t, x, color='blue', alpha=0.5)
plt.plot(t, p_*.03, color='purple', alpha=0.5)
# plt.plot(t, x_.iloc[r], color='red', alpha=0.5)
# plt.axhline(y=0, color='black', linestyle='dashed')
for _, e in events.loc[events['timepoint'] < t.iloc[-1]].iterrows():
    c = 'g' if 'start' in e['event'] else 'r'
    plt.axvline(e['timepoint'], color=c, linestyle='dashed')

t_min = 650

sig = d['pos_z'].iloc[t_min:]
prev_val = sig.iloc[t_min-1]

# build time series of 30 samples at a time; one packet = 30 samples
packet_size = 30
sampling_rate = 30

n = int(np.ceil((sig.shape[0]) / float(packet_size)))

xyz = pd.Series()
sig_track = pd.Series()
mean_track = pd.Series()
prev_dat = None
prev_mean = None
cross_track = []
ts = []
starts = []
stops = []
t_track = 0.
# idx_max = t_min
hold = False
for i in range(1, n):
    print i
    t0 = t_min+(i-1)*packet_size
    t1 = t_min+i*packet_size - 1  # right-exclude

    # split off current packet
    packet = sig.loc[t0:t1].copy()

    if prev_dat is None:  # no preceding packets. can't process
        d_ = packet.rolling(window=5, min_periods=0, center=True).mean().fillna(packet.mean())
        # calculate truncated rolling mean
        m_ = packet.rolling(window=15, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).fillna(prev_val)
        prev_dat = packet
        prev_mean = m_
    else:
        # bring in previous data to apply rolling window over, then slice out most recent data via iloc
        dat = prev_dat.append(packet)
        # smooth data
        d_ = dat.rolling(window=5, min_periods=0, center=True).mean().iloc[packet_size:]
        # calculate truncated rolling mean
        m_ = dat.rolling(window=15, min_periods=0, center=False).apply(
            lambda y: np.mean(sorted(y)[3:-3])).iloc[packet_size:]
        prev_dat = packet
        prev_mean = m_

    sig_track = sig_track.append(d_)
    mean_track = mean_track.append(m_)
    pos = (d_ > m_) * 1
    # any positive values represent crossings from 0 to 1
    # any negative values represent crossings from 1 to 0
    p_ = pos.diff().fillna(0.)

    crossings = p_.loc[abs(p_) > 0]

    if crossings.shape[0] > 0:
        xyz = xyz.append(crossings)
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
            hold_c = cross_track[4:]  # preserve future crossings
            hold_t = ts[4:]
            cross_track = cross_track[:4]  # move forward with current crossings
            ts = ts[:4]
            hold = True
        else:  # registered 4 crossings
            starts.append(ts[0])
            stops.append(ts[3])

        # clear tracking variables
        if hold:
            cross_track = copy(hold_c)
            ts = copy(hold_t)
            hold = False
        else:
            cross_track = []
            ts = []


s_ = sig.rolling(window=5, min_periods=0, center=True).mean().fillna(0.)
x_ = sig.rolling(window=15, min_periods=0, center=False).apply(lambda y: np.mean(sorted(y)[3:-3])).fillna(0.)

plt.figure()
plt.plot(s_, 'black', alpha=0.5)
plt.plot(sig_track, 'blue', linestyle='dashed', alpha=0.5)

for i, start in enumerate(starts):
    plt.axvline(start, color='g', linestyle='dashed')
    plt.axvline(stops[i], color='r', linestyle='dashed')

# for tp in xyz.index:
#     plt.axvline(tp, color='purple', alpha=0.5)

calc_reps = len(starts)
n_reps = metadata['final_num_reps'].iloc[0] if metadata['final_num_reps'].iloc[0] is not None else metadata['init_num_reps'].iloc[0]
print('calculated reps: {r1}, actual reps: {r2}'.format(r1=calc_reps, r2=n_reps))

from utils import proximity_dist, prox_ordered_dists, rigid_ordered_dist

start1 = copy(starts)
start2 = copy((events.loc[events['event'].eq('rep_start')]['timepoint']*30.).values)

m1, sd1 = proximity_dist(start1, start2)
m1 = m1 / 30.  # in seconds

m1, sd1, rem = prox_ordered_dists(start1, list(start2))

stop1 = copy(stops)
stop2 = copy((events.loc[events['event'].eq('rep_stop')]['timepoint']*30.).values)

m1, sd1 = proximity_dist(stop1, stop2)

