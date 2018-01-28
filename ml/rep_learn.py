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

