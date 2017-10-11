# To learn how to identify rep start/stop points
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from databasing.conn_strings import db_conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data

#: Step 1 - Pull all rep_start and rep_stop events
conn = create_engine(db_conn_string)

events_sql = '''
SELECT
    lift_id
    , ROUND(timepoint::NUMERIC, 2) AS timepoint
    , event
FROM lift_event
WHERE event IN ('onset', 'offset')
'''
events = pd.read_sql(events_sql, conn)


# make sure that all even timepoints occur on an even point (multiple of 0.02)
# if the point is odd, then subtract 0.01 from it to make it even
def clean_event_times(events):
    events['timepoint'] = [x if round(x*100, 2) % 2 == 0 else round(x, 2)-0.01 for x in events['timepoint']]
    return events

events = clean_event_times(events)

# pull data and metadata for all lifts in events df
lift_ids = events['lift_id'].unique()

metadata, dat = pd.DataFrame(), pd.DataFrame()

# for id in lift_ids:
id = lift_ids[0]

# pull unprocessed lift header (metadata) and data (acceleration values)
header, data = pull_data_by_lift(id)

# process acceleration and calculate a/v/pwr/pos/f
a, v, pwr, pos, force = process_data(header, data, RMS=False, highpass=True, verbose=True)

# build entire input data set; join in all data signals
d = a.join(v).join(pwr).join(pos).join(force)
d['timepoint'] = data['timepoint']
d['lift_id'] = header['lift_id']

dat = dat.append(d, ignore_index=True)
# metadata = metadata.append(pd.Series(header), ignore_index=True)
metadata = header.to_frame()

# Temp filter for events from single lift
events = events.loc[events['lift_id'].eq(id)]
#: Join in relevant data
events = pd.merge(events, metadata, on='lift_id', how='left')

#: Split into starts and stops - for use later
starts = events.loc[events['event'].eq('onset')].reset_index(drop=True)
stops = events.loc[events['event'].eq('offset')].reset_index(drop=True)


def build_rep_prob_signal(t, sampling, t_window):
    #: Based on input t_range (e.g. number seconds of signal to include), calculate
    #: appropriate mean, std
    #: Where mean = t_start (or t_stop) ( = t)
    #:       std = such that any values outside of +/- 4*std will be truncated to zero
    t_half = t_window/2.  # half of range on each side (+/-) of t
    sig = (t_half/4.)   # anything outside this gets truncated to prob = 0

    #: NOTE TO SELF: Look into why this doesn't create numbers spaced evenly
    # test = np.linspace(t - t_half, t + t_half, num=t_range * sampling)

    bound = int(t_half*sampling)
    # Round so that float conversion won't introduce error and make
    # resultant timepoints un-alignable with original data timepoints.
    ts = [round(t + (x/sampling), 3) for x in range(-1*bound, bound+1)]
    g_x = [np.exp(-np.power(x - t, 2.) / (2 * np.power(sig, 2.))) for x in ts]

    #: Convert to series
    g_x = pd.Series(data=g_x, index=ts)
    return g_x


# Prep split into test/train
# NOTE: the "timepoint" field will be used as index
N = len(dat)
N1 = int(float(N)*3/5)

tr_idx, te_idx = dat.loc[:N1, 'timepoint'], dat.loc[N1:, 'timepoint']

#: Build probability signals
#: NOTE: p0 = prob(stop), p1 = prob(start)
probs = pd.Series()
for i, row in starts.iterrows():
    probs = probs.append(build_rep_prob_signal(row['timepoint'], row['sampling_rate'], t_window=1.))

# everything point outside of +/- 1/2 t_window is set to probability 0
zeros = pd.Series(0., index=dat['timepoint'])  # signal of zeros
p1 = (zeros + probs).fillna(0.)
p1.name = 'p1'

probs = pd.Series()
for i, row in stops.iterrows():
    probs = probs.append(build_rep_prob_signal(row['timepoint'], row['sampling_rate'], t_window=1.))

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
