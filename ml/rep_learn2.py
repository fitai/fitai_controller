# To learn how to identify rep start/stop points
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from databasing.db_conn_strings import conn_string
from databasing.database_pull import pull_data_by_lift
from processing.util import process_data

#: Step 1 - Pull all rep_start and rep_stop events
conn = create_engine(conn_string)

events_sql = '''
SELECT * FROM lift_event
WHERE event IN ('rep_start', 'rep_stop')
'''
events = pd.read_sql(events_sql, conn)

# pull data and metadata for all lifts in events df
lift_ids = events['lift_id'].unique()

metadata, dat = pd.DataFrame(), pd.DataFrame()
# for id in lift_ids:
id = lift_ids[0]

header, data = pull_data_by_lift(id)

a, v, pwr, pos = process_data(header, data, RMS=False, verbose=True)

d = a.join(v).join(pwr).join(pos)
d['timepoint'] = data['timepoint']
d['lift_id'] = header['lift_id']

dat = dat.append(d, ignore_index=True)
metadata = metadata.append(pd.Series(header), ignore_index=True)

# Temp filter for events from single lift
events = events.loc[events['lift_id'].eq(id)]
#: Join in relevant data
events = pd.merge(events, metadata, on='lift_id', how='left')

# ts = 1.
# event_dat = pd.DataFrame()
# for i, row in events.iterrows():
#     #: everything within +/- ts of event timepoint
#     tmp = dat.loc[(dat['timepoint'] >= row['timepoint'] - ts) & (dat['timepoint'] <= row['timepoint'] + ts)]
#     tmp['event_id'] = i
#
#     event_dat = event_dat.append(tmp.drop('lift_id', axis=1), ignore_index=True)


#: Split into starts and stops - for use later
starts = events.loc[events['event'].eq('rep_start')].reset_index(drop=True)
stops = events.loc[events['event'].eq('rep_stop')].reset_index(drop=True)


def build_rep_prob_signal(t, sampling, t_window):
    # print 'row as passed (type{t}): \n{v}'.format(t=type(row), v=row)
    # # i, s = row
    # t = row['timepoint']
    # sampling = row['sampling_rate']

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


#: Build probability signals
#: NOTE: p0 = prob(stop), p1 = prob(start)

# starts.apply(lambda row: build_prob_signal(row['timepoint'], row['sampling_rate'], t_range=1.), axis=1)
probs = pd.Series()
for i, row in starts.iterrows():
    probs = probs.append(build_rep_prob_signal(row['timepoint'], row['sampling_rate'], t_window=1.))


zeros = pd.Series(0., index=dat['timepoint'])
p1 = (zeros + probs).fillna(0.)
p1.name = 'p1'

probs = pd.Series()
for i, row in stops.iterrows():
    probs = probs.append(build_rep_prob_signal(row['timepoint'], row['sampling_rate'], t_window=1.))

p0 = (zeros + probs).fillna(0.)
p0.name = 'p0'

# plt.plot(p0, 'r', p1, 'g')

#: Prep inputs
X = dat.drop(['lift_id', 'timepoint'], axis=1).copy()

#: Train model on each

from sklearn.linear_model import LinearRegression, Ridge

start_model = LinearRegression()
start_model = Ridge()
start_model.fit(X, y=p1.reset_index(drop=True))

y_hat = start_model.predict(X)

plt.plot(p1.reset_index(drop=True), 'black')
plt.plot(y_hat, 'blue')

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='tanh')
mlp.fit(X, p1.reset_index(drop=True))
y_hat = mlp.predict(X)

plt.plot(p1.reset_index(drop=True), 'black')
plt.plot(y_hat, 'blue')

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200)

rf.fit(X, p1.reset_index(drop=True))
y_hat = rf.predict(X)
