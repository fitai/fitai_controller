# To learn how to identify rep start/stop points
import pandas as pd

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
for id in lift_ids:
    header, data = pull_data_by_lift(id)

    a, v, pwr, pos = process_data(header, data, RMS=False, verbose=True)

    d = a.join(v).join(pwr).join(pos)
    d['timepoint'] = data['timepoint']
    d['lift_id'] = header['lift_id']

    dat = dat.append(d, ignore_index=True)
    metadata = metadata.append(pd.Series(header), ignore_index=True)

#: Join in relevant data
events = pd.merge(events, metadata, on='lift_id', how='left')

ts = 1.
event_dat = pd.DataFrame()
for i, row in events.iterrows():
    #: everything within +/- ts of event timepoint
    tmp = dat.loc[(dat['timepoint'] >= row['timepoint'] - ts) & (dat['timepoint'] <= row['timepoint'] + ts)]
    tmp['event_id'] = i

    event_dat = event_dat.append(tmp.drop('lift_id', axis=1), ignore_index=True)


#: Split into starts and stops - for use later
starts = events.loc[events['event'].eq('rep_start')].reset_index(drop=True)
stops = events.loc[events['event'].eq('rep_stop')].reset_index(drop=True)
