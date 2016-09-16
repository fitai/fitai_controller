from sqlalchemy import create_engine
import pandas as pd
from numpy import abs, round
from sqlalchemy.exc import ProgrammingError, OperationalError

# ex postgresql://admin:sacharters123@192.168.150.159:5432/smsport
# NOTE TO SELF - update this so that your password doesn't show!
aws_conn_string = "postgresql://db_user:dbuserpassword@test-db-instance.cls1x0o0bifh.us-east-1.rds.amazonaws.com:5432/fitai"
aws_conn = create_engine(aws_conn_string)

local_conn_string = "postgresql://localhost:5432/fitai"
local_conn = create_engine(local_conn_string)

# conn = aws_conn
conn = local_conn

# data = pd.read_csv('test_accel_dat.csv')
# data['lift_id'] = 0
# data = data.drop('time', axis=1)
# data = data.reset_index().rename(columns={
#     'index': 'timepoint', 'x-acceleration': 'a_x', 'y-acceleration': 'a_y', 'z-acceleration': 'a_z'})
#
# # Basic upload. Will want to write in checks and balances
# data.to_sql('lift_data', aws_conn, if_exists='append', index=False, index_label=['lift_id', 'timepoint'])


def push_to_db(header, content):
    if header is not None:
        print 'pushing header to db...'
        try:
            header.to_sql('athlete_lift', conn, if_exists='fail', index=False, index_label='lift_id')
        except OperationalError, e:
            print '!!!!!COULD NOT PUSH HEADER TO DATABASE!!!!'
            print 'Likely because PostgreSQL server not running.\n Error: {}'.format(e)
            print 'skipping push to athlete_lift and resuming MQTT listening'
            return None
        else:
            print 'header push successful. moving on to pushing content...'

    if content is not None:
        lift_id = content.lift_id.unique()[0]
        try:
            ts = pd.read_sql('SELECT timepoint::DOUBLE PRECISION FROM lift_data WHERE lift_id = {id} ORDER BY timepoint DESC LIMIT 2'.format(id=lift_id), conn)['timepoint']
            # NOTE: Console prints max_t = 19.89999999999999 - will this cause a rounding error??
            max_t = round(max(ts), 2)
            delta_t = abs(ts[1] - ts[0])
            print 'lift_id: {id}, max_t: {t}'.format(id=lift_id, t=max_t)
            if max_t is None:
                max_t = 0
        except ProgrammingError:
            print 'table lift_data does not exist...'
            max_t = 0

        # print 'lift_id exists with max time point {}'.format(max_t)
        content.timepoint += (max_t + delta_t)  # Have to step up from max_t because timepoint[0] = 0
        # print 'New content values: \n{}'.format(content.head())
        print 'pushing new content values to db...'
        content.to_sql('lift_data', conn, if_exists='append', index=False, index_label=['lift_id', 'timepoint'])
        print 'done'
