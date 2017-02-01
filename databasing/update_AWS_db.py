from sqlalchemy import create_engine
import pandas as pd

# NOTE TO SELF - update this so that your password doesn't show!
from databasing.db_conn_strings import aws_conn_string, local_conn_string

aws_conn = create_engine(aws_conn_string)
local_conn = create_engine(local_conn_string)

# Switch between conns here
# conn = aws_conn
conn = local_conn

# data = pd.read_csv('test_accel_dat.csv')
# data['lift_id'] = 0
# data = data.drop('time', axis=1)
# data = data.reset_index().rename(columns={
#     'index': 'timepoint', 'x-acceleration': 'a_x', 'y-acceleration': 'a_y', 'z-acceleration': 'a_z'})
#
# # Basic upload. Will want to write in checks and balances
# data.to_sql('lift_data', conn, if_exists='append', index=False, index_label=['lift_id', 'timepoint'])

pd.read_sql('SELECT * FROM lift_data', local_conn)