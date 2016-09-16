from sqlalchemy import create_engine
import pandas as pd

# NOTE TO SELF - update this so that your password doesn't show!
aws_conn_string = "postgresql://db_user:dbuserpassword@test-db-instance.cls1x0o0bifh.us-east-1.rds.amazonaws.com:5432/fitai"
aws_conn = create_engine(aws_conn_string)

local_conn_string = "postgresql://kyle:Seda2012@localhost:5432"
local_conn = create_engine(local_conn_string)

# Switch between conns here
# conn = aws_conn
conn = local_conn

query = '''
SELECT
    *
FROM lift_data
--WHERE lift_id NOT IN (SELECT DISTINCT lift_id FROM lift_data_storage)
'''

tmp = pd.read_sql(query, conn)


temp = pd.read_sql('SELECT * FROM lift_data', conn)
temp.to_sql('lift_data_backup', conn, index=False, if_exists='replace')

# # WANT TO AVOID THIS
# id = tmp.lift_id.ix[0]
# ts = [float(x) for x in tmp.timepoint.ix[0][:200]]
# ax = list(tmp.a_x.ix[0][:200])

# temp = pd.DataFrame(data={'lift_id': id, 'timepoint': list(ts), 'a_x': list(ax)})

tmp.to_sql('lift_data_backup', conn, index=False, if_exists='append')

pd.read_sql('SELECT * FROM lift_data_backup')