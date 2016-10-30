from sqlalchemy import create_engine
import pandas as pd

from db_conn_strings import aws_conn_string


conn = create_engine(aws_conn_string)

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