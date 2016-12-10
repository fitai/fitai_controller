from sqlalchemy import create_engine
import pandas as pd

from databasing.db_conn_strings import aws_conn_string


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

tmp.to_sql('lift_data_backup', conn, index=False, if_exists='append')

pd.read_sql('SELECT * FROM lift_data_backup')