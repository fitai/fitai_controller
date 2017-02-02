from sqlalchemy import create_engine
import pandas as pd

from databasing.db_conn_strings import conn_string


conn = create_engine(conn_string)

#: Transfer over any lift_data not currently in the backup
query = '''
SELECT
    *
FROM lift_data
'''

tmp = pd.read_sql(query, conn)

tmp.to_sql('lift_data_backup', conn, index=False, if_exists='append')


t = pd.read_sql('SELECT * FROM lift_data_backup')

temp = pd.read_sql('SELECT * FROM athlete_lift ORDER BY lift_id ASC', conn)

temp.to_sql('athlete_lift_backup', conn, index=False, if_exists='replace')