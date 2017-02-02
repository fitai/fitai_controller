from sqlalchemy import create_engine

from databasing.db_conn_strings import conn_string


conn = create_engine(conn_string)


# #: Trying to make this a utility function that can transfer any data between any two tables..
# def transfer_new_data(conn, from_table='lift_data', to_table='lift_data_backup', key_field='lift_id', ignore_entries_in_backup=True):
#     #: Transfer over any lift_data not currently in the backup
#     query = '''
#     SELECT
#         *
#     FROM {t1}
#     '''.format(t1=from_table)
#
#     if ignore_entries_in_backup:
#         query += 'WHERE {k} NOT IN (SELECT DISTINCT {k} FROM {t2)'.format(t2=to_table, k=key_field)
#
#     tmp = pd.read_sql(query, conn)
#
#     tmp.to_sql('to_table', conn, index=False, if_exists='append')


def backup_lift_data(conn):
    query = '''
    INSERT INTO lift_data_backup(lift_id, a_x, a_y, a_z, timepoint)
    SELECT lift_id, a_x, a_y, a_z, timepoint
    FROM lift_data
    WHERE lift_data.lift_id NOT IN (
        SELECT DISTINCT ldb.lift_id FROM lift_data_backup AS ldb
        );
    '''
    conn.execute(query)


def transfer_lift_data(conn):
    query = '''
    INSERT INTO lift_data(lift_id, a_x, a_y, a_z, timepoint)
    SELECT lift_id, a_x, a_y, a_z, timepoint
    FROM lift_data_temp
    WHERE lift_data_temp.lift_id NOT IN (
        SELECT DISTINCT ld.lift_id FROM lift_data AS ld
        );
    '''
    conn.execute(query)

# transfer_lift_data()


def transfer_arbitrary(conn, from_table, from_cols, to_table, to_cols, keys, ignore_entries_in_backup=True):
    #: Formulate the basic insert statement
    query = '''
    INSERT INTO {t2}({c2})
    SELECT {c1}
    FROM {t1}
    '''.format(t1=from_table, c1=from_cols, t2=to_table, c2=to_cols)

    #: Apply any filter keys provided
    if ignore_entries_in_backup:
        for key in keys:
            query += '''
            WHERE {t1}.{k} NOT IN (
                SELECT DISTINCT t.{k} FROM {t2} AS t
                )
            '''.format(t1=from_table, t2=to_table, k=key)

    #: May not be necessary to close out via semi-colon
    query += ';'

    # Hope it works
    conn.execute(query)

    print 'done'
