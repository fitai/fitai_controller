from sqlalchemy import create_engine
from pandas import read_sql
# from numpy import abs, round
from sqlalchemy.exc import OperationalError

from databasing.db_conn_strings import aws_conn_string

# TODO: move this in to the proper functions
# Global for now. Should be fixed..
conn = create_engine(aws_conn_string)


def pull_data_by_lift(lift_id):
    dat_query = '''
    SELECT
        *
    FROM lift_data
    WHERE lift_id = {}
    '''.format(lift_id)

    header_query = '''
    SELECT
        *
    FROM athlete_lift
    WHERE lift_id = {}
    '''.format(lift_id)
    try:
        dat = read_sql(dat_query, conn)
        header = read_sql(header_query, conn)
    except OperationalError, e:
        print 'Unable to connect to database. Check firewall settings?'
        print e
        print 'Returning None...'
        header, dat = None, None
    return header, dat


# Optional input for user name; should allow for tailoring training later
def pull_lift_ids(user=None):
    query = '''
        SELECT
            al.lift_type,
            ARRAY_AGG(al.lift_id) AS lift_ids
        FROM athlete_lift AS al
        INNER JOIN athlete_info AS ai
            ON al.athlete_id = ai.athlete_id
        '''
    if user is not None:
        query += 'WHERE ai.athlete_name = {}'.format(user)

    query += 'GROUP BY al.lift_type'

    try:
        ids = read_sql(query, conn)
    except OperationalError, e:
        print 'Unable to connect to database. Check firewall settings?'
        print e
        print 'Returning None...'
        ids = None

    return ids
