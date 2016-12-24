from sqlalchemy import create_engine
from pandas import read_sql, DataFrame
from sys import argv, path as syspath
from os.path import dirname, abspath
from optparse import OptionParser

from sqlalchemy.exc import OperationalError

from db_conn_strings import conn_string

# TODO: move this in to the proper functions
# Global for now. Should be fixed..
conn = create_engine(conn_string)


# Utility function for other functions. Won't be seen by anything in database_pull.py
def pull_max_lift_id():
    query = 'SELECT MAX(lift_id) FROM athlete_lift'
    lift_id = read_sql(query, conn).iloc[0][0]
    return lift_id


#: Accepts lift_id, queries database for all data associated with that lift
#: Processes header and data into acceleration, velocity, and power
#: Pushes those vectors into a dataframe, adds index, then formats as JSON and outputs
# TODO: skip converting to dataframe and convert directly to json
# TODO: Put this in a different file or a different folder
def lift_to_json(lift_id):
    # Retrieve data from last lift, process into vel and power, push to frontend
    header, data = pull_data_by_lift(lift_id)
    a, v, p = process_data(header, data)

    data_out = DataFrame(data={'a_rms': a,
                               'v_rms': v,
                               'p_rms': p,
                               'timepoint': data['timepoint']},
                         index=a.index)

    return data_out.to_json(orient='split')


#: The workhorse function - pulls all acceleration data by lift_id
#: along with accompanying metadata for that lift
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
        # header returns as a dataframe, but want a dict. this converts.
        header = read_sql(header_query, conn).drop_duplicates().to_dict(orient='index')[0]

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


# Establish default behaviors of command-line call
def establish_cli_parser():
    parser = OptionParser()
    parser.add_option('-l', '--lift', dest='lift_id', default=None,
                      help='If provided, pull data for specified lift_id only')
    parser.add_option('-u', '--user', dest='user_name', default=None,
                      help='If provided, pull all data for specified user (by name)')
    parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')
    return parser


def main(args):

    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    lift_id = cli_options.lift_id
    user_name = cli_options.user_name
    verbose = cli_options.verbose

    if lift_id is not None:
        if verbose:
            print 'Retrieving data for lift_id: {}'.format(lift_id)
        print lift_to_json(lift_id)

    #: TODO: Collapse all lift_ids returned (split by lift_type) into single list of lift_ids
    elif user_name is not None:
        if verbose:
            print 'Retrieving all data for user_id: {}'.format(user_name)
        #: NOTE: returns DataFrame(lift_type, ARRAY(lift_id) AS lift_ids)
        #: If all you want is all lifts by user, collapse into single list of ids
        lift_ids = pull_lift_ids(user_name)
        # temp_df = DataFrame()
        for lift_id in lift_ids:
            # temp_df = temp_df.append(pull_data_by_lift(lift_id), ignore_index=True)
            print lift_to_json(lift_id)

        # TODO: Figure out how to format this so that PHP can accept it
        # print temp_df.to_json(orient='split')

#: NOTE: The only external (CLI) ping to this file will be from the PHP
#: This means that all flags should lead to a JSON output, and not to anything python-readable
if __name__ == '__main__':
    #: Don't need to add path if this is being called internally, so I
    #: placed it here
    try:
        path = dirname(dirname(abspath(__file__)))
        # print 'Adding {} to sys.path'.format(path)
        syspath.append(path)
    except NameError:
        syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
        print 'Working in Dev mode.'

    from processing.util import process_data

    main(argv[1:])
