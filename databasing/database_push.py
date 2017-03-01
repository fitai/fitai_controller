from sqlalchemy import create_engine
from pandas import read_sql
from numpy import abs, round
from sqlalchemy.exc import ProgrammingError, OperationalError, IntegrityError

from db_conn_strings import conn_string

# TODO: move this in to the proper functions
# Global for now. Should be fixed..
conn = create_engine(conn_string)


def push_to_db(header, content, crossings):
    if header is not None:
        print 'pushing collar metadata to db...'
        try:
            header.to_sql('athlete_lift', conn, if_exists='append', index=False, index_label='lift_id')
        except OperationalError, e:
            print '!!!!!Could not push collar metadata to database!!!!'
            print 'Likely because PostgreSQL server not running.\nError message: {}'.format(e)
            print 'skipping push to athlete_lift and resuming MQTT listening'
            return None
        except IntegrityError, e:
            print '!!!!! Could not push collar metadata to database !!!!!'
            print 'Likely because lift_id already exists in athlete_lift. \nError message: {}'.format(e)
            print 'Moving forward without pushing header into athlete_lift...'
        else:
            print 'collar metadata push successful. moving on to pushing content...'

    if content is not None:
        lift_id = content.lift_id.unique()[0]
        try:
            ts = read_sql(
                'SELECT timepoint::DOUBLE PRECISION FROM lift_data WHERE lift_id = {id} ORDER BY timepoint DESC LIMIT 2'.format(id=lift_id), conn)['timepoint']
            # NOTE: Console prints max_t = 19.89999999999999 - will this cause a rounding error??
            if len(ts) > 1:
                max_t = float(round(max(ts), 2))
                delta_t = abs(ts[1] - ts[0])
                print 'lift_id: {id}, max_t: {t}'.format(id=lift_id, t=max_t)
                if max_t is None:
                    max_t = 0.
            else:
                max_t = 0.
                delta_t = 0.
        except ProgrammingError:
            print 'table lift_data does not exist...'
            max_t = 0.
            delta_t = 0.

        content.timepoint += (max_t + delta_t)  # Have to step up from max_t because timepoint[0] = 0
        print 'Pushing new content values to db...'
        try:
            content.to_sql('lift_data', conn, if_exists='append', index=False, index_label=['lift_id', 'timepoint'])
        except OperationalError, e:
            print '!!!!!Count not push content data to database!!!!!'
            print ':\'-('
            print 'Error: {}'.format(e)
        except IntegrityError, e:
            print '!!!!! Could not push content data to database!!!!!'
            print 'Potential primary key overlap on (lift_id, timepoint)'
            print 'Error: {}'.format(e)
        else:
            print 'Done'

    #: 2/17 - Dont push crossings to DB until rep counter is figured out
    # if crossings is not None:
    #     if crossings.shape[0] > 0:
    #         print 'Pushing crossings to database...'
    #         try:
    #             crossings[['lift_id', 'timepoint', 'event']].to_sql(
    #                 'lift_event', conn, if_exists='append', index=False, index_label=['lift_id', 'timepoint'])
    #         except OperationalError, e:
    #             print '!!!!!Could not push crossings data to database!!!!'
    #             print 'Likely because PostgreSQL server not running.\nError message: {}'.format(e)
    #             print 'skipping push to lift_event and resuming MQTT listening'
    #             return None
    #         except IntegrityError, e:
    #             print '!!!!! Could not push crossings to database !!!!!'
    #             print 'Likely because (lift_id, timepoint) combo already exists in lift_event. ' \
    #                   '\nError message: {}'.format(e)
    #             print 'Moving forward without pushing crossings into lift_event...'
    #         else:
    #             print 'Done'


def update_calc_reps(collar):
    sql = '''
    UPDATE athlete_lift SET calc_reps = {cr}::NUMERIC
    WHERE lift_id = {l}::INT;
    '''.format(cr=collar['calc_reps'], l=collar['lift_id'])

    read_sql(sql, conn)
