from sqlalchemy import create_engine
from pandas import read_sql
from numpy import abs, round
from sqlalchemy.exc import ProgrammingError, OperationalError, IntegrityError

from db_conn_strings import aws_conn_string

# TODO: move this in to the proper functions
# Global for now. Should be fixed..
conn = create_engine(aws_conn_string)


def pull_data_by_lift(lift_id):
    return read_sql('SELECT * FROM lift_data WHERE lift_id = {}'.format(lift_id), conn)
