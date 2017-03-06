from sys import path as syspath, argv
from os.path import dirname, abspath
from optparse import OptionParser


try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.redis_controls import retrieve_collar_by_id, update_collar_by_id, establish_redis_client
from databasing.redis_conn_strings import redis_host


def reset_reps(collar_id):
    redis_client = establish_redis_client(hostname=redis_host)

    collar = retrieve_collar_by_id(redis_client, collar_id)
    collar['calc_reps'] = 0

    print 'pushing reset through pipeline'

    res = update_collar_by_id(redis_client, collar, collar['collar_id'], verbose=False)

    if res:
        print 'Successfully reset reps on collar {}'.format(collar_id)
    else:
        print 'Failed to reset reps on collar {}'.format(collar_id)


# Establish default behaviors of command-line call
def establish_cli_parser():
    parser = OptionParser()
    parser.add_option('-c', '--collar', dest='collar_id', default=None,
                      help='collar_id to reset reps of')
    return parser


def main(args):

    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    collar_id = cli_options.collar_id

    reset_reps(collar_id)


if __name__ == '__main__':
    main(argv[1:])
