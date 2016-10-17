import json
from optparse import OptionParser
from sys import argv, path as syspath, exit
from os.path import dirname, abspath

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.redis_controls import establish_redis_client, update_collar_by_id


# Establish default behaviors of command-line call
def establish_cli_parser():
    parser = OptionParser()
    parser.add_option('-j', '--json', dest='json_str', default=None,
                      help='JSON string used to update redis object')
    parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')
    return parser


def main(args):
    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    dat = json.loads(cli_options.json_str)
    verbose = cli_options.verbose

    if verbose:
        print 'Received json: {}'.format(dat)

    redis_client = establish_redis_client(verbose=verbose)

    if redis_client is None:
        print 'Unsuccessful attempt to launch redis client. Cannot update.'
        exit(200)

    try:
        print 'Found collar_id {}'.format(dat['collar_id'])
        id = redis_client.get('lift_id')
        if id is None:
            print 'No Redis variable "lift_id" found. Will set to 0'
            id = '0'
            redis_client.set('lift_id', id)
        dat['lift_id'] = id
        update_collar_by_id(redis_client, dat, dat['collar_id'], verbose)
        redis_client.incr('lift_id', 1)
    except KeyError, e:
        print 'Couldnt extract collar_id from json object. Cannot update.'
        if verbose:
            print 'Error message: \n{}'.format(e)
        exit(200)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
