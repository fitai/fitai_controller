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
        collar = redis_client.get(dat['collar_id'])

        next_lift_id = redis_client.get('lift_id')
        if next_lift_id is None:
            # TODO: pull max lift_id from athlete_lift table instead of defaulting to 0
            # Want to be really careful here; could reset all lift_ids and throw everything off
            print 'No Redis variable "lift_id" found. Will set to 0'
            next_lift_id = '0'
            redis_client.set('lift_id', next_lift_id)

        if dat['lift_id'] == 'None':
            # lift_id = 'None' is sent to trigger new workout, which means lift_id needs to be updated.
            # DO iterate lift_id in this case
            update_lift_id = True
            for key in dat.keys():
                collar[key] = dat[key]
            collar['lift_id'] = next_lift_id
        else:
            # No lift_id field occurs when End Lift button is pressed, and we want to stop pushing data to db
            # In this case, just update collar object with new values and push to redis. DO NOT iterate lift_id
            update_lift_id = False
            for key in dat.keys():
                collar[key] = dat[key]

        response = update_collar_by_id(redis_client, dat, dat['collar_id'], verbose)

        if response & update_lift_id:
            # lift_id was 'None', and the redis collar object was successfully updated
            redis_client.incr('lift_id', 1)
        elif ~response:
            print 'Redis object not updated properly. Will not increment lift_id.'
        elif ~update_lift_id:
            print 'JSON object did not include lift_id. Should be a trigger to end lift and stop pushing to db'
        else:
            print 'SHOULDNT SEE THIS!?!'

    except KeyError, e:
        print 'Couldnt extract collar_id from json object. Cannot update.'
        if verbose:
            print 'Error message: \n{}'.format(e)
        exit(200)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
