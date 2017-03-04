from json import loads
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

from databasing.database_pull import lift_to_json, pull_max_lift_id
from databasing.database_push import update_calc_reps
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
    """
    update_redis.py is meant to be called exclusively from PHP - either when initiating a new lift (via the
    Submit form) or when ending a current lift session (via the End Lift button). In each circumstance,
    the PHP calls this script with a specific set of parameters, which dictate which part of the logic
    is triggered.

    Via Submit form:
    ---------------

    python comms/update_redis.py -v -j '{"collar_id": "555", "athlete_id": "1", "lift_id": "None", ... "active": True}'

    :param args: (-j) JSON string containing a lot of fields, most notably "collar_id", "active", and "lift_id"

    The fields listed are what are relevant to the switching logic in update_redis.py. There are more fields in the
    JSON string: they are pieces of metadata that needs to be attached to the lift, and are updated in the
    collar object, but do not impact anything here (other than getting incorporated into the collar object).

    The "collar_id" field tells update_redis.py which collar to grab/update. The "active" field doesn't have
    an impact in this script, but it tells update_redis.py to change the active state of the collar,
    which mqtt_client.py will interpret as a sign to START pushing any data received for that collar to the
    database. The "athlete_id" field also has no impact on update_redis.py, but will impact which PHP frontend
    is listening to the data pushed to the websocket from mqtt_client.py

    "lift_id": "None" is the key trigger in this JSON string. It triggers the logic
    "if dat['lift_id'] == "None" ", which tells update_redis.py that the user has triggered a new lift, and
    (importantly) tells update_redis.py to iterate the stored Redis lift_id object, which is how we track
    the most recent lift_id in a shared manner (all collars will have access to the Redis storage). The collar
    object is updated with all the relevant information contained in the passed JSON string, and is then
    saved to the Redis server. Assuming proper saving of the Redis object, the
    " if response & update_lift_id: " logic is triggered, which causes the "lift_id" Redis object to be incremented
    and also skips the logic that would trigger a Summary Screen.

    If there was a problem updating the Redis object, for whatever reason, " elif not response: " logic is
    triggered, which has the effect of printing a warning to stdout and then continuing operation as normal.

    ----- This should be altered such that any failed update of redis triggers a recursive call to
    update_redis.py with the same JSON string so that the update can be tried again. We don't want the user
     to lift if the redis object is collecting that data for the wrong lift ------

    :return: N/A

    ----------------------------------------

    Via End Lift button
    -------------------

    python comms/update_redis.py -v -j '{"collar_id":"555","active":false}'

    :param args: (-j) JSON string containing "collar_id" and "active"

    The "collar_id" field tells update_redis.py which collar to grab/update, and the "active" field
    tells update_redis.py to change the active state of the collar, which mqtt_client.py will interpret as
    a sign to STOP pushing any data received for that collar to the database.

    Note that field "lift_id" is NOT present in this JSON string. This triggers the "If lift_id not in dat.keys()"
    logic, which tells the script to disregard any potential change to lift_id info (i.e. don't iterate the
    max lift_id object stored in Redis), to wipe the athlete_id currently associated with that collar (because
    the lift is finished), and to switch the "active" field from true to False.

    The final if/elif/else logic block is then triggered. In this instance, the "elif not update_lift_id" should
    trigger, which calls lift_to_json and passes in whatever lift_id the collar has associated with it, which
    should be the lift_id of whichever lift was just ended. lift_to_json queries the database for data relevant
    to that lift_id, then processes it and prints to stdout, which is how the PHP gets ahold of it. The PHP then
    uses that printed data to build the Summary Screen.

    :return: N/A (prints to stdout)

    """
    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    dat = loads(cli_options.json_str)
    verbose = cli_options.verbose

    if verbose:
        print 'Received json: {}'.format(dat)

    # redis_client = establish_redis_client(hostname='localhost', verbose=verbose)
    redis_client = establish_redis_client(hostname='52.204.229.101', verbose=True)

    if redis_client is None:
        print 'Unsuccessful attempt to launch redis client. Cannot update.'
        exit(200)

    try:
        if verbose:
            print 'Found collar_id {}'.format(dat['collar_id'])
        collar = loads(redis_client.get(dat['collar_id']))

        next_lift_id = redis_client.get('lift_id')
        # In case redis can't be reached, can move forward assuming that the content of athlete_lift table
        # is reliable
        if next_lift_id is None:
            next_lift_id = pull_max_lift_id() + 1
            print 'No Redis variable "lift_id" found. Will set to {} (from athlete_lift)'.format(next_lift_id)
            redis_client.set('lift_id', next_lift_id)

        if 'lift_id' not in dat.keys():
            # No lift_id field occurs when End Lift button is pressed, and we want to stop pushing data to db
            # In this case, first update the athlete_lift table with the calculated number of reps,
            # then update collar object with new values and push to redis. DO NOT iterate lift_id
            update_lift_id = False
            for key in dat.keys():
                #: Temporary workaround until patrick renames this field
                # if key == 'lift_num_reps':
                collar['init_num_reps'] = dat[key]
                # else:
                #     collar[key] = dat[key]
            collar['athlete_id'] = 'None'

            # Update calc_reps in database with final calculated value
            if ('calc_reps' in collar.keys()) & (collar['calc_reps'] is not None):
                update_calc_reps(collar)
            # else:
            #     print 'Meant to update calc_reps in db, but collar {} does not contain valid calc_reps entry'.\
            #         format(collar['collar_id'])

        elif dat['lift_id'] == 'None':
            # lift_id = 'None' is sent to trigger new workout, which means lift_id needs to be updated.
            # DO iterate lift_id in this case
            update_lift_id = True
            for key in dat.keys():
                #: Temporary workaround until patrick renames this field
                if key == 'lift_num_reps':
                    collar['init_num_reps'] = dat[key]
                else:
                    collar[key] = dat[key]
            collar['active'] = True
            collar['lift_id'] = next_lift_id
        else:
            print 'sent update explicitly for lift_id {}, which is not currently handled.'.format(dat['lift_id'])
            update_lift_id = False
            collar = dat

        response = update_collar_by_id(redis_client, collar, collar['collar_id'], verbose)

        #: Switching logic to dictate whether or not the script should call up the info stored
        #: for whatever lift just ended.
        if response & update_lift_id:
            #: This is triggered when a Submit form is sent, and the user is about to START lifting
            # print 'Redis object updated properly. Will increment lift_id'
            print 'lift_id: {}'.format(collar['lift_id'])
            # lift_id was 'None', and the redis collar object was successfully updated
            redis_client.incr('lift_id', 1)
        elif not response:
            #: This is triggered when a Submit form is sent, but redis couldn't be updated properly.
            #: User intends to START lifting, but there may be technical issues, as Redis didn't update..
            print 'Redis object not updated properly. Will not increment lift_id.'
        elif not update_lift_id:
            #: Triggered when the End Lift button is triggered on the frontend.
            #: Indicates that the user intends to STOP lifting (or has already stopped).

            # In this case, we want to retrieve whatever information we just stored about the lift the user
            # just finished. We do this by calling a function (lift_to_json) that will pull any acceleration
            # data associated with the lift_id currently in the collar, will process it, and will print to
            # stdout so that the PHP can retrieve it.

            # print 'JSON object did not include lift_id'
            if verbose:
                print 'found lift_id: {}'.format(collar['lift_id'])
            print lift_to_json(collar['lift_id'])
        else:
            print 'SHOULDNT SEE THIS!?!'

    except KeyError, e:
        print 'Couldnt extract collar_id from json object. Cannot update.'
        if verbose:
            print 'Error message: \n{}'.format(e)
        # exit(200)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
