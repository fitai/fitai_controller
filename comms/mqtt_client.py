import paho.mqtt.client as mqtt

from sys import argv, path as syspath, exit
from os.path import dirname, abspath
from optparse import OptionParser
from json import loads, dump
from datetime import datetime as dt
from pandas import DataFrame

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.database_push import push_to_db
from databasing.redis_controls import establish_redis_client, retrieve_collar_by_id, update_collar_by_id, get_default_collar
from processing.util import read_header_mqtt, read_content_mqtt, process_data
from comms.ws_publisher import ws_pub
from processing.ml_test import find_threshold, calc_reps

# TODO: Turn this entire file into a class. Will allow us to use objects like the redis_client
# TODO: Push thresh_dict load into separate file
# as class attributes instead of forcing us to keep them global
#: Alpha = learning rate - make smaller to learn slower and take more iterations, make larger to learn faster and
#: risk non-convergence
fname = 'thresh_dict.txt'
try:
    tmp = open(fname, 'r')
    thresh_dict = loads(tmp.read())
    tmp.close()
    print 'Loaded thresh_dict from file'
except IOError:
    print 'Couldnt find saved thresh_dict file'
    thresh_dict = find_threshold(alpha=0.05, smooth=True, plot=False, verbose=False)
    with open('thresh_dict.txt', 'w') as outfile:
        dump(thresh_dict, outfile)

# NOTE TO SELF: NEED A BETTER WAY TO MAKE THIS GLOBAL
# should probably turn the entire script into an object....
# Attempt to connect to redis server
redis_client = establish_redis_client(hostname='localhost', verbose=True)
# redis_client = establish_redis_client(hostname='52.204.229.101', verbose=True)  # in case conn to server is needed

# If connection fails, MQTT client will not be able to update collar object, and will be useless. Kill and try again
if redis_client is None:
    print 'Couldnt connect to redis. Killing MQTT client.'
    exit(100)


# The callback for when the client successfully connects to the broker
def mqtt_on_connect(client, userdata, rc):
    ''' We subscribe on_connect() so that if we lose the connection
        and reconnect, subscriptions will be renewed. '''

    # client.subscribe('fitai')
    print 'connected'


#: The callback for when a PUBLISH message is received from the broker
#: There will need to be a lot of logic wrappers here; a lot could go wrong, and it should all be handled
#: as gracefully as possible
def mqtt_on_message(client, userdata, msg):

    topic = msg.topic
    print 'Received message from topic "{}"'.format(topic)

    try:
        data = loads(msg.payload)

        head = read_header_mqtt(data)

        # TODO: If collar returns without all necessary fields, what should happen??
        collar = retrieve_collar_by_id(redis_client, head['collar_id'])
        # Quick check that at least one expected field is in collar object
        if 'p_thresh' not in collar.keys():
            print 'Redis collar object {} appears broken. ' \
                  'Will replace with default and update as needed.'.format(collar['collar_id'])
            collar_tmp = collar.copy()
            collar = get_default_collar()
            collar.update(collar_tmp)

        # The only piece of information from the device not provided by the frontend:
        collar['sampling_rate'] = head['sampling_rate']

        # TODO: Don't like doing all these checks. Think of a more efficient way...
        # If collar is newly generated, threshold will be 'None'
        if any([(collar[col] == 'None') for col in ['a_thresh', 'v_thresh', 'p_thresh']]):
            print 'Missing at least one signal threshold. Resetting all...'
            try:
                # try to extract lift_type
                lift_thresh = thresh_dict[collar['lift_type']]
                collar['a_thresh'] = lift_thresh['a_thresh']
                collar['v_thresh'] = lift_thresh['v_thresh']
                collar['p_thresh'] = lift_thresh['p_thresh']
            except KeyError:
                print 'Couldnt find any thresholds for lift_type {}. Defaulting to 1.'.format(collar['lift_type'])
                collar['a_thresh'], collar['v_thresh'], collar['p_thresh'] = 1., 1., 1.

        if collar['lift_start'] == 'None':
            collar['lift_start'] = dt.now()

        #: Should only happen with default collar initialization
        if collar['collar_id'] == 'None':
            collar['collar_id'] = head['collar_id']

        if 'athlete_id' in head.keys():
            collar['athlete_id'] = head['athlete_id']

        #: Left over from old collar format. Shouldn't need this forever - remove key "threshold" if exists
        collar.pop('threshold', None)

        print 'collar contains: \n{}'.format(collar)

        # print 'reading content...'
        accel = read_content_mqtt(data, collar)

        # This will ONLY happen if reset_reps.py is triggered, which means the only action that needs to be taken
        # is to zero out the reps
        if head['lift_id'] != 'None':
            print 'resetting reps'
            collar['calc_reps'] = 0

        # Before taking the time to push to db, process the acceleration and push to PHP websocket
        print 'p_thresh: {}'.format(collar['p_thresh'])
        a, v, p = process_data(collar, accel, RMS=False)
        reps, curr_state, crossings = calc_reps(
            a, v, p, collar['calc_reps'], collar['curr_state'],
            collar['a_thresh'], collar['v_thresh'], collar['p_thresh'])

        # Assign timepoints to crossings, if there are any
        if crossings is not None:
            if crossings.shape[0] > 0:
                crossings['timepoint'] = (collar['max_t'] + crossings.index*(1./collar['lift_sampling_rate'])).values
                crossings['lift_id'] = collar['lift_id']
        # except AttributeError, e:
        #     print 'Crossings does not exist'
        #     print e

        # reps = 0
        # update state of user via 'collar' dict
        collar['calc_reps'] = reps
        collar['curr_state'] = curr_state
        collar['max_t'] += len(accel) * 1./collar['sampling_rate']  # track the last timepoint

        if 'active' not in collar.keys():
            # print 'collar {} has no Active field set. Will create and set to False'.format(collar['collar_id'])
            collar['active'] = False

        ws_pub(collar, v, p, reps)

        _ = update_collar_by_id(redis_client, collar, collar['collar_id'], verbose=True)

        if collar['active']:
            header = DataFrame(data=collar, index=[0]).drop(
                ['active', 'calc_reps', 'collar_id', 'curr_state',
                 'a_thresh', 'v_thresh', 'p_thresh', 'max_t'], axis=1)
            # Temporary to avoid pushing old field into database
            if 'lift_num_reps' in header.columns:
                header = header.drop('lift_num_reps', axis=1)

            # print 'header has: \n{}'.format(header)
            push_to_db(header, accel, crossings)
        else:
            print 'Received and processed data for collar {}, but collar is not active...'.format(collar['collar_id'])

    except KeyError, e:
        print 'Key not found in data header. ' \
              'Cannot retrieve relevant information and update redis object.'
        print 'Error message: \n{}'.format(e)
    except ValueError, e:
        print 'Error processing JSON object. Message: \n{}'.format(str(e))
        print 'received: {}'.format(str(msg.payload))
    except TypeError, e:
        print 'received: {}'.format(msg.payload)
        print 'Error processing string input. Message: \n{}'.format(str(e))


def establish_mqtt_client(ip, port, topic):
    client = mqtt.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message

    print 'Connecting MQTT client...'
    client.connect(ip, port, 60)  # AWS IP
    print 'Subscribing to topic "{}"'.format(topic)
    client.subscribe(topic=topic, qos=2)
    print 'MQTT client ready'
    return client


def run_client(client):
    print 'Looping MQTT client...'
    client.loop_forever()
    print 'Done looping??'


def kill_client(client):
    print 'Disconnecting MQTT client...'
    client.disconnect()
    print 'Disconnected.'


# Establish default behaviors of command-line call
def establish_cli_parser():
    parser = OptionParser()
    parser.add_option('-p', '--port', dest='host_port', default=1883,
                      help='Port on server hosting MQTT')
    parser.add_option('-i', '--ip', dest='host_ip', default='52.204.229.101',
                      help='IP address of server hosting MQTT')
    parser.add_option('-t', '--topic', dest='mqtt_topic', default='fitai',
                      help='MQTT topic messages are to be received from')
    parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')
    return parser


def main(args):

    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    host_ip = cli_options.host_ip
    host_port = cli_options.host_port
    mqtt_topic = cli_options.mqtt_topic
    verbose = cli_options.verbose

    if verbose:
        print 'Received args {}'.format(argv)
        print 'Attempting MQTT connection to {i}:{p} on topic {t}'.format(i=host_ip, p=host_port, t=mqtt_topic)

    mqtt_client = establish_mqtt_client(host_ip, host_port, mqtt_topic)
    run_client(mqtt_client)
    kill_client(mqtt_client)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
