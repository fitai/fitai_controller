import paho.mqtt.client as mqtt

from sys import argv, path as syspath, exit
from os.path import dirname, abspath
from optparse import OptionParser
from json import loads
from datetime import datetime as dt
from pandas import Series

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.database import push_to_db
from databasing.redis_controls import establish_redis_client, retrieve_collar_by_id, update_collar_by_id
from comms.php_process_data import process_data
from processing.util import read_header_mqtt, read_content_mqtt
from comms.ws_publisher import ws_pub
from processing.ml_test import find_threshold, calc_reps

# TODO: Move this in to relevant functions
thresh = find_threshold()
collar_id = 0

# NOTE TO SELF: NEED A BETTER WAY TO MAKE THIS GLOBAL
# should probably turn the entire script into an object....
# Attempt to connect to redis server
redis_client = establish_redis_client(hostname='52.204.229.101', verbose=True)
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

        print 'reading header...'
        head = read_header_mqtt(data)
        print 'header contains: \n{}'.format(head)

        collar = retrieve_collar_by_id(redis_client, head['collar_id'])
        # The only piece of information from the device not provided by the frontend:
        collar['lift_sampling_rate'] = head['lift_sampling_rate']

        # TODO: Don't like doing all these checks. Think of a more efficient way...
        # If collar is newly generated, threshold will be 'None'
        if collar['threshold'] == 'None':
            collar['threshold'] = thresh

        if collar['lift_start'] == 'None':
            collar['lift_start'] = dt.strftime(dt.now(), '%Y-%m-%d')

        print 'collar contains: \n{}'.format(collar)

        print 'reading content...'
        accel = read_content_mqtt(data, collar)

        # This will ONLY happen if reset_reps.py is triggered, which means the only action that needs to be taken
        # is to zero out the reps
        if head['lift_id'] != 'None':
            print 'resetting reps'
            collar['calc_reps'] = 0

        # Before taking the time to push to db, process the acceleration and push to PHP websocket
        _, v, p = process_data(collar, accel)
        reps, curr_state = calc_reps(p, collar['calc_reps'], collar['curr_state'], collar['threshold'])
        # reps = 0
        # update state of user via 'collar' dict
        collar['calc_reps'] = reps
        collar['curr_state'] = curr_state

        if 'active' not in collar.keys():
            print 'collar {} has no Active field set. Will create and set to False'.format(collar['collar_id'])
            collar['active'] = False

        ws_pub(collar, v, p, reps)

        update_collar_by_id(redis_client, collar, collar['collar_id'], verbose=True)

        # temporarily disabling
        if collar['active']:
            print 'TEST: would push to db'
            # header = Series(data=collar)
            # push_to_db(header, accel)
        else:
            print 'TEST: would NOT push to db'
            # print 'Received and processed data for collar {}, but collar is not active...'.format(collar['collar_id'])

    except KeyError, e:
        print 'Key not found in data header. ' \
              'Cannot retrieve relevant information and update redis object.'
        print 'Error message: \n{}'.format(e)
    except ValueError, e:
        print 'Error processing JSON object. Message: \n{}'.format(str(e))
        print 'received: {}'.format(str(msg.payload))
    except TypeError, e:
        print 'Error processing string input. Message: \n{}'.format(str(e))
        print 'received: {}'.format(msg.payload)


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
