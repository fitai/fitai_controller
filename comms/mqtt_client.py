import paho.mqtt.client as mqtt

from sys import argv, path as syspath, exit
from os.path import dirname, abspath
from optparse import OptionParser
from json import loads
from pandas import DataFrame

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.database_push import push_to_db
from databasing.redis_controls import establish_redis_client, retrieve_collar_by_id, update_collar_by_id
from processing.util import read_header_mqtt, read_content_mqtt, process_data, prep_collar
from ml.thresh_learn import calc_reps, load_thresh_dict
from databasing.conn_strings import redis_host
from comms.redis_pubsub import redis_pub

# TODO: Turn this entire file into a class. Will allow us to use objects like the redis_client
# TODO: Push thresh_dict load into separate file
# as class attributes instead of forcing us to keep them global
#: Alpha = learning rate - make smaller to learn slower and take more iterations, make larger to learn faster and
#: risk non-convergence

#: Optional arg fname, defaults to thresh_dict.txt
thresh_dict = load_thresh_dict(fname='thresh_dict.txt')

# NOTE TO SELF: NEED A BETTER WAY TO MAKE THIS GLOBAL
# should probably turn the entire script into an object....
# Attempt to connect to redis server
redis_client = establish_redis_client(hostname=redis_host, verbose=True)
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

        collar = prep_collar(retrieve_collar_by_id(redis_client, head['collar_id']), head, thresh_dict)

        accel = read_content_mqtt(data, collar)

        # NOTE: process_data() returns accel, vel, power, position. All those returns are useful for
        #       calc_reps(), but are re-calculated differently before being pushed to websocket,
        #       so I decided just to pass them straight through to the calc_reps function
        collar, crossings = calc_reps(process_data(collar, accel, RMS=False, highpass=True), collar)

        # ws_pub(collar, process_data(collar, accel, RMS=True, highpass=True))
        redis_pub(redis_client, 'lifts', collar, process_data(collar, accel, RMS=True, highpass=True), source='real_time')

        _ = update_collar_by_id(redis_client, collar, collar['collar_id'], verbose=True)

        if collar['active']:
            header = DataFrame(data=collar, index=[0]).drop(
                ['active', 'collar_id', 'curr_state', 'a_thresh', 'v_thresh', 'pwr_thresh', 'pos_thresh', 'max_t'],
                axis=1)
            # print 'would push to db here'
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
    client.connect(ip, port, 60)
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
    parser.add_option('-i', '--ip', dest='host_ip', default='localhost',
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
