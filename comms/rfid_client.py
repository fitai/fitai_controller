import paho.mqtt.client as mqtt

from sys import argv, path as syspath, exit
from os.path import dirname, abspath
from optparse import OptionParser
from json import loads

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.redis_controls import establish_redis_client
from databasing.conn_strings import redis_host
from comms.redis_pubsub import redis_pub

# TODO: Turn this entire file into a class. Will allow us to use objects like the redis_client
# TODO: NEED A BETTER WAY TO MAKE THIS GLOBAL
# Attempt to connect to redis server
redis_client = establish_redis_client(hostname=redis_host, verbose=True)

# If connection fails, MQTT client will not be able to update collar object, and will be useless. Kill and try again
if redis_client is None:
    print 'Couldnt connect to redis. Killing MQTT client.'
    exit(100)


# The callback for when the client successfully connects to the broker
def mqtt_on_connect(client, userdata, rc):
    print 'connected'


#: The callback for when a PUBLISH message is received from the broker
#: There will need to be a lot of logic wrappers here; a lot could go wrong, and it should all be handled
#: as gracefully as possible
def mqtt_on_message(client, userdata, msg):

    topic = msg.topic
    print 'Received message from topic "{}"'.format(topic)

    try:
        print 'Pushing message through to redis pubsub...'
        redis_pub(redis_client, 'rfid', None, msg.payload, source='rfid')

        # Don't need to do this, but it would be good to confirm the payload is properly formatted
        # data = loads(msg.payload)
        # print 'Received data: {}'.format(data)

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
    parser.add_option('-t', '--topic', dest='mqtt_topic', default='rfid',
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
