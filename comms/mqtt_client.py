import json

from sys import argv, path as syspath
from os.path import dirname, abspath
from optparse import OptionParser
from urllib2 import urlopen

import paho.mqtt.client as mqtt

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.database import push_to_db
from comms.php_process_data import process_data
from processing.util import read_header_mqtt, read_content_mqtt
from comms.ws_publisher import ws_pub

my_ip = urlopen('http://ip.42.pl/raw').read()


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
        data = json.loads(msg.payload)

        print 'reading header...'
        head = read_header_mqtt(data)
        print 'reading content...'
        accel = read_content_mqtt(data, head)

        # Before taking the time to push to db, process the acceleration and push to PHP websocket
        _, v, p = process_data(head, accel)
        ws_pub(head, v, p)

        # temporarily disabling
        push_to_db(head, accel)

    except ValueError, e:
        print 'Error processing JSON object. Message: \n{}'.format(str(e))
        print 'received: {}'.format(str(msg.payload))
    except TypeError, e:
        print 'Error processing string input. Message: \n{}'.format(str(e))
        print 'recieved: {}'.format(msg.payload)


def establish_client(ip, port, topic):
    client = mqtt.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message

    print 'Connecting MQTT client...'
    client.connect(ip, port, 60)  # AWS IP
    # print 'Connection successful'
    # client.connect('72.227.147.224', 1883, 60)
    # client.connect('localhost', 1883, 60)
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


def main(args):
    cli_parser = OptionParser()
    cli_parser.add_option('-p', '--port', dest='host_port', default=1883,
                      help='Port on server hosting MQTT')
    cli_parser.add_option('-i', '--ip', dest='host_ip', default='52.204.229.101',
                      help='IP address of server hosting MQTT')
    cli_parser.add_option('-t', '--topic', dest='mqtt_topic', default='fitai',
                      help='MQTT topic messages are to be received from')
    cli_parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')

    (cli_options, _) = cli_parser.parse_args(args)

    host_ip = cli_options.host_ip
    host_port = cli_options.host_port
    mqtt_topic = cli_options.mqtt_topic
    verbose = cli_options.verbose

    if verbose:
        # print 'options (type {t}): {o}'.format(t=type(cli_options), o=cli_options)
        # print 'args: {}'.format(args)
        print 'Received args {}'.format(argv)
        print 'Attempting MQTT connection to {i}:{p} on topic {t}'.format(i=host_ip, p=host_port, t=mqtt_topic)

    mqtt_client = establish_client(host_ip, host_port, mqtt_topic)
    run_client(mqtt_client)
    kill_client(mqtt_client)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])

# Sample message:
# "{'header': {'athlete_id': 0, 'lift_id': 0, 'lift_sampling_rate': 50, 'lift_start': '2016-09-01', 'lift_type': 'deadlift', 'lift_weight': 100, 'lift_weight_units': 'lbs', 'lift_num_reps': 10 }, 'content': 'a_x': [0, 1, 2, 1, 3, 2, 1, 3, 4]}"
# '{"header": {"athlete_id": 0, "lift_id": 10, "lift_sampling_rate": 50, "lift_start": "2016-09-01", "lift_type": "deadlift", "lift_weight": 100, "lift_weight_units": "lbs", "lift_num_reps": 10 }, "content": {"a_x": [0, 1, 2, 1, 3, 2, 1, 3, 4]} }'
