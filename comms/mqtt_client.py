import json
import time
import sys
import os
from optparse import OptionParser

from datetime import datetime as dt
from urllib2 import urlopen

import paho.mqtt.client as mqtt

try:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    sys.path.append(path)
except NameError:
    sys.path.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'working in Dev mode.'

from databasing.database import push_to_db
from comms.php_process_data import process_data
from processing.util import read_header_mqtt, read_content_mqtt
from comms.ws_publisher import ws_pub

my_ip = urlopen('http://ip.42.pl/raw').read()


# The callback for when the client successfully connects to the broker
def mqtt_on_connect(client, userdata, rc):
    ''' We subscribe on_connect() so that if we lose the connection
        and reconnect, subscriptions will be renewed. '''

    print 'connection successful'
    print 'ready'
    client.subscribe("fitai")


#: The callback for when a PUBLISH message is received from the broker
#: There will need to be a lot of logic wrappers here; a lot could go wrong, and it should all be handled
#: as gracefully as possible
def mqtt_on_message(client, userdata, msg):

    topic = msg.topic
    print 'Received message from topic "{}"'.format(topic)
    # print 'received message (type {t}): {m}'.format(t=type(msg.payload), m=msg.payload)

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


# Test script to publish to a broker of choice
def test_recurring_messages(client):
    i = 0
    while i < 5:
        print 'pushing message {} to broker'.format(i)
        write_message = '({i}) Message from IP {ip}. \nCurrent time: {t}'.format(i=i, ip=my_ip, t=dt.now())
        client.publish('fitai', write_message)
        time.sleep(5)
        i += 1


def main(argv):

    parser = OptionParser()
    parser.add_option('-p', '--port', dest='host_port', default=1883,
                      help='Port on server hosting MQTT')
    parser.add_option('-i', '--ip', dest='host_ip', default='52.204.229.101',
                      help='IP address of server hosting MQTT')
    parser.add_option('-t', '--topic', dest='mqtt_topic', default='fitai',
                      help='MQTT topic messages are to be received from')
    parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')

    (options, args) = parser.parse_args(argv)

    print 'options (type {t}): {o}'.format(t=type(options), o=options)
    # print 'args: {}'.format(args)

    host_ip = options.host_ip
    host_port = options.host_port
    mqtt_topic = options.mqtt_topic
    verbose = options.verbose

    if verbose:
        print 'received args {}'.format(argv)
        print 'Attempting MQTT connection to {i}:{p} on topic {t}'.format(i=host_ip, p=host_port, t=mqtt_topic)

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = mqtt_on_connect
    mqtt_client.on_message = mqtt_on_message

    print 'connecting client...'
    mqtt_client.connect(host_ip, host_port, 60)  # AWS IP
    # client.connect('72.227.147.224', 1883, 60)
    # client.connect('localhost', 1883, 60)
    mqtt_client.subscribe(topic=mqtt_topic, qos=2)

    mqtt_client.loop_forever()
    mqtt_client.disconnect()

# Receives initial ping to file
if __name__ == '__main__':
    main(sys.argv[1:])

# Sample message:
# "{'header': {'athlete_id': 0, 'lift_id': 0, 'lift_sampling_rate': 50, 'lift_start': '2016-09-01', 'lift_type': 'deadlift', 'lift_weight': 100, 'lift_weight_units': 'lbs', 'lift_num_reps': 10 }, 'content': 'a_x': [0, 1, 2, 1, 3, 2, 1, 3, 4]}"
# '{"header": {"athlete_id": 0, "lift_id": 10, "lift_sampling_rate": 50, "lift_start": "2016-09-01", "lift_type": "deadlift", "lift_weight": 100, "lift_weight_units": "lbs", "lift_num_reps": 10 }, "content": {"a_x": [0, 1, 2, 1, 3, 2, 1, 3, 4]} }'
