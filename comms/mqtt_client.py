import json
import time
import sys
import os

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

# PERSONAL NOTES

# To run the mosquitto service on mac osx
# /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf


# The callback for when the client successfully connects to the broker
def on_connect(client, userdata, rc):
    ''' We subscribe on_connect() so that if we lose the connection
        and reconnect, subscriptions will be renewed. '''

    print 'connection successful'
    print 'ready'
    client.subscribe("fitai")


#: The callback for when a PUBLISH message is received from the broker
#: There will need to be a lot of logic wrappers here; a lot could go wrong, and it should all be handled
#: as gracefully as possible
def on_message(client, userdata, msg):

    topic = msg.topic
    print 'Received message from topic "{}"'.format(topic)
    # print 'received message (type {t}): {m}'.format(t=type(msg.payload), m=msg.payload)

    try:
        data = json.loads(msg.payload)
        # print 'Received json data \n{}'.format(data)

        # To save the most recent data packet so it can be loaded in for dev purposes
        # in case code breaks downstream
        # with open('test_dict.txt', 'w') as outfile:
        #     json.dump(data, outfile)

        # Load saved data packet
        # data = dict(data)

        print 'reading header...'
        head = read_header_mqtt(data)
        print 'reading content...'
        accel = read_content_mqtt(data, head)

        # Before taking the time to push to db, process the acceleration and push to PHP websocket
        _, v, p = process_data(head, accel)
        ws_pub(head, v, p)

        push_to_db(head, accel)

    except ValueError, e:
        print 'Error processing JSON object. Message: \n{}'.format(str(e))
        print 'received: {}'.format(str(msg.payload))
    except TypeError, e:
        print 'Error processing string input. Message: \n{}'.format(str(e))
        print 'recieved: {}'.format(msg.payload)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print 'connecting client...'
# client.connect("52.204.229.101", 1883, 60)  # AWS IP
# client.connect('72.227.147.224', 1883, 60)
client.connect('localhost', 1883, 60)
client.subscribe(topic='fitai', qos=2)


# Test script to publish to a broker of choice
def test_recurring_messages(client):
    i = 0
    while i < 5:
        print 'pushing message {} to broker'.format(i)
        write_message = '({i}) Message from IP {ip}. \nCurrent time: {t}'.format(i=i, ip=my_ip, t=dt.now())
        client.publish('fitai', write_message)
        time.sleep(5)
        i += 1

# Blocking call which processes all network traffic and dispatches
# callbacks (see on_*() above). It also handles reconnecting.

client.loop_forever()
client.disconnect()

# Sample message:
# "{'header': {'athlete_id': 0, 'lift_id': 0, 'lift_sampling_rate': 50, 'lift_start': '2016-09-01', 'lift_type': 'deadlift', 'lift_weight': 100, 'lift_weight_units': 'lbs', 'lift_num_reps': 10 }, 'content': 'a_x': [0, 1, 2, 1, 3, 2, 1, 3, 4]}"
#'{"header": {"athlete_id": 0, "lift_id": 0, "lift_sampling_rate": 50, "lift_start": "2016-09-01", "lift_type": "deadlift", "lift_weight": 100, "lift_weight_units": "lbs", "lift_num_reps": 10 }, "content": {"a_x": [0, 1, 2, 1, 3, 2, 1, 3, 4]} }'
