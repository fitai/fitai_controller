import json

from numpy import random

from sys import argv, path as syspath
from os.path import dirname, abspath
from optparse import OptionParser
from time import sleep

import paho.mqtt.client as mqtt

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'


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


def establish_client(ip, port, topic):
    client = mqtt.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message

    print 'Connecting MQTT client...'
    client.connect(ip, port, 60)  # AWS IP
    # print 'Connection successful'
    # client.connect('72.227.147.224', 1883, 60)
    # client.connect('localhost', 1883, 60)
    # print 'Subscribing to topic "{}"'.format(topic)
    # client.subscribe(topic=topic, qos=2)
    print 'MQTT client ready'
    return client


def kill_client(client):
    print 'Disconnecting MQTT client...'
    client.disconnect()
    print 'Disconnected.'


def publish(client, topic='fitai', message='test message'):
    print 'Publishing to topic {t} message:\n{m}'.format(t=topic, m=message)
    client.publish(topic=topic, payload=message)


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

    mqtt_client = establish_client(host_ip, host_port, mqtt_topic)

    mu = 2
    var = 0.75
    sleep_time = 0.05
    if verbose:
        print 'Publishing data with mean {} and variance {}'.format(mu, var)

    for i in range(int(100./sleep_time)):
        print 'Loop {}'.format(i)
        a_test = [random.normal(mu, var) for _ in range(30)]
        if i % (10./sleep_time) == 0:
            header = {"header": {"lift_id": '-1', "lift_sampling_rate": 50, "collar_id": "-1"}}
            data = {"content": {"a_x": [0, 0, 0, 0]}}
        else:
            header = {"header": {"lift_id": "None", "lift_sampling_rate": 50, "collar_id": "-1"}}
            data = {"content": {"a_x": a_test}}

        packet = dict(dict(**header), **data)

        publish(mqtt_client, mqtt_topic, message=json.dumps(packet))
        sleep(sleep_time)

    kill_client(mqtt_client)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
