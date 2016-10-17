import json

from sys import path as syspath
from os.path import dirname, abspath

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


def establish_client(ip, port, topic):
    client = mqtt.Client()
    client.on_connect = mqtt_on_connect

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


if __name__ == '__main__':

    # Static
    host_ip = '52.204.229.101'
    host_port = 1883
    mqtt_topic = 'fitai'

    mqtt_client = establish_client(host_ip, host_port, mqtt_topic)

    header = {"header": {"lift_id": '-1', "lift_sampling_rate": 50, "collar_id": "-1"}}
    data = {"content": {"a_x": [0, 0, 0, 0, 0, 0, 0]}}
    packet = dict(dict(**header), **data)

    print 'pushing reset through pipeline'
    publish(mqtt_client, mqtt_topic, message=json.dumps(packet))

    # run_client(mqtt_client)
    kill_client(mqtt_client)

