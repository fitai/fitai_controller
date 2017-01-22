import json

from numpy import random, ceil

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

from databasing.database_pull import pull_data_by_lift
from databasing.redis_controls import reset_collar_by_id


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
    parser.add_option('-n', '--num_packets', default=None,
                      help='Specify a number of packets. If not specified, sends 100/sleep_time (2000) packets.')
    parser.add_option('-l', '--lift_id', dest='lift_id', default=None,
                      help='Specific lift_id to use as data source.')
    parser.add_option('-s', '--sleep', dest='sleep_time', default=None,
                      help='Delay between sending packets (in seconds).')
    parser.add_option('-c', '--collar', dest='collar_id', default='555',
                      help='Specify a collar to use, other than 555 (default)')
    parser.add_option('-a', '--athlete', dest='athlete_id', default=None,
                      help='Specify an athlete_id to pass in')
    return parser


def main(args):

    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    host_ip = cli_options.host_ip
    host_port = cli_options.host_port
    mqtt_topic = cli_options.mqtt_topic
    verbose = cli_options.verbose
    N = cli_options.num_packets
    lift_id = cli_options.lift_id
    sleep_time = cli_options.sleep_time
    collar_id = cli_options.collar_id
    athlete_id = cli_options.athlete_id

    if verbose:
        print 'Received args {}'.format(argv)
        print 'Attempting MQTT connection to {i}:{p} on topic {t}'.format(i=host_ip, p=host_port, t=mqtt_topic)

    mqtt_client = establish_client(host_ip, host_port, mqtt_topic)

    #: Set collar_id to default values
    reset_collar_by_id(collar_id)

    #: Delay between packets, in seconds
    if sleep_time is None:
        sleep_time = 0.1

    #: If lift_id is supplied, pull the actual data and send it to MQTT client
    if lift_id is not None:
        if verbose:
            print 'Publishing data from lift_id {}'.format(lift_id)

        header, dat = pull_data_by_lift(lift_id)
        send_head = {"header": {"lift_id": "None",
                                "lift_sampling_rate": header['lift_sampling_rate'],
                                "collar_id": collar_id}
                     }

        if athlete_id is not None:
            send_head['header']['athlete_id'] = athlete_id

        #: Hard code for now. Figure out how to arrange dynamically later
        packet_length = 20.
        #: Number of packets needed to be sent
        N = int(ceil(dat.shape[0]/packet_length))

        for i in range(N):
            #: calculate indices to be used on this loop
            idx = range((20*i),((i+1)*20))
            data = {"content": {"a_x": list(dat.ix[idx]['a_x'].values)}}
            packet = dict(dict(**send_head), **data)

            publish(mqtt_client, mqtt_topic, message=json.dumps(packet))
            sleep(sleep_time)

    #: If not lift_id is supplied, send random data
    else:
        mu = 2
        var = 0.75
        if verbose:
            print 'Publishing data with mean {} and variance {}'.format(mu, var)

        if N is None:
            N = int(100./sleep_time)
        else:
            N = int(N)

        for i in range(N):
            print 'Loop {}'.format(i)
            a_test = [random.normal(mu, var) for _ in range(30)]
            if (i % (10./sleep_time) == 0) & (N > 1):
                header = {"header": {"lift_id": '-1', "lift_sampling_rate": 50, "collar_id": collar_id}}
                data = {"content": {"a_x": [0, 0, 0, 0]}}
            else:
                header = {"header": {"lift_id": "None", "lift_sampling_rate": 50, "collar_id": collar_id}}
                data = {"content": {"a_x": a_test}}

            packet = dict(dict(**header), **data)

            publish(mqtt_client, mqtt_topic, message=json.dumps(packet))
            sleep(sleep_time)

    kill_client(mqtt_client)


# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
