import paho.mqtt.client as mqtt

from sys import argv, path as syspath, exit
from os.path import dirname, abspath
from optparse import OptionParser
from json import loads
from pandas import DataFrame, merge
from threading import Thread


try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/projects/fitai_controller')
    print 'Working in Dev mode.'

from databasing.database_push import push_to_db
from databasing.redis_controls import establish_redis_client, retrieve_tracker_by_id
from processing.util import read_header_mqtt, read_content_mqtt, process_data, prep_tracker
from ml.rep_learn import run_detector
from databasing.conn_strings import redis_host
from comms.redis_pubsub import redis_pub

# TODO: Turn this entire file into a class. Will allow us to use objects like the redis_client
# TODO: Push thresh_dict load into separate file
# as class attributes instead of forcing us to keep them global
#: Alpha = learning rate - make smaller to learn slower and take more iterations, make larger to learn faster and
#: risk non-convergence

# NOTE TO SELF: NEED A BETTER WAY TO MAKE THIS GLOBAL
# should probably turn the entire script into an object....
# Attempt to connect to redis server
redis_client = establish_redis_client(hostname=redis_host, verbose=True)
# redis_client = establish_redis_client(hostname='18.221.103.145', verbose=True)  # in case conn to server is needed

# If connection fails, MQTT client will not be able to update tracker object, and will be useless. Kill and try again
if redis_client is None:
    print 'Couldnt connect to redis. Killing MQTT client.'
    exit(100)


def get_default_detector():
    detector_input = {
        'vel': None
        , 'sampling_rate': None
        , 'prev_dat': []
        , 'prev_vz': 0.
        , 'prev_filt_vz': 0.
        , 'packet_size': None
        , 't_prev': 0.
        , 'hold': False
        , 'cross_track': []
        , 'ts': []
        , 'var_max': 0.05
        , 'min_irt_samples': 10
        , 'min_intra_samples': 60
        , 'starts': []
        , 'stops': []
        , 'n_reps': 0.
        , 't_min': 0.
        # extra, just for plotting purposes:
        , 'sig_track': None
        , 'p_track': None
    }

    return detector_input


# The callback for when the client successfully connects to the broker
def mqtt_on_connect(client, userdata, rc):
    """
    We subscribe on_connect() so that if we lose the connection and reconnect, subscriptions will be renewed.
    """

    # client.subscribe('fitai')
    print 'connected'


def update_tracker_obj(client, tracker_id):
    print('Updating tracker {}'.format(tracker_id))
    client.trackers.update({tracker_id: retrieve_tracker_by_id(redis_client, tracker_id)})
    return client


#: The callback for when a PUBLISH message is received from the broker
#: There will need to be a lot of logic wrappers here; a lot could go wrong, and it should all be handled
#: as gracefully as possible
def mqtt_on_message(client, userdata, msg):

    # topic = msg.topic
    # print 'Received message from topic "{}"'.format(topic)

    try:
        data = loads(msg.payload)
        header = read_header_mqtt(data)

        tracker_id = str(header['tracker_id'])
        # if client doesn't contain tracker object for this tracker_id, create a holder for it
        if tracker_id not in client.trackers.keys():
            client.trackers.update({tracker_id: None})

        if tracker_id not in client.detectors.keys():
            client.detectors.update({tracker_id: None})

        tracker_stat = tracker_id + '_status'
        # Check status of this object (separate stored item in redis). status == 'stale', means
        #   that the object has been updated elsewhere and needs to be refreshed
        if (client.trackers[tracker_id] is None) or (redis_client.get(tracker_stat) == 'stale'):
            print 'refreshing tracker'
            client = update_tracker_obj(client, tracker_id)
            redis_client.set(tracker_stat, 'fresh')
            client.trackers[tracker_id]['push_header'] = True  # on first tracker update, push header to db
            client.detectors[tracker_id] = get_default_detector()

        tracker = prep_tracker(client.trackers[tracker_id], header, client.thresh_dict)
        detector = client.detectors[tracker_id]

        accel, gyro = read_content_mqtt(data, tracker)

        # NOTE: process_data() returns accel, vel, power, position. All those returns are useful for
        #       calc_reps(), but are re-calculated differently before being pushed to redis pubsub,
        #       so I decided just to pass them straight through to the calc_reps function
        filt_inits = {'a_z': {'x': tracker['prev_az'], 'y': tracker['prev_filt_az']}, 'v_z': detector['prev_vz']}
        acc, vel, _, _, _ = process_data(tracker, accel, filt_inits, RMS=False, highpass=True)

        detector.update({'vel': vel['v_z'], 'sampling_rate': header['sampling_rate'], 'packet_size': header['sampling_rate']})

        sig, prev_dat, hold, cross_track, ts, n_reps, _, _ = run_detector(**detector)

        # store outputs from detector for next loop
        detector.update({'prev_dat': prev_dat, 'hold': hold, 'cross_track': cross_track, 'ts': ts, 'n_reps': n_reps,
                         'sig_track': None, 'p_track': None})
        tracker['calc_reps'] = n_reps

        # filter initial conditions for next loop
        detector['prev_vz'] = vel['v_z'].iloc[-1]
        detector['prev_filt_vz'] = sig.iloc[-1]
        tracker['prev_az'] = accel['a_z'].iloc[-1]
        tracker['prev_filt_az'] = acc['a_z'].iloc[-1]

        redis_pub(redis_client, 'lifts', tracker, process_data(tracker, accel, filt_inits, RMS=True, highpass=True), 'real_time')

        client.trackers[tracker_id] = tracker  # update stored tracker object

        if tracker['active']:
            if 'lift_start' in tracker.keys():
                tracker.pop('lift_start')
            header = DataFrame(data=tracker, index=[0]).drop(
                ['active', 'curr_state', 'max_t', 'prev_az', 'prev_filt_az'],
                axis=1)
            content = merge(accel, gyro, on='timepoint', how='left').fillna(0.)

            # create new thread for the db push
            db_thread = Thread(target=push_to_db, args=(header, content))
            db_thread.start()

            if client.trackers[tracker_id]['push_header']:  # assume push was done
                client.trackers[tracker_id]['push_header'] = False  # skip the header push to db on all subsequent loops
        else:
            print 'Processed data for tracker {}, but tracker is not active...'.format(tracker['tracker_id'])

    except KeyError, e:
        print 'Key not found in data header. Cannot retrieve relevant information and update redis object.'
        print 'Error message: \n{}'.format(e)
    except ValueError, e:
        print 'Error processing JSON object. Message: \n{}'.format(str(e))
        print 'received: {}'.format(str(msg.payload))
    except TypeError, e:
        print 'received: {}'.format(msg.payload)
        print 'Error processing string input. Message: \n{}'.format(str(e))


def establish_mqtt_client(ip, port, topic):
    client = mqtt.Client()
    # custom additions
    # client.thresh_dict = load_thresh_dict(fname='thresh_dict.txt')
    client.thresh_dict = {}
    client.trackers = {}
    client.detectors = {}

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
    parser.add_option('-i', '--ip', dest='host_ip', default='localhost',  # 18.221.103.145 fitai-dev
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

# Sample data packet from device
#
# data = {"header": {"tracker_id": 555,"lift_id": "None","sampling_rate":50},"content":{
# "a_x": [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
# "a_y":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
# "a_z":[0.10,0.20,0.30,0.20,0.10,0.00,-0.10,-0.20,-0.30,-0.20,-0.10,0.00,0.10,-0.10,0.00],
# "g_x":[-1.75,-1.83,-1.87,-1.85,-1.87,-1.97,-1.83,-1.82,-1.72,-1.79,-1.81,-1.84,-1.85,-1.82,-1.91],
# "g_y":[1.49,1.46,1.46,1.54,1.53,1.57,1.73,1.63,1.60,1.60,1.64,1.63,1.59,1.56,1.61],
# "g_z":[0.77,0.74,0.76,0.84,0.89,0.88,0.85,0.87,0.80,0.82,0.85,0.82,0.83,0.85,0.85],
# "millis":[8808,8813,8818,8823,8828,8833,8844,8849,8854,8859,8864,8869,8874,8880,8885]}}

# dat = '{"header": {"tracker_id": 556,"lift_id": "None" ,"sampling_rate":50},"content":{"a_x": [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],"a_y":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],"a_z":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],"g_x":[-2.17,-2.17,-2.09,-2.02,-1.88,-1.59,-2.09,-2.16,-2.04,-1.87,-1.75,-1.69,-1.87,-1.90,-1.90],"g_y":[1.28,1.35,1.40,1.44,1.57,1.75,1.53,1.42,1.50,1.50,1.59,1.65,1.47,1.48,1.48],"g_z":[0.75,0.74,0.80,0.72,0.67,0.73,0.80,0.72,0.66,0.70,0.72,0.77,0.69,0.72,0.72],"millis":[134164,134169,134175,134180,134185,134190,134202,134207,134212,134217,134222,134227,134232,134237,134242]}}'