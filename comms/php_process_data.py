import sys
import getopt
import json
from pandas import DataFrame

# try:
#     print 'Adding {} to sys.path'.format(os.path.dirname(os.path.abspath(__file__)))
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# except NameError:
#     print 'working in Dev mode.'

from databasing.redis_controls import establish_redis_client, retrieve_tracker_by_id, update_tracker_by_id
from processing.util import read_header_mqtt, read_content_mqtt, process_data, prep_collar
from databasing.conn_strings import redis_host

redis_client = establish_redis_client(hostname=redis_host, verbose=True)


# This function should get called by calling the file
# If any flags are passed, use getopt to get them
def main(argv):
    print 'received args {}'.format(argv)
    try:
        opts, args = getopt.getopt(argv, 'd:h:', 'data=')
    except getopt.GetoptError:
        print 'php_process_data.py -d (--data) <JSON string>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -d (--data) <JSON string>'
            sys.exit()
        elif opt in ("-d", "--data"):
            data = arg
    print 'received data {}'.format(data)

    if data is None:
        print 'Did not capture JSON data. Will exit..'
        sys.exit(100)

    header = read_header_mqtt(data)

    collar = prep_collar(retrieve_tracker_by_id(redis_client, header['tracker_id']), header, None)

    accel = read_content_mqtt(data, collar)

    if header is None:
        print 'Could not read header data'
        response = raw_input('Assume default values? (fs = 20Hz, weight = 22.5 kg) - [Y]/n')
        if (response is None) or (response in ('y', 'Y')):
            print 'Moving forward with default values.'
            header = {'sampling_rate': 20, 'lift_weight': 22.5, 'weight_units': 'kg'}
        elif response in ('N', 'n'):
            print 'Exiting..'
            sys.exit(20)
        else:
            print 'Unexpected response {}. Exiting..'.format(response)
            sys.exit(20)

    if accel is None:
        print 'Could not read content. Exiting...'
        sys.exit(30)

    a, v, pwr, pos, force = process_data(header, accel, RMS=True)

    data_out = DataFrame(data={'lift_id': [header['lift_id']]*len(a),
                               'timepoint': accel['timepoint'],
                               'a_rms': a,
                               'v_rms': v,
                               'pwr_rms': pwr,
                               'pos_rms': pos,
                               'force_rms': force},
                         index=a.index)

    print 'Processed headers into:\n{}'.format(json.dumps(list(data_out.columns)))
    print 'Processed data into:\n{}'.format(data_out.head().to_json(orient='values'))
    # PHP doesnt capture any return statements. Only what is sent to stdout (e.g. print statements)
    # return data_out.to_json()

# Receives initial ping to file
if __name__ == '__main__':
    main(sys.argv[1:])
