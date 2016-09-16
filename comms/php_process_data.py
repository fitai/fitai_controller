import sys, os
import getopt
import json
from pandas import DataFrame

try:
    print 'Adding {} to sys.path'.format(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    print 'working in Dev mode.'

from mqtt.functions import calc_vel2, calc_rms, calc_power
from mqtt.util import parse_data, extract_weight, extract_sampling_rate


# Expects a dataframe with known fields
# Timepoint, a_x, (a_y, a_z), lift_id
def process_data(header, content):
    if not isinstance(content, DataFrame):
        print 'Content (type {}) is not a dataframe. Will try to convert...'.format(type(content))
        content = DataFrame(content)

    accel_headers = [x for x in content.columns if x in ['a_x', 'a_y', 'a_z']]

    fs = extract_sampling_rate(header)
    weight = extract_weight(header)

    if len(accel_headers) == 0:
        print 'Could not find acceleration field(s). Cannot process'
        sys.exit(10)
    elif len(accel_headers) == 1:
        print 'Found single axis of data'

        vel = calc_vel2(content[accel_headers[0]], fs)
        pwr = calc_power(content[accel_headers[0]], vel, weight)

        return content[accel_headers[0]], vel, pwr
    else:
        print 'Found multiple axes of data. Will combine into RMS.'
        a_rms = calc_rms(content, accel_headers)
        v_rms = calc_vel2(a_rms, fs)

        p_rms = calc_power(a_rms, v_rms, weight)

        return a_rms, v_rms, p_rms


# This function should get called by calling the file
# If any flags are passed, use getopt to get them
def main(argv):
    print 'received args {}'.format(argv)
    try:
        opts, args = getopt.getopt(argv, 'd:h:', 'data=')
    except getopt.GetoptError:
        print 'test.py -d (--data) <JSON string>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -d (--data) <JSON string>'
            sys.exit()
        elif opt in ("-d", "--data"):
            json_data = arg
    print 'received data {}'.format(json_data)

    if json_data is None:
        print 'Did not capture JSON data. Will exit..'
        sys.exit(100)

    header, content = parse_data(json_data)

    if header is None:
        print 'Could not read header data'
        response = raw_input('Assume default values? (fs = 20Hz, weight = 22.5 kg) - [Y]/n')
        if (response is None) or (response in ('y', 'Y')):
            print 'Moving forward with default values.'
            header = {'lift_sampling_rate': 20, 'lift_weight': 22.5, 'lift_weight_units': 'kg'}
        elif response in ('N', 'n'):
            print 'Exiting..'
            sys.exit(20)
        else:
            print 'Unexpected response {}. Exiting..'.format(response)
            sys.exit(20)

    if content is None:
        print 'Could not read content. Exiting...'
        sys.exit(30)

    a, v, p = process_data(header, content)

    data_out = DataFrame(data={'lift_id': [header['lift_id']]*len(a),
                               'timepoint': content['timepoint'],
                               'a_rms': a,
                               'v_rms': v,
                               'p_rms': p},
                         index=a.index)

    print 'Processed headers into:\n{}'.format(json.dumps(list(data_out.columns)))
    print 'Processed data into:\n{}'.format(data_out.head().to_json(orient='values'))
    # PHP doesnt capture any return statements. Only what is sent to stdout (e.g. print statements)
    # return data_out.to_json()

# Receives initial ping to file
if __name__ == '__main__':
    main(sys.argv[1:])

# json_data = '{"header":{"lift_id":0,"lift_sampling_rate":50,"lift_weight":100,"lift_weight_units":"lbs"},"content":{"timepoint":["0,0.02,0.04"],"a_x":["230,773,169"]}}'
