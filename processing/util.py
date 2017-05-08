import sys
from pandas import DataFrame, Series
from datetime import datetime as dt

from processing.functions import calc_integral, calc_rms, calc_power, calc_force
from processing.filters import filter_signal
from databasing.redis_controls import get_default_collar


def read_header_mqtt(data):
    try:
        head = data['header']
    except AttributeError, e:
        print 'No "header" field. Error: {}'.format(e)
        head = None

    return head


def read_content_mqtt(data, collar_obj):
    try:
        accel = DataFrame(data['content'])
        accel = accel.reset_index().rename(columns={'index': 'timepoint'})
    except AttributeError:
        print 'No "content" field. Returning None'
        return None

    try:
        # Scale timepoint values
        accel.timepoint = accel.timepoint.astype(float) / float(collar_obj['sampling_rate'])
    except KeyError:
        print 'Couldnt extract sample rate from header. Defaulting to 20 Hz'
        accel.timepoint = accel.timepoint.astype(float) / 20.

    try:
        accel['lift_id'] = collar_obj['lift_id']
    except KeyError:
        print 'Couldnt extract lift_id from header'
        accel['lift_id'] = 0

    return accel


# def read_content_fitai(data, content_key='content'):
#     try:
#         for key in data['content'].keys():
#             data[content_key][key] = [float(x) for x in data[content_key][key][0].split(',')]
#         accel = DataFrame(data[content_key], index=data[content_key]['timepoint']).reset_index(drop=True)
#     except AttributeError:
#         print 'No "content" field. Returning None'
#         return None
#
#     return accel


# def parse_data(json_string):
#     # try:
#     data = json.loads(json_string)
#     # What should the except statement be??
#
#     header = read_header_mqtt(data)
#     content = read_content_fitai(data)
#
#     return header, content


def extract_weight(header, verbose):
    try:
        if header['weight_units'] == 'lbs':
            weight = int(header['lift_weight']) * (1./2.5)
            if verbose:
                print 'Converted weight from lbs ({w1}) to kg ({w2})'.format(w1=header['lift_weight'], w2=weight)
        elif header['weight_units'] == 'kg':
            weight = int(header['lift_weight'])
        else:
            if verbose:
                print 'Unexpected weight unit type {un}. Will leave weight as {w}'.format(un=header['weight_units'], w=header['lift_weight'])
            weight = int(header['lift_weight'])
    except KeyError, e:
        print 'Error finding weight - {}. Will default to 22.5kg'.format(e)
        weight = 22.5

    return weight


def extract_sampling_rate(header):
    # Read in header data
    try:
        fs = float(header['sampling_rate'])
    except KeyError, e:
        print 'sampling_rate KeyError {}. Assuming 20Hz'.format(e)
        fs = 20.

    return fs


# Expects a dataframe with known fields
# Timepoint, a_x, (a_y, a_z), lift_id
#: NOTE: default is to process accel & vel into RMS signals
def process_data(collar_obj, content, RMS=False, highpass=True, verbose=False):
    if not isinstance(content, DataFrame):
        if verbose:
            print 'Content (type {}) is not a dataframe. Will try to convert...'.format(type(content))
        content = DataFrame(content)

    # Drop columns that contain more than 10% missing values
    x = content.isnull().sum(axis=0) > 0.1*content.shape[0]
    content = content.loc[:, x.loc[~x].index]

    #: Establish column headers so that any piece of the function can access
    accel_headers = [x for x in content.columns if x in ['a_x', 'a_y', 'a_z']]
    vel_headers = ['v_' + x.split('_')[-1] for x in accel_headers]
    pwr_headers = ['pwr_' + x.split('_')[-1] for x in accel_headers]
    pos_headers = ['pos_' + x.split('_')[-1] for x in accel_headers]
    force_headers = ['force_' + x.split('_')[-1] for x in accel_headers]

    # Try to impose high pass on acceleration - see if it will fix the velocity drift
    if highpass:
        for col in accel_headers:
            content[col] = filter_signal(content[col], filter_type='highpass', f_low=0.1, fs=collar_obj['sampling_rate'])

    fs = extract_sampling_rate(collar_obj)
    weight = extract_weight(collar_obj, verbose)

    if len(accel_headers) == 0:
        if verbose:
            print 'Could not find acceleration field(s). Cannot process'
        sys.exit(10)

    else:
        # Can't calculate integral on rectified signal - will result in a positively drifting signal
        # Have to leave acceleration split into constituent dimensions, calculate velocity along each,
        # then combine into RMS signal
        vel = DataFrame(columns=vel_headers)
        for i, header in enumerate(accel_headers):
            vel[vel_headers[i]] = calc_integral(content[header], scale=1., fs=fs)

        pwr = DataFrame(columns=pwr_headers)
        for i in range(len(accel_headers)):
            pwr[pwr_headers[i]] = calc_power(content[accel_headers[i]], vel[vel_headers[i]], weight)

        pos = DataFrame(columns=pos_headers)
        for i, header in enumerate(accel_headers):
            pos[pos_headers[i]] = calc_integral(content[header], scale=1., fs=fs)

        force = DataFrame(columns=force_headers)
        for i, header in enumerate(accel_headers):
            force[force_headers[i]] = calc_force(content[header], weight)

    if RMS:
        a = calc_rms(content, accel_headers).to_frame()
        v = calc_rms(vel, vel_headers).to_frame()
        pwr = calc_rms(pwr, pwr_headers).to_frame()
        pos = calc_rms(pos, pos_headers).to_frame()
        force = calc_rms(force, force_headers).to_frame()

    else:
        a = content
        v = vel

    return a, v, pwr, pos, force


def prep_collar(collar, head, thresh_dict):
    # Quick check that at least one expected field is in collar object
    if 'pwr_thresh' not in collar.keys():
        print 'Redis collar object {} appears broken. ' \
              'Will replace with default and update as needed.'.format(collar['collar_id'])
        collar_tmp = collar.copy()
        collar = get_default_collar()
        collar.update(collar_tmp)

    # The only piece of information from the device not provided by the frontend:
    collar['sampling_rate'] = head['sampling_rate']

    # TODO: Don't like doing all these checks. Think of a more efficient way...
    # If collar is newly generated, threshold will be 'None'
    if any([(collar[col] == 'None') for col in ['a_thresh', 'v_thresh', 'pwr_thresh', 'pos_thresh']]):
        print 'Missing at least one signal threshold. Resetting all...'
        try:
            # try to extract lift_type
            lift_thresh = thresh_dict[collar['lift_type']]
            collar['a_thresh'] = lift_thresh['a_thresh']
            collar['v_thresh'] = lift_thresh['v_thresh']
            collar['pwr_thresh'] = lift_thresh['pwr_thresh']
            collar['pos_thresh'] = lift_thresh['pos_thresh']
        except KeyError:
            print 'Couldnt find any thresholds for lift_type {}. Defaulting to 1.'.format(collar['lift_type'])
            collar['a_thresh'], collar['v_thresh'], collar['pwr_thresh'], collar['pos_thresh'] = 1., 1., 1., 1.

    if collar['lift_start'] == 'None':
        collar['lift_start'] = dt.now()

    #: Should only happen with default collar initialization
    if collar['collar_id'] == 'None':
        collar['collar_id'] = head['collar_id']

    if 'athlete_id' in head.keys():
        collar['athlete_id'] = head['athlete_id']

    #: Left over from old collar format. Shouldn't need this forever - remove key "threshold" if exists
    collar.pop('threshold', None)

    # If stored collar doesn't have "active" field, create it and set to False
    if 'active' not in collar.keys():
        collar['active'] = False

    return collar
