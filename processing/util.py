from pandas import DataFrame
import json
import sys

from processing.functions import calc_vel2, calc_rms, calc_power


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
        accel.timepoint = accel.timepoint.astype(float) / float(collar_obj['lift_sampling_rate'])
    except KeyError:
        print 'Couldnt extract sample rate from header. Defaulting to 20 Hz'
        accel.timepoint = accel.timepoint.astype(float) / 20.

    try:
        accel['lift_id'] = collar_obj['lift_id']
    except KeyError:
        print 'Couldnt extract lift_id from header'
        accel['lift_id'] = 0

    return accel


def read_content_fitai(data, content_key='content'):
    try:
        for key in data['content'].keys():
            data[content_key][key] = [float(x) for x in data[content_key][key][0].split(',')]
        accel = DataFrame(data[content_key], index=data[content_key]['timepoint']).reset_index(drop=True)
    except AttributeError:
        print 'No "content" field. Returning None'
        return None

    return accel


def parse_data(json_string):
    # try:
    data = json.loads(json_string)
    # What should the except statement be??

    header = read_header_mqtt(data)
    content = read_content_fitai(data)

    return header, content


def extract_weight(header):
    try:
        if header['lift_weight_units'] == 'lbs':
            weight = int(header['lift_weight']) * (1./2.5)
            print 'Converted weight from lbs ({w1}) to kg ({w2})'.format(w1=header['lift_weight'], w2=weight)
        elif header['lift_weight_units'] == 'kg':
            weight = int(header['lift_weight'])
        else:
            print 'Unexpected weight unit type {un}. Will leave weight as {w}'.format(un=header['lift_weight_units'], w=header['lift_weight'])
            weight = int(header['lift_weight'])
    except KeyError, e:
        print 'Error finding weight - {}. Will default to 22.5kg'.format(e)
        weight = 22.5

    return weight


def extract_sampling_rate(header):
    # Read in header data
    try:
        fs = float(header['lift_sampling_rate'])
    except KeyError, e:
        print 'sampling_rate KeyError {}. Assuming 20Hz'.format(e)
        fs = 20.

    return fs


# Expects a dataframe with known fields
# Timepoint, a_x, (a_y, a_z), lift_id
def process_data(collar_obj, content):
    if not isinstance(content, DataFrame):
        print 'Content (type {}) is not a dataframe. Will try to convert...'.format(type(content))
        content = DataFrame(content)

    accel_headers = [x for x in content.columns if x in ['a_x', 'a_y', 'a_z']]

    fs = extract_sampling_rate(collar_obj)
    weight = extract_weight(collar_obj)

    if len(accel_headers) == 0:
        print 'Could not find acceleration field(s). Cannot process'
        sys.exit(10)
    elif len(accel_headers) == 1:
        print 'Found single axis of data'

        vel = calc_vel2(content[accel_headers[0]], fs=fs)
        pwr = calc_power(content[accel_headers[0]], vel, weight)

        return content[accel_headers[0]], vel, pwr
    else:
        print 'Found multiple axes of data. Will combine into RMS.'
        a_rms = calc_rms(content, accel_headers)
        v_rms = calc_vel2(a_rms, fs=fs)

        p_rms = calc_power(a_rms, v_rms, weight)

        return a_rms, v_rms, p_rms
