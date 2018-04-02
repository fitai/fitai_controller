import sys
from pandas import DataFrame
from datetime import datetime as dt
from copy import copy

from processing.functions import calc_integral, calc_rms, calc_power, calc_force, calc_pos
from processing.filters import filter_signal
from databasing.redis_controls import get_default_tracker


def read_header_mqtt(data):
    try:
        head = data['header']
    except AttributeError, e:
        print 'No "header" field. Error: {}'.format(e)
        head = None

    return head


def read_content_mqtt(data, tracker_obj):
    try:
        dat = DataFrame(data['content'])
    except AttributeError:
        print 'No "content" field. Returning None'
        return None, None
    else:
        dat = dat.reset_index().rename(columns={'index': 'timepoint'})
        a_cols = [x for x in dat.columns if 'a_' in x]
        g_cols = [x for x in dat.columns if 'g_' in x]
        accel = dat.loc[:, a_cols + ['timepoint', 'millis']]
        gyro = dat.loc[:, g_cols + ['timepoint']]

    try:
        # Scale timepoint values
        accel['timepoint'] = accel['timepoint'].astype(float) / float(tracker_obj['sampling_rate'])
        gyro['timepoint'] = gyro['timepoint'].astype(float) / float(tracker_obj['sampling_rate'])
    except KeyError:
        print 'Couldnt extract sample rate from header. Defaulting to 20 Hz'
        accel['timepoint'] = accel['timepoint'].astype(float) / 20.
        gyro['timepoint'] = gyro['timepoint'].astype(float) / 20.

    try:
        accel['lift_id'] = tracker_obj['lift_id']
    except KeyError:
        print 'Couldnt extract lift_id from header'
        accel['lift_id'] = 0

    return accel, gyro


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
def process_data(tracker_obj, content, inits={}, RMS=False, highpass=True, verbose=False):
    if not isinstance(content, DataFrame):
        if verbose:
            print 'Content (type {}) is not a dataframe. Will try to convert...'.format(type(content))
        content = DataFrame(content)

    content = copy(content)

    #: Establish column headers so that any piece of the function can access
    accel_headers = [x for x in content.columns if x in ['a_x', 'a_y', 'a_z']]
    vel_headers = ['v_' + x.split('_')[-1] for x in accel_headers]
    pwr_headers = ['pwr_' + x.split('_')[-1] for x in accel_headers]
    pos_headers = ['pos_' + x.split('_')[-1] for x in accel_headers]
    force_headers = ['force_' + x.split('_')[-1] for x in accel_headers]

    if highpass:
        for col in accel_headers:
            if col in inits.keys():
                y0 = inits[col]  # runs online HP code; adjusts for initial offsets
            else:
                y0 = {'x': 0, 'y': 0}

            content[col] = filter_signal(content[col].values, y0, 'highpass', freqs=[.6, None], fs=tracker_obj['sampling_rate'], filter_order=1)

    fs = extract_sampling_rate(tracker_obj)
    weight = extract_weight(tracker_obj, verbose)

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
            if 'z' in header and 'v_z' in inits.keys():
                v0 = inits['v_z']
            else:
                v0 = 0.
            vel[vel_headers[i]] = calc_integral(content[header], v0, scale=1., fs=fs)

        pos = DataFrame(columns=pos_headers)
        for i, header in enumerate(accel_headers):
            pos[pos_headers[i]] = calc_pos(content[header], scale=1., fs=fs)

        force = DataFrame(columns=force_headers)
        for i, header in enumerate(accel_headers):
            force[force_headers[i]] = calc_force(content[header], weight)

        pwr = DataFrame(columns=pwr_headers)
        for i in range(len(accel_headers)):
            pwr[pwr_headers[i]] = calc_power(content[accel_headers[i]], vel[vel_headers[i]], weight)

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


def prep_tracker(tracker, head, thresh_dict):

    # Quick check that at least one expected field is in tracker object
    if 'prev_az' not in tracker.keys():
        print 'Redis tracker object {} appears broken. ' \
              'Will replace with default and update as needed.'.format(tracker['tracker_id'])
        tracker_tmp = tracker.copy()
        tracker = get_default_tracker()
        tracker.update(tracker_tmp)

    # The only piece of information from the device not provided by the frontend:
    tracker['sampling_rate'] = head['sampling_rate']

    # TODO: Don't like doing all these checks. Think of a more efficient way...
    # If tracker is newly generated, threshold will be 'None'
    # Don't want to check all 15 possible fields being 'None', so just check a couple
    # if any([(tracker[col] == 'None') for col in ['a_x_thresh', 'v_x_thresh', 'pwr_x_thresh', 'pos_x_thresh']]):
    #     print 'Missing at least one signal threshold. Resetting all...'
    #     try:
    #         # try to extract lift_type
    #         tracker.update(thresh_dict[tracker['lift_type']].copy())
    #     except KeyError:
    #         print 'Couldnt find any thresholds for lift_type {}. Defaulting to 1.'.format(tracker['lift_type'])
    #         for k in thresh_dict[thresh_dict.keys()[0]]:
    #             tracker[k] = 1.

    # start of the lift
    now = dt.now()  # returns local time, i.e. EST
    if tracker['created_at'] == 'None':
        tracker['created_at'] = now

    # added for patrick
    if tracker['updated_at'] == 'None':
        tracker['updated_at'] = now

    #: Should only happen with default tracker initialization
    if tracker['tracker_id'] == 'None':
        tracker['tracker_id'] = head['tracker_id']

    if 'athlete_id' in head.keys():
        tracker['athlete_id'] = head['athlete_id']

    if tracker['init_num_reps'] is None:
        tracker['init_num_reps'] = 0

    #: Left over from old tracker format. Shouldn't need this forever - remove key "threshold" if exists
    tracker.pop('threshold', None)

    # If stored tracker doesn't have "active" field, create it and set to False
    if 'active' not in tracker.keys():
        tracker['active'] = False

    return tracker
