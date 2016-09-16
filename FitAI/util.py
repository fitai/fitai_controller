import json

from pandas import DataFrame


def read_header(data, header_key='header'):
    try:
        head = dict(data[header_key])
    except AttributeError, e:
        print 'No "header" field. Returning None'
        return None

    return head


def read_content(data, content_key='content'):
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

    header = read_header(data)
    content = read_content(data)

    return header, content


def extract_weight(header):
    try:
        if header['lift_weight_units'] == 'lbs':
            weight = header['lift_weight'] * (1./2.5)
            print 'Converted weight from lbs ({w1}) to kg ({w2})'.format(w1=header['lift_weight'], w2=weight)
        elif header['lift_weight_units'] == 'kg':
            weight = header['lift_weight']
        else:
            print 'Unexpected weight unit type {un}. Will leave weight as {w}'.format(un=header['lift_weight_units'], w=header['lift_weight'])
            weight = header['lift_weight']
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
