#!/usr/bin/python

import sys, getopt
import json

from pandas import DataFrame


def parse_data(json_string):
    # try:
    data = json.loads(json_string)
    # What should the except statement be??

    return data


# Expects a dataframe with known fields
# Timepoint, a_x, (a_y, a_z), lift_id
def process_data(data):
    if not isinstance(data, DataFrame):
        print 'Data (type {}) is not a dataframe. Will try to convert...'.format(type(data))
        data = DataFrame(data)

    accel_headers = [x for x in data.columns if x in ['a_x', 'a_y', 'a_z']]

    if not (len(accel_headers) > 0):
        print 'Could not find acceleration field(s). Cannot process'
        sys.exit(10)


# This function should get called by calling the file
# If any flags are passed, use getopt to get them
def main(argv):
    print 'received args {}'.format(argv)
    try:
      opts, args = getopt.getopt(argv,'d:h:', 'data=')
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
    # data = parse_data(data)


# Receives initial ping to file
if __name__ == '__main__':
    main(sys.argv[1:])
