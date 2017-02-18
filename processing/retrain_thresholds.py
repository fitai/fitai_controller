from sys import argv, path as syspath
from os.path import dirname, abspath, join as os_join, exists as os_exists
from os import remove
from optparse import OptionParser
from json import dump

try:
    path = dirname(dirname(abspath(__file__)))
    print 'Adding {} to sys.path'.format(path)
    syspath.append(path)
except NameError:
    syspath.append('/Users/kyle/PycharmProjects/fitai_controller')
    print 'Working in Dev mode.'

from ml_test import find_threshold


def retrain(fname, alpha, verbose):
    if os_exists(fname):
        if verbose:
            print '{} already exists. Removing...'.format(fname)
        remove(fname)

    print 'Retraining threshold dict...'
    thresh = find_threshold(alpha=alpha, smooth=True, plot=False, verbose=verbose)

    if verbose:
        print 'Saving threshold dict to {}...'.format(fname)

    with open(fname, 'w') as outfile:
        dump(thresh, outfile)


def establish_cli_parser():
    parser = OptionParser()
    parser.add_option('-p', '--path', dest='fpath', default='/var/opt/python/fitai_controller/',
                      help='Whether or not to plot learning curves')
    parser.add_option('-s', '--suffix', dest='suffix', default='.txt',
                      help='Append this suffix (dictates filetype)')
    parser.add_option('-f', '--fname', dest='fname', default='thresh_dict',
                      help='Specify name of thresh_dict file, if not "thresh_dict"')
    parser.add_option('-a', '--alpha', dest='alpha', default=0.05,
                      help='Specify a learning rate (alpha)')
    parser.add_option('-v', '--verbose', dest='verbose', default=False, action='store_true',
                      help='Increase console outputs (good for dev purposes)')
    return parser


def main(args):

    cli_parser = establish_cli_parser()

    (cli_options, _) = cli_parser.parse_args(args)

    fpath = cli_options.fpath
    fname = cli_options.fname
    suffix = cli_options.suffix
    alpha = cli_options.alpha
    verbose = cli_options.verbose

    fname = os_join(fpath, fname+suffix)

    retrain(fname, alpha, verbose)

    if verbose:
        print 'Done'

# Receives initial ping to file
if __name__ == '__main__':
    main(argv[1:])
