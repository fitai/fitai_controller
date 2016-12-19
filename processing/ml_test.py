from os.path import join

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas import DataFrame

from processing.functions import method1, calc_integral
from FitAI.load_data import load_file
from FitAI.predictors import make_predictions
from processing.filters import simple_highpass
from processing.util import process_data, extract_weight, extract_sampling_rate
from databasing.database_pull import pull_data_by_lift, pull_lift_ids


bin_size = 1.  # Number of samples to collect before processing the signal (float)


def find_threshold(smooth=False, plot=False, verbose=False):

    # Each lift_type will have multiple lift_ids associated with it
    # Learn on each lift_type separately, and average resultant thresholds FOR NOW! (12/9/16)
    lift_ids = pull_lift_ids().set_index('lift_type')

    train_dict = {}

    if lift_ids is not None:
        for lift_type in list(lift_ids.index):
            ids = lift_ids.ix[lift_type]['lift_ids']
            type_dict = {}
            for i, lift_id in enumerate(ids):
                type_dict[i] = learn_on_lift_id(int(lift_id), smooth, plot, verbose)

            errs = [d['error'] for _, d in type_dict.iteritems()]
            thresholds = [d['thresh'] for _, d in type_dict.iteritems()]

            if any([e > 0 for e in errs]):
                max_err = max(errs)
                # 1. - abs(err)/max_err
                # will eliminate observation with max error, and will vary directly with distance from max_err
                w = [1. - np.abs(err)/max_err for err in errs]
                thresh = np.sum(w*thresholds)
            else:
                # No info on error, or all errors are the same - weight equally
                thresh = np.sum(thresholds)/len(thresholds)

            train_dict.update({lift_type: thresh})
    else:
        print 'Unable to retrieve lift_ids from database. Cannot learn.'
        return None

    if verbose:
        print 'Found thresholds for the following lift_types:\n{}'.format(train_dict.keys())

    return train_dict


def learn_on_lift_id(lift_id, smooth, plot, verbose):
    header, data = pull_data_by_lift(lift_id)
    header = header.ix[0].to_dict()
    a, v, p = process_data(header, data, verbose)

    data = DataFrame(data={'a_rms': a,
                           'v_rms': v,
                           'p_rms': p},
                     index=a.index)

    if smooth:
        # Needed to scale signal and run calcs
        fs = extract_sampling_rate(header)
        weight = extract_weight(header, verbose)

        # re-calculate everything from smoothed acceleration signal
        rms_raw = data['a_rms']
        a_rms = list()
        #: Implement the simple high pass filter
        for i in range(len(rms_raw)):
            if i > 0:
                rms_raw[i] = 0.66*rms_raw[i] + 0.33*rms_raw[i-1]
                a = simple_highpass(rms_raw[i], rms_raw[i-1], a_rms[-1], fc=1., fs=fs)
            else:
                a = 0
            a_rms.append(a)

        a_rms = np.array(a_rms)
        v_euler = calc_integral(a_rms, scale=bin_size, fs=fs)
        pwr_rms = a_rms * weight * v_euler
    else:
        # Don't smooth; use calculated power
        pwr_rms = data['p_rms']

    try:
        # true_reps = int([x for x in filename.split('_') if 'rep' in x][0].split('rep')[0])
        true_reps = header['lift_num_reps']
    except IndexError:
        print 'Couldnt find number of reps in header from lift_id {}.\n Cannot learn from this file'
        print 'Defaulting to threshold of 1'
        return {'thresh': 1., 'error': None}

    #: For use in learning - number of signal points to exclude from calculation of standard deviation
    offset = 0
    scale = 0.

    pwr_imp, pwr_thresh, pwr_reps = method1(pwr_rms, calc_offset=offset, scale=scale)

    err = pwr_reps - true_reps
    err_tracking = [err]
    cnt = 0
    scales = [scale]
    while np.abs(err) > 0:
        #: Increasing scale increases the value of the threshold, thereby making the classifier LESS sensitive
        #: Decreasing scale makes classifier MORE sensitive
        if (err > 0) | (pwr_thresh == 0):
            scale += 0.1
        elif err < 0:
            scale *= 0.9
        else:
            print 'Shouldnt reach this'

        pwr_imp, pwr_thresh, pwr_reps = method1(pwr_rms, calc_offset=offset, scale=scale)

        err = pwr_reps - true_reps
        err_tracking.append(err)
        scales.append(scale)
        if verbose:
            print 'iteration {c}: error of {e} reps'.format(c=cnt, e=err)
        cnt += 1

        #: Force break if err can't get to 0
        if cnt > 100:
            print 'Couldnt converge. Breaking..'
            break

    # if plot:
    #     #: Plot learning curves
    #     plot_learning(err_tracking, scales)
    #     plot_cutoffs(pwr_rms, pwr_imp, pwr_thresh)

    #: Attempt at basic learning
    if verbose:
        print 'Final power cutoff: {}'.format(pwr_thresh)

    return {'thresh': pwr_thresh, 'error': err}


# def try_classifier():
#     data_folder = '/Users/kyle/PycharmProjects/FitAI/data_files/'
#     person_folder = 'Kyle'
#
#     ohp_file1 = 'OHP_45lb_10rep_1.csv'  # 0 - 850
#     ohp_file2 = 'OHP_45lb_10rep_2.csv'  # 0 - 950
#     ohp_file3 = 'OHP_45lb_10rep_3.csv'  # 0 - 875
#
#     dl_file1 = 'DL_45lb_10rep_1.csv'  # 0 - 1700
#     dl_file2 = 'DL_45lb_10rep_2.csv'  # 0 - 1220
#     dl_file3 = 'DL_45lb_10rep_3.csv'  # 0 - 1400
#
#     # sq_file1 = 'Kyle_SQ_45lb_10rep_1.csv'  # NOTE: workout 1 ends @ 980, workout 2 ends @ 2200
#     # sq_file2 = 'Kyle_SQ_45lb_10rep_2.csv'
#
#     filenames = [ohp_file1, ohp_file2, ohp_file3, dl_file1, dl_file2, dl_file3]
#
#     ## Split data into training and testing ##
#     train_storage = pd.DataFrame()
#     test_storage = pd.DataFrame()
#     train_labels = []
#     test_labels =[]
#     for filename in filenames:
#         col_names = ['timestamp', 'x', 'y', 'z']
#         data = load_file(join(data_folder, person_folder, filename), names=col_names, skiprows=1).iloc[0:850]
#         # Extract data from file contents
#         dat = data[['x', 'y', 'z']]
#         rms_raw = [np.sqrt(np.mean(dat.iloc[i].x**2 + dat.iloc[i].y**2 + dat.iloc[i].z**2)) for i, _ in enumerate(tuple(dat.x))]
#         a_rms = list()
#         #: Implement the simple high pass filter
#         for i in range(len(rms_raw)):
#             if i > 0:
#                 a = simple_highpass(rms_raw[i], rms_raw[i-1], a_rms[-1], fc=1., fs=fs)
#             else:
#                 a = 0
#             a_rms.append(a)
#
#         a_rms = np.array(a_rms)
#         v_euler = calc_integral(a_rms, scale=bin_size, fs=fs)
#         pwr_rms = a_rms * weight * v_euler
#
#         tmp_df = pd.DataFrame(data={'accel': a_rms})
#
#         label = filename.split('_')[1]
#         filenum = int(filename.split('_')[-1].split('.')[0])
#
#         # First 2 of 3 files are for training
#         if filenum < 3:
#             train_labels.append(label)
#             train_storage = pd.concat([train_storage, tmp_df], axis=1)
#         # File 3 of 3 is for testing
#         else:
#             test_labels.append(label)
#             test_storage = pd.concat([test_storage, tmp_df], axis=1)
#
#     ## Weak pre-processing
#     train_storage = train_storage.fillna(train_storage.mean())
#     test_storage = test_storage.fillna(test_storage.mean())
#
#     forest_pred = make_predictions(train=train_storage, train_labels=train_labels, test=test_storage, predictor='forest')
#     percent_correct = float(np.sum((test_labels == forest_pred)*1))/float(len(test_labels))
#     print 'Random forest percent correct classification: {}'.format(percent_correct*100.)
#
#     # pred = nn.predict(np.array(test_storage.T.fillna(test_storage.T.mean())))
#     # score = nn.score(np.array(test_storage.T.fillna(test_storage.T.mean())), np.array(test_labels))
#
#     nn_pred = make_predictions(train=train_storage, train_labels=train_labels, test=test_storage, predictor='neural_net')
#     percent_correct = float(np.sum((test_labels == nn_pred)*1))/float(len(test_labels))
#     print 'Neural network percent correct classification: {}'.format(percent_correct*100.)


# From an input power vector, detect any change in state and increment
def calc_reps(pwr, n_reps, state, thresh=0.):
    """
    Simple - any crossing of the power threshold indicates a change in state. From this, determine where the
    user was (in the state-space), and adjust accordingly

    :param pwr: list-like power vector. will be converted to pandas Series
    :param n_reps: number of reps user is at before processing this power vector
    :param thresh: threshold to apply to power vector
    :param state: state of user coming into processing step
    :return:
    """

    if not isinstance(pwr, pd.Series):
        print 'converting power {} to pandas Series...'.format(type(pwr))
        pwr = pd.Series(pwr)

    # Want to keep track of timepoints too
    diff_signal = ((pwr > thresh) * 1).diff()[1:].abs().diff()[1:]

    # Drops two points off front, but forces signal to stay above/below threshold for at least
    # 1 point to be considered a change in state - better than considering any noise around threshold
    # as a change in state. Compare this to how it was before:
    # N = float( ((pwr > thresh) * 1).diff()[1:].abs().sum() )
    N = float( diff_signal.sum() )

    # np.where() is resource intensive - just map for now
    # Want to identify any shift in the state of the user
    if state == 'rest':
        shift = 0
    elif state == 'lift':
        shift = 1
        # Here the athlete was mid-lift, so any deltas seen should be offset by 1 to acknowledge the athlete
        # started from 'lift' position
        N += 1
    else:
        print 'Unsure of lift state {}. Will assume "rest"'.format(state)

    # N = number of threshold crossings (absolute value)
    # So, if user is at rest, it will take 2 crossings to count as a full rep
    # If user is mid-rep, then it will only take 1 crossing (the downswing) to complete a rep
    n_reps += np.floor( (shift+N)/2.)
    print 'User at {} reps'.format(n_reps)

    # Update the state
    # Based on number of state changes, find new state of object
    # new_state = ['rest', 'lift'][int((shift+(N % 2)) % 2)]

    # Find AND TRACK all crossings IF THERE ARE ANY
    # Assign rep action based on direction of diff (+ = start, - = stop)
    crossings = diff_signal[diff_signal != 0].to_frame('diff')

    if crossings.shape[0] > 0:
        crossings['action'] = 'start_rep'
        crossings.loc[crossings['diff'] < 0, 'action'] = 'stop_rep'

        # Update state of object
        if diff_signal[crossings.index[-1]] > 0:
            new_state = 'lift'
        else:
            new_state = 'rest'

    # If there aren't any crossings, then just return whatever came in
    else:
        new_state = state
        crossings = None

    return n_reps, new_state, crossings


# def plot_learning(err_tracking, scales):
#     f, axarr = plt.subplots(2, sharex=True)
#     axarr[0].plot(err_tracking)
#     axarr[0].set_title('Error curve (Final Rep Count: {})'.format(pwr_reps))
#     axarr[0].set_xlabel('Iteration')
#     axarr[0].set_ylabel('Estimated - Actual')
#     axarr[1].plot(scales)
#     axarr[1].set_title('Sigma Multiplier (Final Value: {})'.format(scale))
#     axarr[1].set_xlabel('Iteration')
#     axarr[1].set_ylabel('Scale Value')
#
#
# def plot_cutoffs(pwr_rms, pwr_imp, pwr_thresh):
#     plt.figure(10)
#     p_color = 'purple'
#     plt.plot(pwr_rms, color=p_color)
#     plt.axhline(y=pwr_thresh, color=p_color, linestyle='--')
#     for x in np.where(pwr_imp > 0)[0]:
#         plt.axvline(x=x, color=p_color, linestyle='-.')
