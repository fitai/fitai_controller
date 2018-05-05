# import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import array, isnan, abs, floor, sum as np_sum
from itertools import product
from json import loads, dump

from processing.functions import method1, calc_integral, calc_pos, calc_force
from processing.filters import simple_highpass
from processing.util import process_data, extract_weight, extract_sampling_rate
from databasing.database_pull import pull_data_by_lift, pull_lift_ids


bin_size = 1.  # Number of samples to collect before processing the signal (float)

ALL_SIGS = [x + y for (x, y) in product(['a', 'v', 'pwr', 'pos', 'force'], ['_x', '_y', '_z'])]
ALL_THRESH = [x+'_thresh' for x in ALL_SIGS]


#: Returns dict of lift_type: adjusted_thresholds
#: Where adjusted_thresholds is a dict of a/v/p_thresh: threshold_value
#: So, per lift type, this function should return a dict of 3 threshold values, one for each signal type
def find_threshold(alpha=0.1, smooth=False, plot=False, verbose=False):

    # Each lift_type will have multiple lift_ids associated with it
    # Learn on each lift_type separately, and average resultant thresholds FOR NOW! (12/9/16)
    lift_ids = pull_lift_ids().set_index('lift_type')

    train_dict = {}
    adj_thresh = {}
    if lift_ids is not None:
        for lift_type in list(lift_ids.index):
            ids = lift_ids.ix[lift_type]['lift_ids']
            type_dict = {}
            for i, lift_id in enumerate(ids):
                type_dict[i] = learn_on_lift_id(int(lift_id), smooth, alpha, plot, verbose)

            for label in ALL_SIGS:

                errs = [d[label+'_error'] for _, d in type_dict.iteritems()]
                thresholds = [d[label+'_thresh'] if not isnan(d[label+'_thresh']) else 0. for _, d in type_dict.iteritems()]

                if any([e > 0 for e in errs]):
                    max_err = max(abs(errs))
                    # 1. - abs(err)/max_err
                    # will eliminate observation with max error, and will vary directly with distance from max_err
                    # divide by length of weights so that the weights add to one
                    w = [(1. - abs(err)/abs(max_err))/float(len(errs)-1) for err in errs]
                    thresh = np_sum(a*b for a, b in zip(w, thresholds))
                else:
                    # No info on error, or all errors are the same - weight equally
                    thresh = np_sum(thresholds)/len(thresholds)

                adj_thresh.update({label+'_thresh': thresh})

            train_dict.update({lift_type: adj_thresh})
    else:
        print 'Unable to retrieve lift_ids from database. Cannot learn.'
        return None

    if verbose:
        print 'Found thresholds for the following lift_types:\n{}'.format(train_dict.keys())

    return train_dict


def learn_on_lift_id(lift_id, smooth, alpha, plot, verbose):
    header, dat = pull_data_by_lift(lift_id)
    a, v, pwr, pos, force = process_data(header, dat, RMS=False, highpass=True, verbose=verbose)

    data = a.join(v).join(pwr).join(pos).join(force)

    #: Establish column headers so that any piece of the function can access
    accel_headers = [x for x in data.columns if x in ['a_x', 'a_y', 'a_z']]
    vel_headers = ['v_' + x.split('_')[-1] for x in accel_headers]
    pwr_headers = ['pwr_' + x.split('_')[-1] for x in accel_headers]
    pos_headers = ['pos_' + x.split('_')[-1] for x in accel_headers]
    force_headers = ['force_' + x.split('_')[-1] for x in accel_headers]

    if smooth:
        # Needed to scale signal and run calcs
        fs = extract_sampling_rate(header)
        weight = extract_weight(header, verbose)

        for acc_dim in [x for x in data.columns if 'a_' in x]:
            dim = x.split('_')[1]
            # re-calculate everything from smoothed acceleration signal
            signal = data[acc_dim]
            acc_rms = list()
            #: Implement the simple high pass filter
            for i in range(len(signal)):
                if i > 0:
                    signal[i] = 0.66*signal[i] + 0.33*signal[i-1]
                    a = simple_highpass(signal[i], signal[i-1], acc_rms[-1], fc=1., fs=fs)
                else:
                    a = 0.
                #: After calculating filtered RMS value, push into acc_rms list
                acc_rms.append(a)

            vel_dim = 'v_' + dim
            pwr_dim = 'pwr_' + dim
            pos_dim = 'pos_' + dim
            force_dim = 'force_' + dim

            # Overwrite data points with re-calculated points
            acc_rms = array(acc_rms)
            data[acc_dim] = acc_rms
            data[vel_dim] = calc_integral(acc_rms, scale=bin_size, fs=fs)
            data[pwr_dim] = acc_rms * weight * data[vel_dim]
            data[pos_dim] = calc_pos(acc_rms, scale=bin_size, fs=fs)
            data[force_dim] = calc_force(acc_rms, weight)

    try:
        true_reps = header['final_num_reps'] if header['final_num_reps'] > 0 else header['init_num_reps']
    except IndexError:
        print 'Couldnt find number of reps in header from lift_id {}.\n Cannot learn from this file'
        print 'Defaulting to threshold of 1'
        # return {'a_thresh': 1., 'v_thresh': 1., 'p_thresh': 1., 'a_error': None, 'v_error': None, 'p_error': None}

    #: NOTE: This process considers each signal independently of the others.
    #:        May want to revisit this and figure something out to adjust all three simultaneously..

    thresh_dict = {}
    signal_tracking = {}
    all_headers = accel_headers + vel_headers + pwr_headers + pos_headers + force_headers
    for dim in all_headers:

        #: For use in plotting, if wanted
        signal = data[dim]

        #: For use in learning - number of signal points to exclude from calculation of standard deviation
        offset = 0
        scale = 0.

        signal_imp, signal_thresh, signal_reps = method1(signal, calc_offset=offset, scale=scale)

        err = signal_reps - true_reps
        err_tracking = [err]
        cnt = 0
        scales = [scale]
        while abs(err) > 0:
            #: Increasing scale increases the value of the threshold, thereby making the classifier LESS sensitive
            #: Decreasing scale makes classifier MORE sensitive
            if (err > 0) | (signal_thresh <= 0):
                scale += alpha
            elif err < 0:
                # scale *= (1.-alpha)
                scale -= alpha
            else:
                print 'Shouldnt reach this'

            signal_imp, signal_thresh, signal_reps = method1(signal, calc_offset=offset, scale=scale)

            err = signal_reps - true_reps
            err_tracking.append(err)
            scales.append(scale)
            if verbose:
                print 'iteration {c}: error of {e} reps'.format(c=cnt, e=err)
            cnt += 1

            #: Force break if err can't get to 0
            if cnt > int(300./alpha):
                print 'lift_id {l} ({t}): Couldnt converge to 0 error after {n} iterations. ' \
                      '(error: {e} reps). Breaking..'.format(t=header['lift_type'], l=lift_id, n=cnt, e=err)
                break

        #: Store are pieces necessary to build plots for EACH signal
        signal_tracking[dim] = {}
        signal_tracking[dim]['signal'] = signal
        signal_tracking[dim]['signal_imp'] = signal_imp
        signal_tracking[dim]['signal_thresh'] = signal_thresh
        signal_tracking[dim]['err_tracking'] = err_tracking
        signal_tracking[dim]['scale'] = scale
        signal_tracking[dim]['scales'] = scales
        signal_tracking[dim]['signal_reps'] = signal_reps

        #: Attempt at basic learning
        if verbose:
            print 'Final {l} cutoff: {t}'.format(l=dim, t=signal_thresh)

        #: Store threshold result for each signal
        thresh_dict.update({dim+'_thresh': signal_thresh, dim+'_error':err})

    # if plot:
        #: Plot learning curves
        # plot_learning(signal_tracking)
        # plot_cutoffs(signal_tracking)

    print 'Done with lift_id {l} ({t})'.format(l=lift_id, t=header['lift_type'])

    return thresh_dict


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
#         rms_raw = [sqrt(mean(dat.iloc[i].x**2 + dat.iloc[i].y**2 + dat.iloc[i].z**2)) for i, _ in enumerate(tuple(dat.x))]
#         a_rms = list()
#         #: Implement the simple high pass filter
#         for i in range(len(rms_raw)):
#             if i > 0:
#                 a = simple_highpass(rms_raw[i], rms_raw[i-1], a_rms[-1], fc=1., fs=fs)
#             else:
#                 a = 0
#             a_rms.append(a)
#
#         a_rms = array(a_rms)
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
#     percent_correct = float(np_sum((test_labels == forest_pred)*1))/float(len(test_labels))
#     print 'Random forest percent correct classification: {}'.format(percent_correct*100.)
#
#     # pred = nn.predict(array(test_storage.T.fillna(test_storage.T.mean())))
#     # score = nn.score(array(test_storage.T.fillna(test_storage.T.mean())), array(test_labels))
#
#     nn_pred = make_predictions(train=train_storage, train_labels=train_labels, test=test_storage, predictor='neural_net')
#     percent_correct = float(np_sum((test_labels == nn_pred)*1))/float(len(test_labels))
#     print 'Neural network percent correct classification: {}'.format(percent_correct*100.)


# From an input power vector, detect any change in state and increment
def calc_reps((acc, vel, pwr, pos, force), tracker):
    """
    Simple - a crossing of ALL the thresholds (a, v, p) indicates a change in state. From this, determine where the
    user was (in the state-space), and adjust accordingly.

    Takes determination of current state and calculated number of reps and updates tracker object with info. Passes
    back any crossings determined and the updated tracker object.

    :param acc: list-like acceleration vector. will be converted to pandas Series
    :param vel: list-like velocity vector. will be converted to pandas Series
    :param pwr: list-like power vector. will be converted to pandas Series
    :param pos: list-like position vector. will be converted to pandas Series
    :param tracker: (dict) dictionary with relevant metadata for the tracker collecting this data
    :return:
    """

    #: For debug purposes
    # a_thresh = 1.
    # v_thresh = 1.
    # p_thresh = 1.

    n_reps = tracker['calc_reps']
    state = tracker['curr_state']

    data = acc.join(vel).join(pwr).join(pos).join(force)

    diff_data = DataFrame()
    for label in ALL_SIGS:
        t_label = label + '_thresh'
        if t_label in tracker.keys():
            thresh = tracker[t_label]
        else:
            thresh = 1.

        #: Want to keep track of timepoints too
        # for multi-dim signal
        # diff_list.append( ((data[label] > thresh) > 0) * 1)
        diff_data[label] = (data[label] > thresh) * 1

        # for single dimensional signal
        # diff_list.append((signal > thresh) * 1)

    # AND the signals together - will keep only the crossings where ALL signals cross thresholds
    # MORE SENSITIVE
    # diff_signal = (diff_list[0] * diff_list[1] * diff_list[2] * diff_list[3]).diff()[1:]
    diff_signal = ( (diff_data.sum(axis=1) > 5) * 1).diff()[1:]

    # SUM the signal together and apply a thresh of > 2; anywhere at least 2 of the signals cross counts
    # LESS SENSITIVE
    # diff_signal = (((diff_list[0] + diff_list[1] + diff_list[2]) > 2) * 1).diff()[1:]

    # Drops two points off front, but forces signal to stay above/below threshold for at least
    # 1 point to be considered a change in state - better than considering any noise around threshold
    # as a change in state. Compare this to how it was before:
    # N = float( ((pwr > thresh) * 1).diff()[1:].abs().sum() )
    N = float( abs(diff_signal).sum() )

    # numpy where() is resource intensive - just map for now
    # Want to identify any shift in the state of the user
    if state == 'rest':
        shift = 0
    elif state == 'lift':
        # Here the athlete was mid-lift, so any deltas seen should be offset by 1 to acknowledge the athlete
        # started from 'lift' position
        shift = 1
        # N += 1
    else:
        print 'Unsure of lift state {}. Will assume "rest"'.format(state)

    # N = number of threshold crossings (absolute value)
    # So, if user is at rest, it will take 2 crossings to count as a full rep
    # If user is mid-rep, then it will only take 1 crossing (the downswing) to complete a rep
    n_reps += floor( (shift+N)/2. )
    print 'User at {} reps'.format(n_reps)

    # Update the state
    # Based on number of state changes, find new state of object
    # new_state = ['rest', 'lift'][int((shift+(N % 2)) % 2)]

    # Find AND TRACK all crossings IF THERE ARE ANY
    # Assign rep action based on direction of diff (+ = start, - = stop)
    crossings = diff_signal[diff_signal != 0].to_frame('diff')

    if crossings.shape[0] > 0:
        crossings['event'] = 'start_rep'
        crossings.loc[crossings['diff'] < 0, 'event'] = 'stop_rep'

        # Update state of object
        if diff_signal[crossings.index[-1]] > 0:
            new_state = 'lift'
        else:
            new_state = 'rest'

        crossings['timepoint'] = (tracker['max_t'] + crossings.index * (1. / tracker['sampling_rate'])).values
        crossings['lift_id'] = tracker['lift_id']

    # If there aren't any crossings, then just return whatever came in
    else:
        new_state = state
        crossings = None

    # update state of user via 'tracker' dict
    tracker['calc_reps'] = n_reps
    tracker['curr_state'] = new_state
    tracker['max_t'] += len(acc) * 1. / tracker['sampling_rate']  # track the last timepoint

    return tracker, crossings


#: Plot error curve (distance between estimated reps and actual reps
#: 3x2 (RxC) grid of plots, each row should be a new signal
# def plot_learning(track_dict):
#     f, axarr = plt.subplots(len(track_dict.keys()), 2, sharex=True)
#     for i, sig in enumerate(track_dict.keys()):
#         axarr[2*i].plot(track_dict[sig]['err_tracking'])
#         axarr[2*i].set_title('{s} - Error curve (Final Rep Count: {r})'.format(s=sig, r=track_dict[sig]['signal_reps']))
#         axarr[2*i].set_xlabel('Iteration')
#         axarr[2*i].set_ylabel('Estimated - Actual')
#         axarr[(2*i)+1].plot(track_dict[sig]['scales'])
#         axarr[(2*i)+1].set_title('{s} - Sigma Multiplier (Final Value: {v})'.format(s=sig, v=track_dict[sig]['scale']))
#         axarr[(2*i)+1].set_xlabel('Iteration')
#         axarr[(2*i)+1].set_ylabel('Scale Value')


#: Plot the signals with the thresholds and the rep start points
# def plot_cutoffs(track_dict):
#     plt.figure(10)
#     for sig in track_dict.keys():
#         if sig == 'acc':
#             p_color = 'black'
#         elif sig == 'vel':
#             p_color = 'blue'
#         else:
#             p_color = 'purple'
#         plt.plot(track_dict[sig]['signal'], color=p_color)
#         plt.axhline(y=track_dict[sig]['signal_thresh'], color=p_color, linestyle='--')
#         for x in where(track_dict[sig]['signal_imp'] > 0)[0]:
#             plt.axvline(x=x, color=p_color, linestyle='-.')


def load_thresh_dict(fname='thresh_dict.txt'):
    try:
        tmp = open(fname, 'r')
        thresh = loads(tmp.read())
        tmp.close()
        print 'Loaded thresh_dict from file {}'.format(fname)
    except IOError:
        print 'Couldnt find saved thresh_dict file {}'.format(fname)
        thresh = find_threshold(alpha=0.05, smooth=True, plot=False, verbose=False)
        with open(fname, 'w') as outfile:
            dump(thresh, outfile)

    return thresh