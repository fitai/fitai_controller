from os.path import join, dirname, realpath

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processing.functions import method1, calc_vel2
from FitAI.load_data import load_file
from FitAI.predictors import make_predictions
from processing.filters import simple_highpass

# Should use known weight for demo
weight = 50  # kg
fs = 100.  # Sampling rate (default = 100Hz)
bin_size = 1.  # Number of samples to collect before processing the signal (float)


def find_threshold(person_folder=None, filename=None, data_folder=None, smooth=True):
    if not data_folder:
        try:
            dir_path = dirname(dirname(realpath(__file__)))
            print 'found path {}'.format(dir_path)
        except NameError:
            print 'Working in dev mode.'
            dir_path = '/Users/kyle/PycharmProjects'
        data_folder = join(dir_path, 'FitAI/data_files/')

    if not person_folder:
        person_folder = 'Kyle'
        print 'no person_folder passed. defaulting to {}'.format(person_folder)

    if not filename:
        filename = 'OHP_45lb_10rep_1.csv'
        print 'no filename passed. defaulting to {}'.format(filename)

    col_names = ['timestamp', 'x', 'y', 'z']
    data = load_file(join(data_folder, person_folder, filename), names=col_names, skiprows=1)
    # dat = data[['x', 'y', 'z']].iloc[0:250]
    dat = data[['x', 'y', 'z']]

    rms_raw = [np.sqrt(np.mean(dat.iloc[i].x**2 + dat.iloc[i].y**2 + dat.iloc[i].z**2)) for i, _ in enumerate(tuple(dat.x))]

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
    v_euler = calc_vel2(a_rms, scale=bin_size, fs=fs)
    pwr_rms = a_rms * weight * v_euler

    #: For use in learning - number of signal points to exclude from calculation of standard deviation
    offset = 50
    scale = 0.

    pwr_imp, pwr_thresh, pwr_reps = method1(pwr_rms, calc_offset=offset, scale=scale)

    try:
        true_reps = int([x for x in filename.split('_') if 'rep' in x][0].split('rep')[0])
    except IndexError:
        print 'Couldnt find number of reps in filename {}.\n Cannot learn from this file'.format(filename)
        print 'Defaulting to threshold of 1'
        return 1.

    err = pwr_reps - true_reps
    err_tracking = [err]
    cnt = 0
    scales = [scale]
    while np.abs(err) > 0:
        #: Increasing scale increases the value of the threshold, thereby making the classifier LESS sensitive
        #: Decreasing scale makes classifier MORE sensitive
        if err > 0:
            scale += 0.1
        elif err < 0:
            scale -= 0.1
        else:
            print 'Shouldnt reach this'

        pwr_imp, pwr_thresh, pwr_reps = method1(pwr_rms, calc_offset=offset, scale=scale)

        err = pwr_reps - true_reps
        err_tracking.append(err)
        scales.append(scale)
        print 'iteration {c}: error of {e} reps'.format(c=cnt, e=err)
        cnt += 1

        #: Force break if err can't get to 0
        if cnt > 20:
            print 'Couldnt converge. Breaking..'
            break

    # #: Plot learning curves
    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(err_tracking)
    # axarr[0].set_title('Error curve (Final Rep Count: {})'.format(pwr_reps))
    # axarr[0].set_xlabel('Iteration')
    # axarr[0].set_ylabel('Estimated - Actual')
    # axarr[1].plot(scales)
    # axarr[1].set_title('Sigma Multiplier (Final Value: {})'.format(scale))
    # axarr[1].set_xlabel('Iteration')
    # axarr[1].set_ylabel('Scale Value')
    #
    # #: Attempt at basic learning
    # print 'Final power cutoff: {}'.format(pwr_thresh)
    #
    # plt.figure(10)
    # p_color = 'purple'
    # plt.plot(pwr_rms, color=p_color)
    # plt.axhline(y=pwr_thresh, color=p_color, linestyle='--')
    # for x in np.where(pwr_imp > 0)[0]:
    #     plt.axvline(x=x, color=p_color, linestyle='-.')

    return pwr_thresh


def try_classifier():
    data_folder = '/Users/kyle/PycharmProjects/FitAI/data_files/'
    person_folder = 'Kyle'

    ohp_file1 = 'OHP_45lb_10rep_1.csv'  # 0 - 850
    ohp_file2 = 'OHP_45lb_10rep_2.csv'  # 0 - 950
    ohp_file3 = 'OHP_45lb_10rep_3.csv'  # 0 - 875

    dl_file1 = 'DL_45lb_10rep_1.csv'  # 0 - 1700
    dl_file2 = 'DL_45lb_10rep_2.csv'  # 0 - 1220
    dl_file3 = 'DL_45lb_10rep_3.csv'  # 0 - 1400

    # sq_file1 = 'Kyle_SQ_45lb_10rep_1.csv'  # NOTE: workout 1 ends @ 980, workout 2 ends @ 2200
    # sq_file2 = 'Kyle_SQ_45lb_10rep_2.csv'

    filenames = [ohp_file1, ohp_file2, ohp_file3, dl_file1, dl_file2, dl_file3]

    ## Split data into training and testing ##
    train_storage = pd.DataFrame()
    test_storage = pd.DataFrame()
    train_labels = []
    test_labels =[]
    for filename in filenames:
        col_names = ['timestamp', 'x', 'y', 'z']
        data = load_file(join(data_folder, person_folder, filename), names=col_names, skiprows=1).iloc[0:850]
        # Extract data from file contents
        dat = data[['x', 'y', 'z']]
        rms_raw = [np.sqrt(np.mean(dat.iloc[i].x**2 + dat.iloc[i].y**2 + dat.iloc[i].z**2)) for i, _ in enumerate(tuple(dat.x))]
        a_rms = list()
        #: Implement the simple high pass filter
        for i in range(len(rms_raw)):
            if i > 0:
                a = simple_highpass(rms_raw[i], rms_raw[i-1], a_rms[-1], fc=1., fs=fs)
            else:
                a = 0
            a_rms.append(a)

        a_rms = np.array(a_rms)
        v_euler = calc_vel2(a_rms, scale=bin_size, fs=fs)
        pwr_rms = a_rms * weight * v_euler

        tmp_df = pd.DataFrame(data={'accel': a_rms})

        label = filename.split('_')[1]
        filenum = int(filename.split('_')[-1].split('.')[0])

        # First 2 of 3 files are for training
        if filenum < 3:
            train_labels.append(label)
            train_storage = pd.concat([train_storage, tmp_df], axis=1)
        # File 3 of 3 is for testing
        else:
            test_labels.append(label)
            test_storage = pd.concat([test_storage, tmp_df], axis=1)

    ## Weak pre-processing
    train_storage = train_storage.fillna(train_storage.mean())
    test_storage = test_storage.fillna(test_storage.mean())

    forest_pred = make_predictions(train=train_storage, train_labels=train_labels, test=test_storage, predictor='forest')
    percent_correct = float(np.sum((test_labels == forest_pred)*1))/float(len(test_labels))
    print 'Random forest percent correct classification: {}'.format(percent_correct*100.)

    # pred = nn.predict(np.array(test_storage.T.fillna(test_storage.T.mean())))
    # score = nn.score(np.array(test_storage.T.fillna(test_storage.T.mean())), np.array(test_labels))

    nn_pred = make_predictions(train=train_storage, train_labels=train_labels, test=test_storage, predictor='neural_net')
    percent_correct = float(np.sum((test_labels == nn_pred)*1))/float(len(test_labels))
    print 'Neural network percent correct classification: {}'.format(percent_correct*100.)


# From an input power vector, detect any change in state and increment
def calc_reps(pwr, n_reps, state='rest', thresh=0.):
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
    # Better way to do this??
    N = float(((pwr > thresh) * 1).diff()[1:].abs().sum())

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

    # Assume every
    n_reps += np.floor(N/2.)
    print 'User at {} reps'.format(n_reps)

    # Update the state
    state = ['rest', 'lift'][int((shift+(N%2))%2)]

    return n_reps, state
