# To learn how to identify rep start/stop points
from __future__ import division

from pandas import Series, concat
from numpy import zeros, var, mean
from copy import copy

from processing.filters import filter_signal


# attempts to find start/stop points in incoming signal
def run_detector(vel, sampling_rate, prev_dat, prev_vz, prev_filt_vz, packet_size, t_prev, hold, cross_track, ts,
                 var_max, min_irt_samples, min_intra_samples, starts, stops, n_reps, t_min, sig_track=None, p_track=None):
    sig = vel.copy()
    v0 = {'x': prev_vz, 'y': prev_filt_vz}
    sig = Series(filter_signal(sig.values, v0, 'highpass', [.1, None], sampling_rate, 1), index=sig.index, name='v_z')

    if len(prev_dat) < 1:
        # var_ = sig.rolling(window=10, min_periods=0, center=False).apply(lambda y: var(sorted(y)[2:-2])).fillna(0.)
        p_ = Series(zeros(packet_size), name=sig.name)
        prev_dat.append(sig)
    else:
        sig_ = concat(prev_dat + [sig], axis=0)
        var_ = sig_.rolling(window=10, min_periods=0, center=False).apply(lambda y: var(sorted(y)[2:-2])).fillna(0.)
        mask_var = (var_ > 0.02 * var_max) * 1
        v_m = mask_var.rolling(window=10, min_periods=0, center=False).apply(mean)

        p_ = ((v_m > 0.05) * 1).diff().fillna(0.).iloc[packet_size:]
        # var_ = var_.iloc[packet_size:]  # used for tracking signal
        prev_dat[0] = sig  # overwrite old packet

    if sig_track is not None:
        sig_track = sig_track.append(sig)
    if p_track is not None:
        p_track = p_track.append(p_)

    # look for crossings
    # any positive values represent crossings from 0 to 1
    # any negative values represent crossings from 1 to 0
    # check that there were any crossings, and that
    # crossings are far enough away from most recent prior stop
    # NOTE: result of p_ > 0 is a Series, result of p_.index - t_prev is an array. These two cannot be
    #       combined directly via & ; have to convert the Series to an array via Series.values
    crossings = p_.loc[(abs(p_) > 0).values & ((p_.index - t_prev) >= min_irt_samples)]

    # handle any crossings
    if crossings.shape[0] > 0:
        # before moving forward, if no positive crossings have been registered (i.e. len(cross_track) == 0),
        #  the next crossing must be positive. Enforce this here
        if (len(cross_track) == 0) and any(crossings == -1) and (crossings.shape[0] > 1):
            # eliminate any negative crossing before a positive crossing
            tc = crossings.loc[crossings == 1].index[0]  # id index of first positive crossing
            crossings = crossings.loc[tc:]  # slice out anything before first positive crossing
        elif len(cross_track) == 0 and any(crossings == -1) and (crossings.shape[0] == 1):
            # only one crossing, and it's negative; do not log this
            return sig, prev_dat, hold, cross_track, ts, n_reps, sig_track, p_track

        # update tracking variables
        if len(cross_track) == 0:
            cross_track = list(crossings.values)
            ts = list(crossings.index)
        else:
            cross_track += list(crossings.values)
            ts += list(crossings.index)

        if len(cross_track) < 2:
            return sig, prev_dat, hold, cross_track, ts, n_reps, sig_track, p_track
        elif len(cross_track) > 2:
            if ts[2] - ts[1] > min_irt_samples:
                hold_c = cross_track[2:]  # preserve future crossings
                hold_t = ts[2:]
                hold = True
            cross_track = cross_track[:2]  # move forward with current crossings
            ts = ts[:2]

        # ensure the rep lasted at least sampling_rate samples - cut out jitter
        if ts[1] - ts[0] > min_intra_samples:
            print('logging start/stop pair')
            starts.append(ts[0] - t_min)
            stops.append(ts[1] - t_min)
            t_prev = ts[1]
            n_reps += 1
            print('user at {} reps'.format(n_reps))

        # triggers when len(cross_track) >= 4
        # clear tracking variables
        if hold and (ts[0] - t_prev >= min_irt_samples):
            cross_track = copy(hold_c)
            ts = copy(hold_t)
        else:
            cross_track = []
            ts = []

        # because of the "and" in the if condition, hard-reset hold each time this triggers, just in case
        hold = False

    return sig, prev_dat, hold, cross_track, ts, n_reps, sig_track, p_track
