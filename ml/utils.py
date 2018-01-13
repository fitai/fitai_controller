import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from databasing.conn_strings import db_conn_string


def proximity_dist(s1, s2):
    s1, s2 = truncate(s1, s2)

    # calculate mean + variance of distances between points across two series that are closest to each other
    dists = []
    for val in s1:
        d = abs(s2 - val)
        i = np.argmin(d)
        dist = np.abs(val - s2[i])
        dists.append(dist)

    m = np.mean(dists)
    s = np.std(dists)

    return m, s


def rigid_ordered_dist(s1, s2):
    # values in are two series aligned as presented, then distance is calculated

    s1, s2 = truncate(s1, s2)

    dists = np.abs(s1 - s2)

    m = np.mean(dists)
    s = np.std(dists)

    return m, s


def prox_ordered_dists(s1, s2, remove_s2=[]):
    s1, s2 = truncate(s1, s2)

    if len(remove_s2) > 0:  # we know a priori to drop some points
        for idx in remove_s2:
            del s2[idx]

    # values must go in order, but value from s1 is matched with closest of its neighbors in s2
    dists = []
    removed = []
    for i, val in enumerate(s1):
        if i < len(s2) - 1:
            d_l = np.abs(val-s2[i])  # distance to left point in s2
            d_r = np.abs(val-s2[i+1])  # distance to right point in s2

            shift = np.argmin([d_l, d_r])
            dist = np.abs(val - s2[i+shift])
            dists.append(dist)

            # the values in s1 skipped a value in s2 and all downstream points in s1 will be thrown off unless
            # this is accounted for
            # if shift > 0:
            #     del s2[i]  # remove the point that was skipped
            #     removed.append(i)
        else:
            dist = np.abs(val - s2[i])
            dists.append(dist)

    m = np.mean(dists)
    s = np.std(dists)

    return m, s, removed


def truncate(s1, s2):
    # ensure they are the same length
    if len(s1) > len(s2):
        s1 = s1[:len(s2)]
    elif len(s1) < len(s2):
        s2 = s2[:len(s1)]

    return s1, s2


def load_events():
    conn = create_engine(db_conn_string)

    events_sql = '''
    SELECT
        lift_id
        , ROUND(timepoint::NUMERIC, 2) AS timepoint
        , event
    FROM lift_event
    WHERE event IN ('rep_start', 'rep_stop')
    '''
    events = pd.read_sql(events_sql, conn)

    # make sure that all even timepoints occur on an even point (multiple of 0.02)
    # if the point is odd, then subtract 0.01 from it to make it even
    def clean_event_times(events):
        events['timepoint'] = [x if round(x * 100, 2) % 2 == 0 else round(x, 2) - 0.01 for x in events['timepoint']]
        return events

    events = clean_event_times(events)

    # pull data and metadata for all lifts in events df
    lift_ids = events['lift_id'].unique()

    return events, lift_ids


def build_rep_prob_signal(t, sampling, t_window):
    #: Based on input t_range (e.g. number seconds of signal to include), calculate
    #: appropriate mean, std
    #: Where mean = t_start (or t_stop) ( = t)
    #:       std = such that any values outside of +/- 4*std will be truncated to zero
    t_half = t_window/2.  # half of range on each side (+/-) of t
    sig = (t_half/3.)   # anything outside this gets truncated to prob = 0

    #: NOTE TO SELF: Look into why this doesn't create numbers spaced evenly
    # test = np.linspace(t - t_half, t + t_half, num=t_range * sampling)

    bound = int(t_half*sampling)
    # Round so that float conversion won't introduce error and make
    # resultant timepoints un-alignable with original data timepoints.
    ts = [round((float(x)/sampling), 3) for x in range(-1*bound, bound+1)]
    g_x = [np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.))) for x in ts]

    #: Convert to series
    g_x = pd.Series(data=g_x, index=ts)

    ts = [round(t + (float(x)/sampling), 3) for x in range(-1*bound, bound+1)]
    return g_x, ts


def build_centered_rep_prob_signal(t, sampling, t_window):
    #: Based on input t_range (e.g. number seconds of signal to include), calculate
    #: appropriate mean, std
    #: Where mean = t_start (or t_stop) ( = t)
    #:       std = such that any values outside of +/- 4*std will be truncated to zero
    t_half = t_window/2.  # half of range on each side (+/-) of t
    sig = (t_half/3.)   # anything outside this gets truncated to prob = 0

    #: NOTE TO SELF: Look into why this doesn't create numbers spaced evenly
    # test = np.linspace(t - t_half, t + t_half, num=t_range * sampling)

    bound = int(t_half*sampling)
    # Round so that float conversion won't introduce error and make
    # resultant timepoints un-alignable with original data timepoints.
    ts = [round((float(x)/sampling), 3) for x in range(-1*bound, bound+1)]
    g_x = [np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.))) for x in ts]

    #: Convert to series
    g_x = pd.Series(data=g_x, index=ts)

    ts = [round(t + (float(x)/sampling), 3) for x in range(-1*bound, bound+1)]
    return g_x, ts

