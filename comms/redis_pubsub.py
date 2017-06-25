from datetime import datetime as dt
from json import dumps


def prep_message(collar_obj, vel, pwr):
    if isinstance(collar_obj['lift_start'], type(dt.now())):
        collar_obj['lift_start'] = dt.strftime(collar_obj['lift_start'], '%Y-%m-%d %H:%M:%S')
    msg_dict = {"header": collar_obj,
                "rep_count": collar_obj['calc_reps'],
                "content":
                    {"v_rms": list(vel['rms']),
                     "p_rms": list(pwr['rms'])}
                }
    msg = dumps(msg_dict)
    return msg


def redis_pub(redis_client, collar_obj, vals):
    vel, pwr = vals[1], vals[2]  # pull out relevant entries

    msg = prep_message(collar_obj, vel, pwr)

    if redis_client is not None:
	redis_client.publish('lifts', msg)
    else:
        print 'Redis connection not established. Cannot publish message.'
