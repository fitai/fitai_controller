from datetime import datetime as dt
from json import dumps


# TODO - clean this up, too much overhead
def prep_message(collar_obj, vel, pwr):
    if isinstance(collar_obj['lift_start'], type(dt.now())):
        collar_obj['lift_start'] = dt.strftime(collar_obj['lift_start'], '%Y-%m-%d %H:%M:%S')
    if isinstance(collar_obj['created_at'], type(dt.now())):
        collar_obj['created_at'] = dt.strftime(collar_obj['lift_start'], '%Y-%m-%d %H:%M:%S')
    if isinstance(collar_obj['updated_at'], type(dt.now())):
        collar_obj['updated_at'] = dt.strftime(collar_obj['lift_start'], '%Y-%m-%d %H:%M:%S')

    msg_dict = {"header": collar_obj,
                "rep_count": collar_obj['calc_reps'],
                "content":
                    {"v_rms": list(vel['rms']),
                     "p_rms": list(pwr['rms'])}
                }
    msg = dumps(msg_dict)
    return msg


def redis_pub(redis_client, redis_channel, collar_obj, vals, source):

    if source == 'real_time':
        vel, pwr = vals[1], vals[2]  # pull out relevant entries
        msg = prep_message(collar_obj, vel, pwr)
    elif source == 'rfid':
        msg = vals  # pass message payload directly
    else:
        print 'Unsure what source of message is: {}'.format(source)
        raise NotImplementedError

    if redis_client is not None:
        redis_client.publish(redis_channel, msg)
    else:
        print 'Redis connection not established. Cannot publish message.'
