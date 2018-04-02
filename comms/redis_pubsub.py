from datetime import datetime as dt
from json import dumps


# TODO - clean this up, too much overhead
def prep_message(tracker_obj, vel, pwr):
    now = dt.now()
    if isinstance(tracker_obj['created_at'], type(now)):
        tracker_obj['created_at'] = dt.strftime(tracker_obj['created_at'], '%Y-%m-%d %H:%M:%S')
    if isinstance(tracker_obj['updated_at'], type(now)):
        tracker_obj['updated_at'] = dt.strftime(tracker_obj['updated_at'], '%Y-%m-%d %H:%M:%S')

    msg_dict = {"header": tracker_obj,
                "rep_count": tracker_obj['calc_reps'],
                "content":
                    {"v_rms": list(vel['rms']),
                     "p_rms": list(pwr['rms'])}
                }
    msg = dumps(msg_dict)
    return msg


def redis_pub(redis_client, redis_channel, tracker_obj, vals, source):

    if source == 'real_time':
        vel, pwr = vals[1], vals[2]  # pull out relevant entries
        msg = prep_message(tracker_obj, vel, pwr)
    elif source == 'rfid':
        msg = vals  # pass message payload directly
    else:
        print 'Unsure what source of message is: {}'.format(source)
        raise NotImplementedError

    if redis_client is not None:
        redis_client.publish(redis_channel, msg)
    else:
        print 'Redis connection not established. Cannot publish message.'
