from redis import Redis
from json import loads, dumps


#: Instantiate a default collar object
#: All fields that trigger logic are commented. DO NOT CHANGE
def get_default_collar():
    return {"collar_id": 'None',  # Needs to be "None"
            "athlete_id": 0,
            "lift_id": 'None',  # Needs to be "None"
            "lift_start": "None",  # Needs to be "None"
            "lift_type": "Barbell Deadlift",
            "lift_weight": 100,
            "weight_units": "lbs",
            "init_num_reps": 0,
            "final_num_reps": 0,
            "calc_reps": 0,
            "a_thresh": "None",
            "v_thresh": "None",
            "p_thresh": "None",  # Needs to be "None"
            "curr_state": 'rest',
            "max_t": 0.,
            "sampling_rate": 20.
            }


#: Opens connection to redis server on whichever host is passed
def establish_redis_client(hostname='localhost', port=6379, password=None, verbose=False):
    r = Redis(host=hostname, port=port, password=password)
    if r is not None:
        if verbose:
            print 'Successfully connected to redis at {}:{}'.format(hostname, port)
        return r
    else:
        if verbose:
            print 'Failed to connect to redis at {}:{}'.format(hostname, port)
        return None


#: Queries redis client for collar_id passed in
def retrieve_collar_by_id(redis_client=None, collar_id=None, verbose=True):

    if redis_client is None:
        print 'No redis client passed??'
        return None
    if collar_id is None:
        print 'No collar id passed??'
        return None

    # NOTE TO SELF: redis doesn't throw error for missing value?
    try:
        collar_obj = loads(redis_client.get(collar_id))
    except TypeError:
        if verbose:
            print 'Collar {} not found in Redis. Will create new'.format(collar_id)
        # For whatever reason, I am making all this the default collar values
        # Should only occur for DEV purposes
        collar_obj = get_default_collar()
        collar_obj['collar_id'] = collar_id
    except ValueError:
        if verbose:
            print 'Collar {} somehow corrupted and no longer a JSON string. Will create new.'.format(collar_id)
        collar_obj = get_default_collar()
        collar_obj['collar_id'] = collar_id

    return collar_obj


#: Sets redis object that matches collar_id to a JSON string version of 'dat' - the dataframe passed in
def update_collar_by_id(redis_client=None, dat=None, collar_id=None, verbose=True):

    if redis_client is None:
        print 'No redis client passed??'
        return None
    if dat is None:
        print 'No collar data object provided??'
        return None
    if collar_id is None:
        print 'No collar id provided??'
        return None

    # update local storage
    response = redis_client.set(collar_id, dumps(dat))
    if response & verbose:
        print 'Successfully updated collar {} redis object.'.format(collar_id)
    if not response:
        print 'Trouble saving collar {} variable to redis server. Not sure what to do'.format(collar_id)

    # update_collars(redis_client, collars, verbose)
    return response


#: Wipes redis object at collar_id and replaces with the default collar object
#: Should only use this if things to horribly wrong
def reset_collar_by_id(collar_id, redis_client=None):
    print 'resetting collar {} to default values...'.format(collar_id)
    if redis_client is None:
        redis_client = establish_redis_client()

    update_collar_by_id(redis_client=redis_client, dat=get_default_collar(), collar_id=collar_id)
