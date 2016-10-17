from redis import Redis
from json import loads, dumps


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
        # For wahtever reason, I am making all this the default collar values
        # Should only occur for DEV purposes
        collar_obj = {"collar_id": collar_id,
                      "athlete_id": 0,
                      "lift_id": 'None',
                      "lift_start": "None",
                      "lift_type": "deadlift",
                      "lift_weight": 100,
                      "lift_weight_units": "lbs",
                      "lift_num_reps": 10,
                      "calc_reps": 0,
                      "threshold": "None",
                      "curr_state": 'rest'
                      }
    return collar_obj


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


# def update_collars(redis_client=None, collars=None, verbose=True):
#     if redis_client is None:
#         print 'No redis client passed??'
#         return None
#     if collars is None:
#         print 'No collars object provided??'
#         return None
#
#     # update local storage
#     response = redis_client.set('collars', dumps(collars))
#     if response & verbose:
#         print 'Successfully updated "collars" redis object.'
#     if not response:
#         print 'Trouble saving "collars" variable to redis server. Not sure what to do'
