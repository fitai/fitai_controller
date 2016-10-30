from websocket import create_connection
import json


# Establish websocket connection with the given IP and host
# Params are self-explanatory. Note they need to be strings
def establish_connection(ip='52.204.229.101', port='8080', type='ws'):
    conn_string = type + '://' + ip + ':' + port
    try:
        print 'Attempting {t} connection - {c}'.format(t='websocket' if type == 'ws' else type, c=conn_string)
        ws = create_connection(conn_string)
    except ValueError, e:
        print '!!! Couldnt connect !!! Error: {}'.format(e)
        return None
    else:
        print 'Connection successful'
        return ws


# Simple - publish to the websocket.
# Params - n_iter (int): number of iterations to publish the mesage
#        - sleep_time (int): number seconds each loop should wait
def push_message(ws, msg=None):
    if msg is None:
        print 'No message provided. Skipping message push...'
        return None
    # print 'Sending message {}'.format(msg)
    print 'sending message to websocket..'
    ws.send(msg)
    print 'Message sent'


# Close the websocket connection
def close_connection(ws):
    print 'closing websocket..'
    ws.close()


# Publish the given header, velocity, and power lists to the PHP websocket server
def ws_pub(collar_obj, vel, pwr, reps=0):
    # msg = '{"header": { "u_id": 0, "lift_id": 1}, "content": {"v_rms": [0, 1, 2, 3, 4], "p_rms": [5, 6, 7, 8, 9] }}'
    msg_dict = {"header": collar_obj, "rep_count": reps, "content": {"v_rms": list(vel), "p_rms": list(pwr)}}
    msg = json.dumps(msg_dict)
    print 'message to websocket: \n{}'.format(msg)
    ws = establish_connection()
    if ws is not None:
        push_message(ws, msg)
        close_connection(ws)
    else:
        print 'Websocket not established. Cannot push message. No websocket to close.'
