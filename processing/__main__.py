from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from FitAI.load_data import load_file
from processing.filters import butter_highpass, simple_highpass
from FitAI.test import find_threshold

# Should use known weight for demo
weight = 50.  # kg
fs = 100.  # Sampling rate (default = 100Hz)
bin_size = 10  # Number of samples to collect before processing the signal (float)
delta_t = 1./fs
v = 0.
#: Collect num_sec number of seconds in beginning of workout to use as noise
num_sec = 1.

data_folder = '/Users/kyle/PycharmProjects/FitAI/data_files/'
person_folder = 'Kyle'
#: File 2 is OHP, 10reps with a pause between start of recording and start of lift, plus pause after lift ends.
file2 = 'OHP_45lb_10rep_full_3.csv'

### Threshold value ###
#: Import from trained model(s)
#: MUST pass appropriate person folder and file. Without either this will break and return None
pwr_thresh = find_threshold(person_folder, file2)
#: In case the number of reps cannot be extracted from the filename, default to a threshold power value of 1
if not pwr_thresh:
    pwr_thresh = 1.

### Load in for processing ###
col_names = ['timestamp', 'x', 'y', 'z']
data = load_file(join(data_folder, person_folder, file2), names=col_names, skiprows=1)
data = data.iloc[0:500]
### Set up filter coefficients ###
b, a = butter_highpass(lowcut=1.0, fs=100., order=3)

#: For use in loop and plotting after
a_rms = list()
a_rms_filt = list()
v_rms = list()
pwr_tracker = list()
mask_tracker = list()

#: Booleans to keep track of the phase of the lift
lift = False
rest = True
exercise = False
reps = 0

for i, t in enumerate(data.timestamp):
    dat = data.iloc[i, 1:]

    #: The acceleration RMS is the only signal we care about.. for now
    rms = np.sqrt(np.mean(dat.x**2 + dat.y**2 + dat.z**2))

    if i > 0:
        #: Temporary filter
        rms = 0.66*rms + 0.33*a_rms[-1]
        rms_filt = simple_highpass(rms, a_rms[-1], a_rms_filt[-1], fc=1., fs=fs)

        #: real-time integral
        v += (rms_filt * delta_t)
        pwr = weight * rms_filt * v

    else:
        #: First loop will not be able to calculate anything
        rms_filt = 0
        v = 0
        pwr = 0

    ### Phase 1 - Determine exercise start ###

    #: Simple - bar starts at rest, so any non-zero acceleration at this point is due to noise. Use this to develop
    #: a basic noise profile, e.g. max, min, mean. Use any variation clearly outside of this profile to trigger
    #: Phase 2

    if (not exercise) & (i > num_sec * fs):
        noise_max = max(pwr_tracker)
        noise_min = min(pwr_tracker)
        noise_std = np.std(pwr_tracker)

        if pwr > noise_max + 3. * noise_std:
            print 'pwr broke threshold of {p} on iteration {i}'.format(p=noise_max+3.*noise_std, i=i)
            exercise = True
            exercise_iter = i

    ### Phase 2 - Track reps ###

    #: Key logical structure to the algorithm. Want to figure out where the signal is, and whether it has trigger
    #: certain transitions. Tracking these phase changes will determine if a REP is counted or not. I have
    #: tried to order the phases logically for the sake of anyone else looking at this, but the idea is that
    #: the user can be at REST phase or LIFT phase. The transition from REST to LIFT signals the start of a lift, and
    #: the transition from LIFT to REST signals the end of a lift, and adds a REP

    if exercise:
        #: REST in progress
        if (pwr < pwr_thresh) & (not lift) & rest:
            print 'At rest.'
        #: Transition from REST to LIFT
        elif (pwr > pwr_thresh) & (not lift) & rest :
            print 'Proper threshold crossing. Count as mid-lift'
            lift = True
            rest = False
            mask_tracker.append(1)
        #: LIFT in progress
        elif (pwr > pwr_thresh) & lift & (not rest):
            print 'Lift in progress'
            mask_tracker.append(1)
        #: Transition from LIFT to REST
        elif (pwr < pwr_thresh) & lift & (not rest):
            print 'Return to rest. Count as REP and end of lift'
            lift = False
            rest = True
            reps += 1
            mask_tracker.append(0)
        #: Should only happen in the beginning of the recording
        elif (pwr > pwr_thresh) & (not rest) & (not lift):
            print 'SIGNAL ARTIFACT'
        else:
            print 'Whats going on here??'
            print 'Power: {p}  (thresh {t})'.format(p=pwr, t=pwr_thresh)
            print 'Lift: {}'.format(lift)
            print 'Rest: {}'.format(rest)

    #: Record all real-time values for checking later
    v_rms.append(v)
    a_rms.append(rms)
    a_rms_filt.append(rms_filt)
    pwr_tracker.append(pwr)

## Plot results ##
plt.figure()
plt.title('Real-Time emulation. Counted {} reps'.format(reps))
plt.plot(v_rms, color='blue')
plt.plot(a_rms_filt, color='red')
plt.plot(pwr_tracker, color='purple')
plt.axhline(y=pwr_thresh, color='purple', linestyle='--')
plt.axvline(x=exercise_iter, color='black', linestyle='--')
