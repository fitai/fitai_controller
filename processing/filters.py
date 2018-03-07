from scipy.signal import butter, lfilter, lfilter_zi
import numpy as np


#: calculating numerators/denominators. May also want to implement the filtering in basic math.
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_highpass(lowcut, fs, order):
    nyq = fs/2.
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    # plot_frequency_response(b, a, lowcut, fs)
    return b, a


def butter_lowpass(highcut, fs, order):
    nyq = fs/2.
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def butter_lowpass_filter(signal, y0, highcut, fs, order):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, signal)
    return y


def butter_highpass_filter(signal, y0, lowcut, fs, order):
    b, a = butter_highpass(lowcut, fs, order=order)
    # zi = lfilter_zi(b, a)
    zi = [y0['y']]

    signal = np.insert(signal, 0, y0['x'])  # insert initial condition
    y, _ = lfilter(b, a, signal, zi=zi)
    y = y[1:]  # remove initial condition
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


#: Starting with the bandpass filter, but want to
def filter_signal(signal, y0, filter_type, freqs, fs, filter_order):
    #: Apply filter
    if filter_type == 'highpass':
        # y = butter_highpass_filter(signal, y0, lowcut=freqs[0], fs=fs, order=filter_order)
        y = simple_highpass(signal, y0, fc=freqs[0], fs=fs)

    elif filter_type == 'bandpass':
        y = butter_bandpass_filter(signal, lowcut=freqs[0], highcut=freqs[1], fs=fs, order=filter_order)

    elif filter_type == 'lowpass':
        y = butter_lowpass_filter(signal, highcut=freqs[1], fs=fs, order=filter_order)

    return y


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


# y0 = {'x': x0, 'y': y0}
def simple_highpass(signal, y0, fc=.1, fs=30.):
    # establish constants
    RC = 1. / (fc * 2. * np.pi)
    dt = 1. / fs
    alpha = RC / (RC + dt)

    # initial code
    # set initial conditions (grows signal by 1 (temporary))
    filt = [y0['y']]
    signal = np.insert(signal, 0, y0['x'])

    for i in range(1, len(signal)):
        x = alpha * (filt[-1] + signal[i] - signal[i-1])  # y[i] := alpha * y[i-1] + alpha * (x[i] - x[i-1])
        filt.append(x)

    # remove inserted initial condition point
    filt = filt[1:]

    return filt
