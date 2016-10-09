from scipy.signal import butter, lfilter
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import freqz


#: TO DO: Replace butter_bandpass/highpass/lowpass with the relevant math. iOS may not have packages to support
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
    # print 'low: {}'.format(low)
    b, a = butter(order, low, btype='highpass')
    # plot_frequency_response(b, a, lowcut, fs)
    return b, a


def butter_lowpass(highcut, fs, order):
    nyq = fs/2.
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


#: Starting with the bandpass filter, but want to
def filter_signal(signal, filter_type='bandpass', f_low=1.0, f_high=40., fs=100., filter_order=3):
    # # Sample rate and desired cutoff frequencies (in Hz).
    # fs = 5000.0
    # lowcut = 500.0
    # highcut = 1250.0

    # # Plot the frequency response for a few different orders.
    # plt.figure(1)
    # plt.clf()
    # for order in [3, 6, 9]:
    #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #     w, h = freqz(b, a, worN=2000)
    #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    #
    # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
    #          '--', label='sqrt(0.5)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')
    #
    # # Filter a noisy signal.
    # T = 0.05
    # nsamples = T * fs
    # t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    # f0 = 600.0
    # x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    # x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    # x += a * np.cos(2 * np.pi * f0 * t + .11)
    # x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    # plt.figure(2)
    # plt.clf()
    # plt.plot(t, x, label='Noisy signal')

    #: Apply filter
    if filter_type == 'highpass':
        y = butter_highpass_filter(signal, lowcut=f_low, fs=fs, order=filter_order)

    elif filter_type == 'bandpass':
        y = butter_bandpass_filter(signal, lowcut=f_low, highcut=f_high, fs=fs, order=filter_order)

    elif filter_type == 'lowpass':
        y = butter_lowpass_filter(signal, highcut=f_high, fs=fs, order=filter_order)

    return y
    # plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    # plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    # plt.grid(True)
    # plt.axis('tight')
    # plt.legend(loc='upper left')
    #
    # plt.show()


#: Will want to look in to the specifics of scipy.freqz so that we can recreate this, if need be, in the app
# def plot_frequency_response(b, a, cutoff, fs):
#     # Plot the frequency response.
#     w, h = freqz(b, a, worN=8000)
#     # plt.subplot(2, 1, 1)
#     plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#     plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#     plt.axvline(cutoff, color='k')
#     plt.xlim(0, 0.5*fs)
#     plt.title("Lowpass Filter Frequency Response")
#     plt.xlabel('Frequency [Hz]')
#     plt.grid()


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def simple_highpass(curr_raw, prev_raw, prev_filtered, fc=1., fs=100.):

    RC = 1. / (fc * 2. * 3.14)
    dt = 1. / fs
    alpha = RC / (RC + dt)

    output = alpha * (prev_filtered + curr_raw - prev_raw)
    return output
