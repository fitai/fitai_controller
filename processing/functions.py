import numpy as np
import scipy as sp

from pandas import Series


def method1(signal, calc_offset=0, scale=3.):
    sigma = np.std(signal[calc_offset:])
    thresh = scale * sigma
    mask = (signal > thresh) * 1  # Convert boolean to binary
    # imp = mask[1:] - mask[:-1]
    imp = np.diff(mask)
    reps = np.sum(imp > 0)
    return imp, thresh, reps


#: Trapezoid method
def calc_vel1(signal, scale, fs):
    delta_t = scale / fs
    integral = np.cumsum((signal[1:] + np.diff(signal) / 2.) * delta_t)
    return integral


#: Euler method
#: Accepts single dimension of values (e.g. series or array)
#: Returns pandas Series
def calc_vel2(signal, scale=1, fs=20):
    delta_t = scale / fs
    integral = np.cumsum(signal * delta_t)
    return Series(integral, name='v_rms')


def calc_pos(signal, scale, fs):
    delta_t = scale / fs
    integral = np.cumsum(0.5 * (signal[1:] + np.diff(signal) / 2.) * (delta_t**2))
    return integral


#: Scipy's trapezoid method
def calc_integral(signal, scale, fs):
    delta_t = scale / fs
    integral = sp.integrate.cumtrapz(signal, dx=delta_t)
    return integral


def calc_derivative(signal, scale, fs):
    N = len(signal)
    derivative = (signal[1:N] - signal[0:N-1]) / (scale/fs)
    return derivative


#: Calculate root-mean-square from a complete signal (e.g. post-measurement)
#: Blends accel dimensions into 1D RMS signal.
#: Returns Series
def calc_rms(df, columns):
    rms = Series(df[columns].apply(lambda x: (x**2)).sum(axis=1).apply(lambda x: np.sqrt(x)), name='a_rms')
    # rms = np.sqrt(np.mean(dat.x**2 + dat.y**2 + dat.z**2))
    return rms


#: Power calculation given complete signal and relevant information
#: Accepts two 1D signals and a scalar
#: 1D accel fed from device/db (or a_rms calculated prior)
#: v_rms calculated from that a_rms/1D accel
#: lift weight (in pounds??)
def calc_power(a_rms, v_rms, weight):
    pwr = weight * a_rms * v_rms
    return Series(pwr, name='p_rms')
