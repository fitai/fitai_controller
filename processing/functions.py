from numpy import std, cumsum, diff, sum, sqrt, insert
from scipy.integrate import cumtrapz
from pandas import Series


#: Calculate standard deviation of signal, starting from index calc_offset
#: Multiply that standard deviation by a scalar value - this is what changes with the learning
#: Take any values above threshold, subtract that from itself; creates an impulse signal
#: Reps = # positive crossings, or # positive impulses
def method1(signal, calc_offset=0, scale=3.):
    sigma = std(signal[calc_offset:])
    thresh = scale * sigma
    mask = (signal > thresh) * 1  # Convert boolean to binary
    imp = diff(mask)
    reps = sum(imp > 0)
    return imp, thresh, reps


#: Trapezoid method
def calc_vel1(signal, scale, fs):
    delta_t = scale / fs
    integral = cumsum((signal[1:] + diff(signal) / 2.) * delta_t)
    return integral


#: Euler method
#: Accepts single dimension of values (e.g. series or array)
#: Returns pandas Series
def calc_integral(signal, scale=1., fs=20.):
    delta_t = scale / fs
    integral = cumsum(signal * delta_t)
    return Series(data=integral, name='integral')


def calc_pos(signal, scale=1., fs=20.):
    delta_t = scale / fs
    integral = cumsum(0.5 * (signal[1:] + diff(signal) / 2.) * (delta_t**2))
    integral[0] = 0.
    # integral = insert(integral, 0, [0])  # because of the use of diff() above, the first value as cut off. insert it here
    return integral


#: Scipy's trapezoid method
def calc_integral_sp(signal, scale, fs):
    delta_t = scale / fs
    integral = cumtrapz(signal, dx=delta_t)

    return integral


def calc_derivative(signal, scale, fs):
    N = len(signal)
    derivative = (signal[1:N] - signal[0:N-1]) / (scale/fs)
    return derivative


#: Calculate root-mean-square from a complete signal (e.g. post-measurement)
#: Blends accel dimensions into 1D RMS signal.
#: Returns Series
def calc_rms(df, columns):
    rms = Series(df[columns].apply(lambda x: (x**2)).sum(axis=1).apply(lambda x: sqrt(x)), name='rms')
    return rms


#: Power calculation given complete signal and relevant information
#: Accepts two 1D signals and a scalar
#: 1D accel fed from device/db (or a_rms calculated prior)
#: v_rms calculated from that a_rms/1D accel
#: lift weight (in pounds??)
def calc_power(a, v, weight):
    pwr = weight * a * v
    return Series(pwr, name='power')


def calc_force(a, weight):
    force = a * weight  # weight is in kg
    return Series(force, name='force')
