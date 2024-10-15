import math
from math import sin, cos, exp
import numpy as np
import pandas as pd
from scipy import signal

RT2 = math.sqrt(2)
PI = math.pi

def SuperSmoother(src, period):
    """Ehlers SuperSmoother
    """
    a = exp(-RT2*PI/period)
    b = 2*a*cos(RT2*PI/period)
    c = a*a
    
    c2 = b
    c3 = -a*a
    c1 = 1 - c2 - c3

    s = np.pad(np.array(src), (period*2, 0), 'edge')
    s = signal.lfilter(b=[c1], a=[1, -c2, -c3], x=s)

    return pd.Series(index=src.index, data=s[period*2:])

def Decycler(src, cutoff):
    """ Ehlers decycler
    AKA a low pass filter
    """
    a1 = (cos(PI/cutoff)+sin(PI/cutoff)-1) / cos(PI/cutoff)
    
    s = np.pad(np.array(src), (cutoff*2, 0), 'edge')    
    s = signal.lfilter(b=[a1/2, a1/2], a=[1, -(1-a1)], x=s)

    return pd.Series(index=src.index, data=s[cutoff*2:])

def Highpass2Pole(src, cutoff):
    """ Ehlers 2-pole high pass filter, for use in a decycler oscillator
    """
    a = (cos(PI/RT2/cutoff) + sin(PI/RT2/cutoff) - 1) / cos(PI/RT2/cutoff)
    b = 1 - a/2
    # return pd.Series(index=src.index, data=signal.lfilter(b=[b*b, -2*b*b, b*b], a=[1, -2*(1-a), (1-a)*(1-a)], x=src))

    s = np.pad(np.array(src), (cutoff*2, 0), 'edge')    
    s = signal.lfilter(b=[b*b, -2*b*b, b*b], a=[1, -2*(1-a), (1-a)*(1-a)], x=s)

    return pd.Series(index=src.index, data=s[cutoff*2:])


def GaussianFilter(src, period, npoles):
    B = (1-math.cos(2*math.pi/period)) / math.pow(2, 1/npoles-1)
    A = -B + math.sqrt(math.pow(B,2)+2*B)
    
    _src = np.pad(np.array(src), (period*2, 0), 'edge')
    result = list(_src[:4])

    if npoles==1:
        for i in range(4, len(_src)):
            result.append(A*_src[i] + (1-A)*(result[i-1]))
    elif npoles==2:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 2)*_src[i] + 2*(1-A)*(result[i-1]) - math.pow(1-A, 2)*result[i-2])
    elif npoles==3:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 3)*_src[i] + 3*(1-A)*(result[i-1]) - 3*math.pow(1-A, 2)*result[i-2] + math.pow(1-A, 3)*result[i-3])
    elif npoles==4:
        for i in range(4, len(_src)):
            result.append(math.pow(A, 4)*_src[i] + 4*(1-A)*(result[i-1]) - 6*math.pow(1-A, 2)*result[i-2] + 4*math.pow(1-A, 3)*result[i-3] -math.pow(1-A, 4)*result[i-4])

    return pd.Series(index=src.index, data=result[period*2:])

def DecyclerOscillator(src, cutoff1, cutoff2):
    """ Ehlers decycler oscillator
    Difference between 2 high pass filters with cutoff1 < cutoff2
    """
    return Highpass2Pole(src, cutoff2) - Highpass2Pole(src, cutoff1)

def Roof(src, cutoff_low=10, cutoff_high=48):
    """ Ehlers roofing filter
    """
    hp = Highpass2Pole(src, cutoff_high)
    return SuperSmoother(hp, cutoff_low)

def SuperPassband(src, fast=40, slow=60):
    """ Ehlers Super Passband Filter
    """
    a1 = 5/fast
    a2 = 5/slow

    # espf = signal.lfilter(b=[(a1-a2), a2*(1-a1)-a1*(1-a2)], a=[1, -(2-a1-a2), (1-a1)*(1-a2)], x=src)
    # return pd.Series(index=src.index, data=espf)

    s = np.pad(np.array(src), (slow*2, 0), 'edge')
    s = signal.lfilter(b=[(a1-a2), a2*(1-a1)-a1*(1-a2)], a=[1, -(2-a1-a2), (1-a1)*(1-a2)], x=s)
    return pd.Series(index=src.index, data=s[slow*2:])


def Bandpass(src, period=10, bandwidth=0.3):
    """ Ehlers bandpass filter
    """

    a2 = (cos(0.5*bandwidth*PI/period) + sin(0.5*bandwidth*PI/period) - 1)/cos(0.5*bandwidth*PI/period)
    hp = signal.lfilter(b=[(1+a2/2), -(1+a2/2)], a=[1, -(1-a2)], x=src)

    b1 = cos(2*PI/period)
    g1 = 1/cos(2*PI*bandwidth/period)
    a1 = g1 - math.sqrt(g1*g1-1)

    bp = signal.lfilter(b=[0.5*(1-a1), 0, -0.5*(1-a1)], a=[1, -b1*(1+a1), a1], x=hp)

    return pd.Series(index=src.index, data=bp)

def EarlyOnsetTrend(src, lpperiod):
    roof = Roof(src, lpperiod, 100)
    x = np.array(li.agc(roof))
    k1 = 0.85
    k2 = 0.4
    q1 = (x+k1)/(k1*x+1)
    q2 = (x+k2)/(k2*x+1)
    return q1, q2

def TrendFlex(s, p=20):
    ssmooth = SuperSmoother(s, p)
    diffs = ssmooth.diff().rolling(p).mean()
    result = diffs/np.sqrt(diffs.pow(2).ewm(span=p).mean())
    return result

def Reflex(s, p=20):
    ssmooth = SuperSmoother(s, p)
    slope = -ssmooth.diff(p)/p
    E = np.array([ssmooth.diff(_p).values for _p in np.arange(p)+1]).sum(axis=0) + (np.arange(p)+1).sum()*slope
    E = E/p    
    result = E/np.sqrt(E.pow(2).ewm(span=p).mean())
    return result

def RMS(src, length):
    """ Root mean square
    """
    return pd.Series(index=src.index, data=np.sqrt(pd.Series(src*src).rolling(length, min_periods=1).sum().values/length))


def AutocorrelationPeriodogram(src, avg_len, min_len, max_len):
    """ Ehlers autocorrelation periodogram
    Calculates dominant cycle and power spectrum
    Returns (dom_cycle, pwr)
    """
    src = np.array(src)
    pwr = []
    dom_cycle = []
    r_last = None
    for i, x in enumerate(src):
        # Compute autocorrelations
        corrs = []
        for lag in range(max_len+1):
            if i < avg_len:
                corrs.append(0)
            Sx = 0
            Sy = 0
            Sxx = 0
            Syy = 0
            Sxy = 0
            for count in range (0, avg_len):
                X = src[i-count]
                Y = src[i-(lag+count)]
                Sx = Sx + X
                Sy = Sy + Y
                Sxx = Sxx + X*X
                Sxy = Sxy + X*Y
                Syy = Syy + Y*Y
            if (avg_len*Sxx - Sx*Sx)*(avg_len*Syy - Sy*Sy) > 0:
                corrs.append((avg_len*Sxy - Sx*Sy)/math.sqrt((avg_len*Sxx - Sx*Sx)*(avg_len*Syy - Sy*Sy)))
            else:
                corrs.append(0)

        # FT power spectrum for each correlation
        r = []
        for period in range(min_len, max_len+1):
            _cos_part = np.sum([corrs[N]*cos(2*PI*N/period) for N in range(avg_len, max_len+1)])
            _sin_part = np.sum([corrs[N]*sin(2*PI*N/period) for N in range(avg_len, max_len+1)])
            _sq_sum = _cos_part*_cos_part + _sin_part*_sin_part
            r.append(_sq_sum*_sq_sum)
        r = np.append(np.zeros(min_len), np.array(r))

        # EMA smoothing on power
        if r_last is not None:
            r = 0.2*r + 0.8*r_last
        r_last = r
        
        # Minmax norm
        max_pwr = r.max()
        _pwr = r/max_pwr
        _pwr_i = np.arange(len(_pwr)) + 1
        
        # CG
        _mask = _pwr > 0.5
        _sp = _pwr[_mask]
        _spx = np.dot(_pwr_i[_mask], _sp)
        _sp = _sp.sum()
        _dom_cycle = _spx/_sp

        pwr.append(_pwr)
        dom_cycle.append(_dom_cycle)
    return dom_cycle, pwr