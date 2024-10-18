import datetime as dt
import numpy as np
import pandas as pd
import tqdm

def between(x, a, b):
    return a <= x <= b

def get_prefixed_cols(_df, pfx):
    return _df.columns[_df.columns.str.startswith(pfx)]

def get_suffixed_cols(_df, sfx):
    return _df.columns[_df.columns.str.endswith(sfx)]

def crossover(a, b):
    """Series a crossing over b
    """
    if type(a) in [float, int] and type(b) in [float, int]:
        print("error, one arg must be array like")
    elif type(a) in [float, int, np.float64]:
        return (a > b) & (a <= b.shift(1))
    elif type(b) in [float, int, np.float64]:
        return (a > b) & (a.shift(1) <= b)
    else:
        return (a > b) & (a.shift(1) <= b.shift(1))
    
def get_triple_barrier(df, ub, lb, tb, use_partial_tb=True):
    """Triple barrier result with partials for time barrier hit
        df: ohlc DataFrame
        ub: pd.Series
        lb: pd.Series
        tb: int
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    o = df['open'].values
    ub = ub.values
    lb = lb.values
    result = []
    for i0 in tqdm.tqdm(range(len(c)-tb-1)):
        i1 = i0+1
        i2 = i0+tb+1
        _h = h[i1:i2]
        _l = l[i1:i2]
        _ub = ub[i0]
        _lb = lb[i0]

        l_crossed = (_l<=_lb).any()
        u_crossed = (_h>=_ub).any()
        if l_crossed and u_crossed: # Both barriers hit, figure out which came first
            i_lc = (_l<=_lb).argmax()
            i_uc = (_h>=_ub).argmax()
            if i_lc == i_uc: # Hit on same bar
                if c[i0+i_lc+1] > o[i0+i_lc+1]: # Close > Open when cross occurs on same bar -> assume low came first
                    result.append(-1)
                else:
                    result.append(1)
            elif i_lc < i_uc:
                result.append(-1)
            else:
                result.append(1)
        elif l_crossed:
            result.append(-1)
        elif u_crossed:
            result.append(1)
        else: # Time barrier hit
            if use_partial_tb:
                c0 = c[i0]
                c2 = c[i2]
                if c2 > c0:
                    result.append((c2-c0)/(_ub-c0))
                elif c2 < c0:
                    result.append((c2-c0)/(c0-_lb))
                else:
                    result.append(0)
            else:
                result.append(0)
    return pd.Series(result+[np.nan]*(tb+1), index=df.index).clip(-1, 1)