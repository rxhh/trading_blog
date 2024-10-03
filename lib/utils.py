import datetime as dt
import numpy as np
import pandas as pd

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