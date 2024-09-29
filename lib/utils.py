import datetime as dt
import numpy as np
import pandas as pd

def between(x, a, b):
    return a <= x <= b

def get_prefixed_cols(_df, pfx):
    return _df.columns[_df.columns.str.startswith(pfx)]

def get_suffixed_cols(_df, sfx):
    return _df.columns[_df.columns.str.endswith(sfx)]