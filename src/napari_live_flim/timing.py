import logging
import time
from functools import wraps
from typing import Optional, Callable

import numpy as np


# adapted from stackoverflow.com :)
def timing(func: Optional[Callable]=None, name=None):
    def _out_func(f):
        fname = func.__name__ if name is None else name
        arr = []
        @wraps(f)
        def _func(*args, **kw):
            nonlocal arr, fname
            ts = time.perf_counter()
            result = f(*args, **kw)
            t = (time.perf_counter() - ts) * 1000
            arr += [t]
            logging.info(f"Function {fname} took {t} milliseconds. Median {np.median(arr)}")
            return result
        return _func

    if func is None:
        return _out_func
    return _out_func(func)
