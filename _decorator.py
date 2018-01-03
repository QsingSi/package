from ._package import wraps
from ._package import time

__all__ = ['log', 'runtime_log']


def log(func):
    @wraps(func)
    def add_start_end(*args, **kw):
        print('Start running %s' % func.__name__)
        res = func(*args, **kw)
        print('End %s' % func.__name__)
        return res
    return add_start_end


def runtime_log(func):
    @wraps(func)
    def get_runtime(*args, **kw):
        start = time.time()
        res = func(*args, **kw)
        end = time.time()
        print('%s runs %.2f seconds' % (func.__name__, end - start))
        return res
    return get_runtime
