import signal
from typing import Callable
import os
from os.path import join as joinpath


def all_exists(*args):
    return all(e is not None for e in args)


def any_exists(*args):
    return any(e is not None for e in args)


class FuncTimeOutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise FuncTimeOutError("Function timed out")


def has_same_obj_in_list(obj, ls):
    return any(obj is e for e in ls)


def wrap_function_with_timeout(func: Callable, timeout: int):
    def wrapped_function(*args, **kwargs):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Reset the alarm
        except FuncTimeOutError:
            return None
        return result

    return wrapped_function

def make_parent_dirs(fp:str):
    parts = fp.split('/')
    if len(parts) == 1:
        return

    parent_dir = joinpath(*parts[:-1])
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)