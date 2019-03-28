import os
import logging
from functools import wraps

MOD = "modification"
PWD = os.getcwd()

modification_logger = logging.getLogger(MOD)
modification_logger.addHandler(
    logging.NullHandler()
)  # TODO: think whether it should be nullhandler by default
modification_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(message)s")

def logged():
    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            response = function(*args, **kwargs)
            return response
        return wrapper
    return wrap


modlog = modification_logger
