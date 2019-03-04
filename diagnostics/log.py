import os
import logging
from logging.handlers import TimedRotatingFileHandler
from functools import wraps

MOD = "modification"
PWD = os.getcwd()

modification_logger = logging.getLogger(MOD)
modification_logger.addHandler(logging.NullHandler())  # TODO: think whether it should be nullhandler by default
modification_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(message)s")

file_handler = TimedRotatingFileHandler(os.path.join(PWD, "modifications.log"), 
                                        when='S', interval=1, backupCount=0, 
                                        encoding=None, delay=False, utc=False)

file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
modification_logger.addHandler(file_handler)



def logged(log=MOD):
    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):  # TODO: shorten args/kwargs when too long (for instance, when containing data)
            logger = logging.getLogger(log)
            logger.debug("Calling function '{}' with args={} kwargs={}"
                             .format(function.__name__, args, kwargs))
            try:
                response = function(*args, **kwargs)
            except Exception as error:
                logger.debug("Function '{}' raised {} with error '{}'"
                                 .format(function.__name__,
                                         error.__class__.__name__,
                                         str(error)))
                raise error
            logger.debug("Function '{}' returned {}"
                             .format(function.__name__,
                                     response))
            return response
        return wrapper
    return wrap


modlog = modification_logger