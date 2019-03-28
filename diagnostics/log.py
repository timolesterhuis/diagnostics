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

def logged(log=MOD):
    def wrap(function):
        @wraps(function)
        def wrapper(
            *args, **kwargs
        ):  # TODO: shorten args/kwargs when too long (for instance, when containing data)
            logger = logging.getLogger(log)
            logger.debug(
                "Calling function '{}' with args={} kwargs={}".format(
                    function.__name__, args, kwargs
                )
            )
            try:
                response = function(*args, **kwargs)
            except Exception as error:
                logger.debug(
                    "Function '{}' raised {} with error '{}'".format(
                        function.__name__, error.__class__.__name__, str(error)
                    )
                )
                raise error
            logger.debug(
                "Function '{}' returned {}".format(function.__name__, response)
            )
            return response
        return wrapper
    return wrap


modlog = modification_logger
