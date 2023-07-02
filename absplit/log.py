import logging
import os
import sys


def get_logger(name):
    dir = os.path.dirname(__file__)
    log_flag = os.path.join(dir, 'log_flag')
    log_level = logging.CRITICAL
    if os.path.exists(log_flag):
        log_level = logging.DEBUG

    # Create a logger
    logger = logging.getLogger(name)

    # Set the default log level. This can be overridden in individual modules
    logger.setLevel(log_level)

    # Create a console handler
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    stdout_hdlr.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Add formatter to console handler
    stdout_hdlr.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Add console handler to logger
    logger.addHandler(stdout_hdlr)

    return logger