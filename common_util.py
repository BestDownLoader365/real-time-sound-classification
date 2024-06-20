import os
from datetime import datetime
import logging as log
import sys
import functools
import time
import psutil

TODAY = datetime.today().strftime('%Y-%m-%d')
TODAY_TIME = datetime.today().strftime('%H_%M_%S')
LOG_FILE_NAME = f'./sound_classification_data/{TODAY}'

os.makedirs(LOG_FILE_NAME, exist_ok=True)
logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
formatter = log.Formatter('[%(asctime)s]\n[ %(levelname)s ]: %(message)s')

log_handler = log.FileHandler(os.path.join(LOG_FILE_NAME, f'{TODAY_TIME}.log'))
log_handler.setFormatter(formatter)

# logger.addHandler(log_handler)
stream_handler = log.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def performance_analyzer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        logger.info(
            f"Elapsed time for function \"{func.__name__}\" is {round((time.perf_counter() - start_time) * 1e3, 2)} ms")
        logger.info(
            f"CPU Utilization when running function \"{func.__name__}\" is {psutil.cpu_percent()}%")
        return value
    return wrapper
