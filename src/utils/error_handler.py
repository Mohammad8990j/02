
import time
from functools import wraps
from src.utils.logger import logger

def retry(ExceptionToCheck, tries=3, delay=2, backoff=2):
    def decorator_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except ExceptionToCheck as e:
                    logger.warning(f"خطا: {e}. تلاش مجدد در {_delay} ثانیه...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator_retry
