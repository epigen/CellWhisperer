import time
import urllib
import logging
import http

from Bio import Entrez
from typing import Callable, Any, Union


def get_chunklimits(iterable_length: int, chunksize: int) -> tuple[int, int]:
    """
    takes the length of a given iterable and the size of the chunks we want to create
    and returns tuples of min and max idx for each of these chunks

    :param iterable_length:    length of the iterable to chunkify
    :param chunksize:          size of the iterable chunks

    :yield:                    a tuple of min idx and max idx
    """
    retmin, retmax = 0, 0
    for i in range(0, iterable_length, chunksize):
        retmin = i 
        retmax = i + chunksize
        yield retmin, retmax


def create_webenv(uids: list[Union[int, str]], db: str) -> dict[str, str]:
    """
    posts the given list of uids to the Entrez history server with the given db
    and returns query key and web env id

    :param uids:   list of uids
    :param db:     string denoting the database the UIDs are belonging to

    :return:       dictionary with 'QueryKey' and 'WebEnv' keys containig the query key and web env ID
    """
    response_handle = retry(
        Entrez.epost,
        db = db,
        id = ','.join(str(uid) for uid in uids)
    )
    return Entrez.read(response_handle)


def print_exception_and_increment_try(n_tries: int, exception: Exception, sleep: int = 10) -> int:
    """
    prints the raised exception, sleeps for sleep seconds, increments the trial counter by one and returns it
    
    :param n_tries:     current number of tries
    :param exception:   raised exception object
    :param sleep:       number of seconds to wait before retrying
    
    :return:            current number of tries incremented by one
    """
    logging.warn(exception)
    logging.warn(f'current try: {n_tries}')
    time.sleep(sleep)
    n_tries += 1
    return n_tries


def retry(func: Callable, *args, n_retries: int = 5, sleep: int = 10, **kwargs) -> Any:
    """
    takes a function that might fail with one of the following exceptions: HTTPError, RuntimeError or IncompleteRead
    and retries n_retries times before raising a RuntimeError. Sleeps sleep seconds before retry to resolve possible
    too many request errors with remote APIs
    
    :param func:         function that might fail and needs to be retried
    :param *args:        any positional arguments passed to func
    :param n_retries:    number of retries before raising RuntimeError
    :param sleep:        seconds to sleep before retry
    :param **kwargs:     any keyword arguments passed to func
    
    :return:             any result func might yield
    """
    n_tries = 1
    success = False
    while not success:
        if n_tries == n_retries:
            raise RuntimeError('Exceeded retries!')
            
        try:
            result = func(*args, **kwargs)
            success = True
            
        # this can also be handled in principle via the Entrez.max_tries
        # and Entrez.sleep_between_tries but we leave this as is for now
        # see https://biopython.org/docs/dev/api/Bio.Entrez.html
        except urllib.error.HTTPError as e:
            n_tries = print_exception_and_increment_try(n_tries, e, sleep)
        
        # this exception seems to be raised when the search backend times out
        # due to too many requests I suspect so we retry and hope for the best
        except RuntimeError as e:
            n_tries = print_exception_and_increment_try(n_tries, e, sleep)
        
        # sometimes happens when reading the response if so we just retry
        except http.client.IncompleteRead as e:
            n_tries = print_exception_and_increment_try(n_tries, e, sleep)
            
        
    return result
