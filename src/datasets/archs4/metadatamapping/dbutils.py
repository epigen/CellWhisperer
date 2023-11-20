import time
import urllib
import os
import logging
import re
import http

from Bio import Entrez
from functools import partial
from typing import Callable, Any, Union, Iterable
from os import PathLike

import multiprocessing as mp
import pandas as pd

from . import concurrency


logging.basicConfig(
    format = '%(threadName)s: %(asctime)s-%(levelname)s-%(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    level = logging.INFO
)


accession_matchers = {
    'experiment': re.compile('Experiment acc="([SRX0-9]+)'),
    'sample': re.compile('Sample acc="(SRS[0-9]+)'),
    'study': re.compile('Study acc="(SRP[0-9]+)'),
    'biosample': re.compile('Biosample>(SAMN[0-9]+)'),
    'bioproject': re.compile('Bioproject>(PRJNA[0-9]+)'),
    'geo': re.compile('name="(GSM[0-9]+)')
}


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
    response_handle = Entrez.epost(
        db = db,
        id = ','.join(str(uid) for uid in uids)
    )
    return Entrez.read(response_handle)


def uid_from_esearch(accession: str, db: str) -> list[list[str]]:
    """
    uses Bio.Entrez.esearch to request the database internal UID of the given accession
    see  https://www.ncbi.nlm.nih.gov/books/NBK25497/table/chapter2.T._entrez_unique_identifiers_ui/?report=objectonly
    for available databases
    
    :param accession:  accession to retrieve UID for
    :param db:         name of the Entrez database to retrieve the UID from
    
    :return:           list of lists of accession, uid and database name the uid was retrieved from
    """
    response_handle = Entrez.esearch(db = db, term = accession)
    uid_list = Entrez.read(response_handle)['IdList']
    return [[accession, uid, db] for uid in uid_list]
        
    
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
        
        
def read_retrieved_accessions(filename: Union[PathLike, str]) -> set[str]:
    """
    reads the file given with filename and return a list of unique accessions
    the file has to contain a table with an 'accession' column
    
    :param filename:   path to the accession table to read
    
    :return:           set of unique accessions
    """
    accession_map = pd.read_csv(
        filename,
        sep = '\t'
    )
    return set(accession_map.accession)


def map_accessions_to_uids(
    accessions: Iterable[pd.Series], 
    db: str, 
    outfilename: Union[PathLike, str], 
    filelock: Union[mp.Manager().Lock, None] = None
) -> None:
    """
    takes a list of pandas.Series object generated by pandas.DataFrame.iterrows and a path to an outputfile
    and uses uid_from_esearch to map the contained accessions to the SRA internal UID. If the function is used
    concurrently a Lock object must be passed via the filelock argument
    
    :param accessions:     list of pandas.Series objects containing the keys srx_accession, biosample_accession and geo_accession
    :param db:             string denoting the NCBI database to retrieve the UIDs from
    :param outfilename:    path to the file to write the retrieved UID mappings to
    :param filelock:       A Lock object used to savely write to the outfile in case of concurrent usage
    
    :return:               None
    """
    logging.info('starting mapping process')
    write_to_file = (
        concurrency.write_to_file_multiprocess if filelock 
        else concurrency.write_to_file_singleprocess
    )
    
    uid_list = []
    start = accessions[0][0]
    for i, row in accessions:
        srx = row.srx_accession
        samn = row.biosample_accession
        gsm = row.geo_accession
        
        if not any([srx, samn]):
            accession = gsm

        else:
            accession = srx if srx else samn
            
        mapped_uids = retry(
            uid_from_esearch,
            accession, 
            db
        )
        
        uid_list.extend(mapped_uids)
        
        if not (i + 1) % 1000:
            write_to_file(
                outfilename,
                uid_list,
                filelock
            )
            uid_list = []
            logging.info(f'Written uids for accessions {start} to {i}')
    
    
    logging.info(f'finishing up. writing {len(uid_list)} remaining uid mappings')

    # write remaining uids if there are any
    if uid_list:
        write_to_file(
            outfilename,
            uid_list,
            filelock
        )


def get_not_yet_mapped(table: pd.DataFrame, outfilename: Union[PathLike, str]) -> pd.DataFrame:
    """
    If outfilename does not exist yet it is created and table is returned as is. If outfilename exists
    accessions contained in it are already mapped and thus will be removed from table before returning it
    
    :param table:         pandas.DataFrame containing the accessions we want to retrieve
    :param outfilename:   path to a file the mapping output should be or is written to
    
    :return:              pandas.DataFrame containing only the accessions that are not already found in outfilename
    """
    # this can definitely be refactored but I leave it as is for now
    if not os.path.exists(outfilename):
        retrieved_accessions = set()
        with open(outfilename, 'w') as outfile:
            outfile.write(
                'accession\tuid\tdatabase\n'
            )
            
        return table

    else:
        retrieved_accessions = read_retrieved_accessions(outfilename)
        retrieved = table.apply(
            lambda x: any(
                x[acc] in retrieved_accessions 
                for acc in ['geo_accession', 'srx_accession', 'biosample_accession']
            ),
            axis = 1
        )
        return table.loc[~retrieved, :]
    

def extract_accessions(
    summary: dict[str, Any], 
    accession_summary: dict[str, list], 
    accession_matchers: dict[str, re.Pattern]
) -> None:
    """
    parses the accessions specified in accession_matchers and adds them to the accession_summary dictionary in place

    :param summay:                  summary item returned by Bio.Entrez.esummary
    :param accession_summary:       dictionary containing the same keys ad accession_matchers and lists as values
    :param accession_matchers:      dictionary containing re.Pattern as values

    :return:                        None
    """
    for accession_type, accession_matcher in accession_matchers.items():
        match = accession_matcher.search(summary['ExpXml'])
        accession = match.groups()[0] if match else None
        accession_summary[accession_type].append(accession)


def accessions_from_esummary_response(
    summary_items: list[dict[str, Any]], 
    accessions: dict[str, list], 
    accession_matchers: dict[str, re.Pattern]
) -> None:
    """
    parses the summary items from eSummary

    :param summary_items:           list of summary items as returned by Bio.Entrez.esummary
    :param accession_summary:       dictionary containing the same keys ad accession_matchers and lists as values
    :param accession_matchers:      dictionary containing re.Pattern as values

    :return:                        None
    """
    for summary in esummary_items:
        summary_accessions = extract_accessions(
            summary,
            accessions,
            accession_matchers
        )


def data_from_esummary(
    web_env_info: dict[str, str], 
    db: str, 
    parse_function: Callable, 
    data_parsers: dict[str, Any], 
    n_uids: int, 
    chunksize: int = 5000
) -> pd.DataFrame:
    """
    retrieves all the data from eSummary associated with the UIDs posted to the Entrez history server

    :param web_env_info:        dictionary containing the keys 'QueryKey' and 'WebEnv' as returned by Entrez.epost
    :param db:                  string denoting the database the supplied UIDs are from
    :param parse_function:      function used to parse the esummary response
    :param data_parsers:        dictionary containing keys and whatever you parser needs to parse the response 
                                keys will be the columns of the returned dataframe
    :param n_uids:              number of UIDs posted to the history server
    :param chunksize:           size of the chunks of summaries to retrieve

    :return:                    pandas.DataFrame with columns equal to data_parsers keys and the parsed data as values
    """
    esummary = partial(
        Entrez.esummary,
        db = db,
        WebEnv = web_env_info['WebEnv'],
        query_key = web_env_info['QueryKey']
    )

    def esummaries_from_history(retstart, retmax):
        response_handle = esummary(
            retstart = retstart, 
            retmax = retmax
        )
        return Entrez.read(response_handle)
    
    data = {
        key.lower(): [] for key in data_parsers.keys()
    }
    
    for retmin, retmax in get_chunklimits(n_summaries, chunksize):
        time.sleep(5) # avoid too many requests error
        response = retry(
            esummaries_from_history,
            retstart = retmin,
            retmax = retmax
        )
        parse_function(
            response,
            data,
            data_parsers
        )
        
    data = pd.DataFrame().from_dict(
        data,
        orient = 'columns'
    )
    
    # somehow we get duplicates, suspect the post call
    # because I did it twice so we might end up with 
    # duplicated uids in the webenv but not sure
    # anyway dropping duplicates here
    return data.drop_duplicates().reset_index(drop = True)


def summaries_from_uids(
    uids: Iterable[Union[int, str]], 
    db: str,
    parse_function: Callable, 
    data_parsers: dict[str, Any],
    chunksize: int = 50000
) -> pd.DataFrame:
    """
    retrieves summaries for uids from eSummary and returns a pandas.DataFrame with columns = keys of data_parsers
    Values in these columns are determined by parse_function

    :param uids:                iterable of uids to retrieve eSummaries for
    :param db:                  database to retrieve the summaries from
    :param parse_function:      function used to parse the summaries
    :param data_parsers:        parser dictionary forwarded to parse function
    :param chunksize:           size of the chunks of uids posted to the Entrez history server

    :return:                    pandas.DataFrame containing the summary parsing results
    """
    uid_batches = it.batched(
        uids,
        n = chunksize
    )

    n = 0
    result_frames = []
    for uid_list in uid_batches:
        logging.info(f'retrieving uids {n} to {n + len(uid_list)}')

        n += len(uid_list)
        web_env_info = create_webenv(uid_list, db)

        logging.debug(web_env_info)
        
        result_frame = data_from_esummary(
            web_env_info,
            db,
            parse_function,
            len(uid_list),
            data_parsers
        )

        result_frames.append(result_frames)

    return pd.concat(result_frames)
