import logging
import certifi
import urllib3

import pandas as pd

from . import parsers
from . import summary
from . import concurrency
from . import dbutils
from typing import Union, Iterable
from os import PathLike
from io import BytesIO

logging.basicConfig(
    format = '%(threadName)s: %(asctime)s-%(levelname)s-%(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    level = logging.INFO
)

def map_accessions_to_srauids(
    table: pd.DataFrame, 
    outfilename: Union[PathLike, str], 
    chunksize: int = 5000, 
    n_processes: int = 1
) -> None:
    """
    takes a pandas.DataFrame containing columns srx_accession, biosample_accession and geo_accession, maps those accessions
    to SRA UIDs and writes the resulting map to outfilename. If n_processes > 1 this will be done concurrently
    
    :param table:         pandas.DataFrame containing columns srx_accession, biosample_accession and geo_accession
    :param db:            string denoting the NCBI database to retrieve the UIDs from
    :param outfilename:   path to the outputfile the resulting map should be written to
    :param chunksize:     size of the indiviually processed chunks of the input table
    :param n_processes:   number of processes to use for mapping if n_processes > 1 this will be done concurrently using multiprocessing.imap
    
    :return:              None
    """
    mapping_table = parsers.get_not_yet_mapped(
        table,
        outfilename
    )
    
    concurrency.process_data_in_chunks(
        mapping_table.iterrows(),
        dbutils.map_accessions_to_uids,
        db = 'sra',
        outfilename = outfilename
    )


def determine_accession_type(accession: str) -> str:
    """
    determines the type of accession used for retrieving the UIDs
    currently identifies SRX SAMN or GSM accessions

    :param accession:   string containing the accession to be identified

    :return:            string giving the type of accession
    """
    types = {
        'experiment': 'SRX',
        'biosample': 'SAMN',
        'geo': 'GSM'
    }
    
    for accession_type, accession_prefix in types.items():
        if accession.startswith(accession_prefix):
            return accession_type
        

def merge_uids_and_accessions(uids: pd.DataFrame, accessions: pd.DataFrame) -> pd.DataFrame:
    """
    merges the uid frame with the retrieved accessions

    :param uids:            pandas.DataFrame containing 'uid' and 'accession' columns
    :param accessions:      pandas.DataFrame containing accessions correspondig to uids

    :return:                pandas.DataFrame merged on uid
    """
    uids = uids.copy()
    uids['accession_type'] = uids.accession.apply(
        determine_accession_type
    )

    merged_groups = []
    for accession_type, group in uids.groupby('accession_type'):
        group = group.rename(
            columns = {'accession': accession_type}
        )
        
        merged_group = group.merge(
            accessions,
            on = accession_type,
            how = 'inner'
        )

        merged_groups.append(merged_group)
        
    return pd.concat(merged_groups)


def srauids_to_accessions(srauids: pd.DataFrame, chunksize: int = 50000) -> pd.DataFrame:
    """
    retrieves all available accessions for the sra uids given in 'uid' column of srauids

    :param srauids:     pandas.DataFrame containing an 'accession' and a 'uid' column
    :param chunksize:   integer giving the size of the chunks posted and retrieved from Entrez

    :return:            pandas.DataFrame containing experiment, sample, biosample, study, bioproject and geo accessions
    """
    accessions = summary.summaries_from_uids(
        srauids.uid,
        'sra',
        parsers.accessions_from_esummary_response,
        parsers.accession_matchers,
        chunksize = chunksize
    )
    
    return merge_uids_and_accessions(srauids, accessions)


def biosample_uids_to_metadata(biosample_uids: Iterable[Union[str, int]], chunksize: int = 50000) -> pd.DataFrame:
    """
    retrieves all available metadata for the biosample uids

    :param biosample_uids:  iterable of biosample UIDs as string or int
    :param chunksize:       integer giving the size of the chunks posted and retrieved from Entrez

    :return:            pandas.DataFrame containing experiment, sample, biosample, study, bioproject and geo accessions
    """
    metadata = summary.summaries_from_uids(
        biosample_uids,
        'biosample',
        parsers.metadata_from_biosample_uids,
        parsers.biosample_metadata_parsers,
        chunksize = chunksize
    )
    
    return metadata


def study_id_to_metasra(study_ids: Iterable[str]) -> pd.DataFrame:
    """
    used the MetaSRA API to retrieve normalized metdata for all samples in 
    a study given by study_ids from their database
    
    :param study_ids:   iterable containing SRA Biostudy accessions

    :return:            pandas.DataFrame containing MetaSRA normalised metadata
    """
    # urllib complains about CERT_NONE but MetaSRA fails with cert so we disable the warning
    urllib3.disable_warnings()
    metasra_data = []
    for i, study_id in enumerate(study_ids):
        # cert_reqs = 'CERT_NONE' not recommended but needed due to unresolvable SSLError
        # http = urllib3.PoolManager(
        #     ca_certs = certifi.where(),
        #     cert_reqs = 'CERT_NONE'                      
        # )

        # retrieving number of studies that match search criteria
        r = urllib3.request(
            'GET', 
            'https://metasra.biostat.wisc.edu/api/v01/samples.csv',
            fields = {
                'study': study_id,
                'species': 'human', 
                'assay': 'RNA-seq', 
                'limit': 10000,
                'skip': 0
            }
        )

        study_data = pd.read_csv(BytesIO(r.data))
        
        if not study_data.empty:
            metasra_data.append(study_data)
    
    return pd.concat(metasra_data)


def metasra_from_study_id(study_ids: Iterable[str]) -> pd.DataFrame:
    """
    used the MetaSRA API to retrieve normalized metdata for all samples in 
    a study given by study_ids from their database
    
    :param study_ids:   iterable containing SRA Biostudy accessions

    :return:            pandas.DataFrame containing MetaSRA normalised metadata
    """
    metasra_data = dbutils.process_in_chunks(
        study_ids,
        study_id_to_metasra,
        chunksize = 1000
    )

    return pd.concat(metasra_data)


# def merge_retrieved_infos():
