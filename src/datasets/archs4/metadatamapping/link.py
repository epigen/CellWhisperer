import logging

from Bio import Entrez
from typing import Any, Union, Iterable
from . import dbutils
from functools import partial

import pandas as pd
import itertools as it


def has_links(link_item: dict[str, Any]) -> bool:
    """
    checks retrieved items for presence of any links

    :param link_item:   dictionary as returned by Bio.Entrez.read

    :return:            True if key 'LinkSetDb' is not an empty list False otherwise
    """
    return True if link_item['LinkSetDb'] else False


def get_links(link_item: dict[str, Any]) -> list[list[str, str]]:
    """
    parses the given link item and retruns all possible links
    as all combinations of the source id and linked ids

    :param link_item:   dictionary as returned by Bio.Entrez.read

    :return:            list of lists containing the respective links
    """
    # some accessions are not released so we can't link them
    if has_links(link_item):
        dbto_ids = link_item['LinkSetDb'][0]['Link']
        
    else:
        dbto_ids = [{'Id': None}]
        
    dbfrom_ids = link_item['IdList']
    
    links = [
        [dbfrom_id, dbto_id['Id']] for dbfrom_id, dbto_id in it.product(dbfrom_ids, dbto_ids)
    ]
    
    return links
    

def link_uids(uid_list: list[Union[str, int]], dbfrom: str, dbto: str) -> pd.DataFrame:
    """
    takes an iterable of UIDs originating from dbfrom and links them to dbto via eLink

    :param uid_list:    list of integers or strings denoting the UIDs from dbfrom
    :param dbfrom:      string denoting the NCBI database the UIDs stem from
    :param dbto:        string denoting the NCBI database to link the UIDs to

    :return:            pandas.DataFrame with columns dbfrom, dbto containing the linked UIDs
    """
    response_handle = Entrez.elink(
        dbfrom = dbfrom,
        db = dbto,
        id = uid_list
    )

    response = Entrez.read(response_handle)

    linked_ids = []
    for link_item in response:
        links = get_links(link_item)
        linked_ids.extend(links)
    

    linked_ids = pd.DataFrame(
        linked_ids,
        columns = [dbfrom, dbto]
    )
    
    return linked_ids


def link_sra_to_biosample(sra_uids: Iterable[Union[str, int]]) -> pd.DataFrame:
    """
    links SRA UIDs to BioSample UIDs

    :param sra_uids:    iterable containg string or integer versions of SRA UIDs to link

    :return:            pandas.DataFrame containing the 'sra' and 'biosample' column with the respective UIDs
    """
    retry_link = partial(
        link_uids,
        dbfrom = 'sra',
        dbto = 'biosample'
    )
    links = dbutils.process_in_chunks(
        sra_uids,
        retry_link,
        chunksize = 1000
    )
    return pd.concat(links)
    