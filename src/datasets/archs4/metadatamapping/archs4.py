import h5py
import gc
import os

import pandas as pd
import numpy as np
import itertools as it

from os import PathLike
from . import concurrency
from . import dbutils
from typing import Union, Iterable, Any


def parse_table(archs4_file: Union[str, PathLike], root_key: str, table_key: str) -> dict[str, Any]:
    """
    parses the archs4 h5 file and extracts the data at /root_key/table_key into a dictionary
    see https://maayanlab.cloud/archs4/help.html for more infos on the available options of
    root_key and table_key
    
    :param archs4_file:   path to the archs4 h5 file to parse
    :param root_key:      root key to use for parsing the file metadata
    :param table_key:     table key to use for parsing the file metadata
    
    :return:              dictionary containing the metadata with column names as keys and column contents as values
    """
    archs4 = h5py.File(archs4_file, 'r')
    
    data = dict()
    table = archs4[root_key][table_key]
    for key in table.keys():
        column_data_piece = table[key][0]
        if isinstance(column_data_piece, np.number):
            column_data = table[key][:]
        
        else:
            column_data = table[key].asstr()[:]
        
        data[key] = column_data
    
    return data


def filter_table_data(data: dict, retain_keys: Iterable[str]) -> dict[str, Any]:
    """
    filters the data dictionary to retain only the keys specified by retain_keys
    
    :param data:          dictionary as returned by parse_table
    :param retain_keys:   list of data keys to retain
    
    :return:              filtered data dictionary
    """
    retain_data = {
        key: data[key].copy() for key in retain_keys
    }
    return retain_data


def get_srx_samn_from_items(relation_items: Iterable[str]) -> tuple[str, str]:
    """
    takes list of strings generated from an item in the 'relation' column
    of the /meta/samples table in the archs4 h5 file and extracts the SRA and
    BioSample accessions from it
    
    :param relation_items:   list of strings of the format 'key: value'
    
    :return:                 sra and bisample accession as strings
    """
    srx, samn = None, None
    for item in relation_items:
        if not item:
            continue
            
        key, value = item.split(': ')
            
        if key == 'SRA':
            srx = value.split('=')[-1]
        
        if key == 'BioSample':
            samn = value.split('/')[-1]
        
    return srx, samn


def extract_srx_and_samn_accessions(relation_data: Iterable[str]) -> dict[str, list[str]]:
    """
    takes the full 'relation' column of the archs4 h5 /meta/samples table and extracts
    SRA and BioSample accessions for each of the items contained
    
    :param relation_data:  list of strings of the format 'key1: value1, key2: value2, ...'
    
    :return:               dictionary containing keys srx_accession, biosample_accession with lists of these accessions as values
    """
    srx_samn_accessions = {
        key: [] for key in ['srx_accession', 'biosample_accession']
    }
    for relation in relation_data:
        relation_items = relation.split(',')
        srx, samn = get_srx_samn_from_items(
            relation_items
        )
        srx_samn_accessions['srx_accession'].append(srx)
        srx_samn_accessions['biosample_accession'].append(samn)
    
    return srx_samn_accessions


def get_filtered_sample_metadata(archs4_file: Union[str, PathLike], keys_to_retain: Iterable[str]) -> pd.DataFrame:
    """
    parses the archs4 h5 file and returns a pandas.DataFrame of with all columns of the /meta/samples table 
    specified in keys_to_retain plus extracted SRA and BioSamples accessions (columns srx_accession and biosample_accession)
    
    :param archs4_file:       path to the archs4 file to parse
    :param keys_to_retain:    list of keys in the /meta/samples table to retain
    
    :return:                  pandas.DataFrame containing the respective metadata + SRA and BioSample accessions
    """
    data = parse_table(
        archs4_file,
        'meta',
        'samples'
    )

    filtered_data = filter_table_data(
        data, 
        keys_to_retain
    )

    srx_samn_accessions = extract_srx_and_samn_accessions(
        filtered_data['relation']
    )

    srx_samn_table = pd.DataFrame().from_dict(
        srx_samn_accessions,
        orient = 'columns'
    )

    table = pd.DataFrame().from_dict(
        filtered_data, 
        orient = 'columns'
    )

    table = pd.concat(
        [table, srx_samn_table],
        axis = 1
    )

    del data, filtered_data, srx_samn_accessions
    gc.collect()
    
    return table


def map_accessions_to_srauids(table: pd.DataFrame, outfilename: Union[PathLike, str], chunksize: int = 5000, n_processes: int = 1) -> None:
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
    mapping_table = dbutils.get_not_yet_mapped(
        table,
        outfilename
    )
    
    concurrency.process_data_in_chunks(
        mapping_table.iterrows(),
        dbutils.map_accessions_to_uids,
        db = 'sra',
        outfilename = outfilename
    )
