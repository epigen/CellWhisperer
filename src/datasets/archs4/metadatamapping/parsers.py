import re

import pandas as pd

from xml.etree import ElementTree
from typing import Any, Union, Callable
from os import PathLike


def node_parser(node: ElementTree.Element, parse_func: Callable, match_string: str) -> tuple[bool, dict[str, str]]:
    """
    parses an xml node if with parse_func if node.tag == match_sting
    and returns True and a dictionary with the nodes attributes
    else returns False and an empty dictionary

    :param node:            ElementTree.Element instance
    :param parse_func:      function taking the node and parses out its information
    :param match_string:    string representing the nodes tag that needs to match for it to be parsed

    :return:                True, dictionary with parsed infos if match_string == node.tag else False, empty dictinary   
    """
    match = False
    if not node.tag == match_string:
        return match, dict()
    
    match = True
    return match, parse_func(node)


def parse_title(node: ElementTree.Element) -> dict[str, str]:
    """
    returns the the text attribute of node if node.tag == 'title'

    :param node:    ElementTree.Element instance

    :return:        dictionary containing the title
    """
    return {'title': node.text}
        
    
def parse_attribute(node: ElementTree.Element) -> dict[str, str]:
    """
    returns the nodes attributes if node.tag == 'attribute'
    keys of these attributes are prepended with 'attribute|'
    for easier downstream processing

    :param node:    ElementTree.Element instance

    :return:        dictionary containing the nodes attributes
    """
    harmonized_name = node.attrib.get('harmonized_name')
    attribute_name = node.attrib.get('attribute_name')
    name = harmonized_name if harmonized_name else attribute_name
    return {f'attribute|{name}': node.text}


def parse_organism(node: ElementTree.Element) -> dict[str, str]:
    """
    returns the organism and taxonomy id if node.tag == 'organism'

    :param node:    ElementTree.Element instance

    :return:        dictionary containing the organism and taxonomy id
    """
    return {'organism': ';'.join(node.attrib[k] for k in ['taxonomy_name', 'taxonomy_id'])}


def parse_biosample_uid(node: ElementTree.Element) -> dict[str, str]:
    """
    returns the biosample accession if node.tag == 'biosample'

    :param node:    ElementTree.Element instance

    :return:        dictionary containing the biosample id
    """
    return {'biosample': node.attrib['id']}


biosample_metadata_parsers = {
    'Title': parse_title,
    'Attribute': parse_attribute,
    'Organism': parse_organism,
    'BioSample': parse_biosample_uid
}


def combine_attribute_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    extracts all keys starting with 'attribute|' from metadata
    and puts them back as a single 'attribute' key joining the
    individual key-value-pairs to string as follows "key: value; key: value; ..."

    :param metadata:    dictionary containing the metadata for a given sample parses from XML

    :return:            metadata with a single attribute key added back instead of all "attribute|" keys
    """
    attribute_kv_list = []
    revised_metadata = dict()
    for key, value in metadata.items():
        if not key.startswith('attribute'):
            revised_metadata[key] = value
            continue
            
        name = key.split('|')[-1]
        attribute_kv_list.append(f'{name}: {value}')
    
    revised_metadata['attribute'] = '; '.join(attribute_kv_list)
    return revised_metadata


def extract_metadata(xmltree: ElementTree, metadata_parsers: dict[str, Callable]) -> dict[str, Any]:
    """
    parses the XML formated metadata and returns a dictionary of it

    :param xmltree:             ElementTree instance initialized with the biosample XML response from eSummary
    :param metadata_parsers:    dictionary containing matching node tags as keys and node parser functions as values

    :return:                    dictionary containing the metadata parsed with metadata_parsers
    """
    metadata = dict()
    for node in xmltree.iter():
        for match_string, parse_function in metadata_parsers.items():
            match, parse_result = node_parser(
                node, 
                parse_function, 
                match_string
            )
            
            if match:
                metadata.update(parse_result)
                break
                
    return combine_attribute_keys(metadata)


def add_parsing_results_to_metadata(parsed_metadata: dict[str, Any], metadata: dict[str, list[Any]]) -> None:
    """
    adds parsed_metadata values to the list under the corresponding key in metadata
    this is done in place

    :param parsed_metadata:     the parsed metadata of a single eSummary response
    :param metadata:            a collection of parsed metadata from multiple eSummary responses

    :return:                    None
    """
    for key in metadata.keys():
        metadata[key].append(
            parsed_metadata.get(key)
        )
        

def metadata_from_biosample_uids(
    summary_items: list[dict[str, Any]],
    metadata: dict[str, list[Any]], 
    metadata_parsers: dict[str, Callable]
) -> None:
    """
    parses the sSummary responses from multiple items and adds it to the metadata dictionary in place

    :param summary_items:       esummary response as returned by Entrez.read
    :param metadata:            dictionary the parsed metadata should be stored in
    :param metadata_parsers:    dictionary containing node.tags as keys and corresponding parser functions as values

    :return:                    None
    """
    for summary in summary_items['DocumentSummarySet']['DocumentSummary']:
        xmltree = ElementTree.fromstring(summary['SampleData'])
        parsed_metadata = extract_metadata(
            xmltree,
            metadata_parsers
        )
        add_parsing_results_to_metadata(
            parsed_metadata,
            metadata
        )


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
    for summary in summary_items:
        extract_accessions(
            summary,
            accessions,
            accession_matchers
        )


accession_matchers = {
    'experiment': re.compile('Experiment acc="([SRX0-9]+)'),
    'sample': re.compile('Sample acc="(SRS[0-9]+)'),
    'study': re.compile('Study acc="(SRP[0-9]+)'),
    'biosample': re.compile('Biosample>(SAMN[0-9]+)'),
    'bioproject': re.compile('Bioproject>(PRJNA[0-9]+)'),
    'geo': re.compile('name="(GSM[0-9]+)')
}


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