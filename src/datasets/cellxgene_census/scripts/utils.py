import requests
import re
from Bio import Entrez

def doi_to_pmid(doi):
    # Construct the URL for eutils
    url = f"http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=PubMed&retmode=xml&term={doi}"

    try:
        # Make a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses

        # Extract PubMed ID using regular expression
        pubmed_id = re.search(r"<Id>(.*?)</Id>", response.text).group(1)
        return pubmed_id
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
    

def get_abstract(doi):
    pmid = doi_to_pmid(doi)
    
    if pmid:
        Entrez.email = "your-email@example.com"  # Provide your email address for identification

        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            record = Entrez.read(handle)
            article = record['PubmedArticle'][0]['MedlineCitation']['Article']

            if 'Abstract' in article:
                abstract = article['Abstract']['AbstractText']
                return str(abstract[0])
            else:
                return f"No abstract found for DOI: {doi} (PMID: {pmid})"
        except Exception as e:
            return f"Error fetching data for DOI: {doi} (PMID: {pmid}). Exception: {e}"
    else:
        return f"No PMID found for DOI: {doi}"
    
