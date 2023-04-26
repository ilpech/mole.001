import requests, sys
import os
import json
from tools_dl.tools import ensure_folder

def getGeneFromApi(
    uniprot_id, 
    outdir_='./bio_dl/testdata/api_out'
):
    """
    Request to Uniprot API
    """
    outdir = os.path.abspath(outdir_)
    if not os.path.isdir(outdir):
        ensure_folder(outdir)
    out = os.path.abspath('{}/{}.json'.format(outdir, uniprot_id))
    if os.path.isfile(out):
        with open(out, 'r') as f:
            try:
                out_from_log = json.load(f)
                return out_from_log
            except:
                raise Exception('error reading {}'.format(out))
    else:
        params = {
            'accession': uniprot_id,
        }
        requestURL = "https://www.ebi.ac.uk/proteins/api/coordinates"
        r = requests.get(
            requestURL, 
            headers={"Accept" : "application/json"},
            params=params
        )
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        responseBody = r.json()
        with open(out, 'w') as f:
            json.dump(responseBody, f)
        print('written out', out)
    return responseBody

def sequence(uniprot_id):
    """
    Aminoacids sequence of protein
    """
    gene_data = getGeneFromApi(uniprot_id)
    if not len(gene_data):
        return ''
    return gene_data[0]['sequence']
    
if __name__ == '__main__':
    gene_id = 'Q9Y5B0'
    print(gene_id, sequence(gene_id))