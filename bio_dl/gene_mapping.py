import os 
import gzip

class GenesMapping:
    """
    Data structure to handle genocentric features
    mappings from different databases
    """
    def __init__(self, readpath=''):
        self.__mapping = {}
        if len(readpath):
            self.read(readpath)
        
    def mapping(self):
        return self.__mapping
    
    def add(self, uniprot_id, ensg_id):
        self.mapping()[uniprot_id] = self.GeneMapping(uniprot_id, ensg_id)
    
    def write(self, outpath):
        if not os.path.isdir(os.path.split(outpath)[0]):
            os.makedirs(os.path.split(outpath)[0])
        with open(outpath, 'w') as f:
            for k,v in self.mapping().items():
                f.write('{} {}\n'.format(k,v.ensg_id))
        print('{} genes mapping written to {}'.format(len(self.mapping()), outpath))
        
    def read(self, inpath):
        with open(inpath, 'r') as f:
            data = f.readlines()
            print('{} mapping lines read from {}'.format(len(data), inpath))
            for d in data:
                d = d.replace('\n', '')
                splt = d.split(' ')
                self.mapping()[splt[0]] = self.GeneMapping(splt[0], splt[1])

    class GeneMapping:
        """
        ensg... -> uniprot id pair
        """
        def __init__(
            self,
            uniprot_id,
            ensg_id
        ):
            self.ensg_id = ensg_id 
            self.uniprot_id = uniprot_id

def uniprot_mapping_header():
    """
    Collection of databases in 
    Uniprot's mapping official file
    """
    return [
        'UniProtKB-AC',
        'UniProtKB-ID',
        'GeneID (EntrezGene)',
        'RefSeq',
        'GI',
        'PDB',
        'GO',
        'UniRef100',
        'UniRef90',
        'UniRef50',
        'UniParc',
        'PIR',
        'NCBI-taxon',
        'MIM',
        'UniGene',
        'PubMed',
        'EMBL',
        'EMBL-CDS',
        'Ensembl',
        'Ensembl_TRS',
        'Ensembl_PRO',
        'Additional PubMed'
    ]
    
def uniq_nonempty_uniprot_mapping_header():
    """
    Collection of genocentric
    databases mappings with several
    features corresponding to one gene
    """
    return [
        'GO',
        'RefSeq',
        'MIM',
        'PubMed',
        'Ensembl_PRO',
        'PDB',
    ]
    

def mapping2dict(path):
    """
    Reader of Uniprot's official 
    mapping file
    """
    with open(path, 'r') as f:
        data = f.readlines()
    databases = uniprot_mapping_header()
    out = {}
    for d in data:
        splt = d.split('\t')
        data = {}
        for i in range(len(databases)):
            database_splt = splt[i].replace(';', '').split()
            data[databases[i]] = database_splt 
            if i == 0:
                main_name = database_splt[0]
        out[main_name] = data
    return out

def rewrite_mapping_with_ids(
    path, 
    uniprot_ids,
    outpath
):
    """
    Create shorter genocentric databases
    mapping file based on list of uniprot
    genes ids
    """
    with gzip.open(path, 'rt') as f:
        data = f.readlines()
    lines_ids2remain = []
    databases = uniprot_mapping_header()
    for line_id in range(len(data)):
        splt = data[line_id].split('\t')
        for i in range(len(databases)):
            database_splt = splt[i].replace(';', '').split()
            if i == 0:
                main_name = database_splt[0]
                if main_name in uniprot_ids:
                    lines_ids2remain.append(line_id)
    new_data = [data[i] for i in range(len(data)) if i in lines_ids2remain]
    with open(outpath, 'w') as f:
        f.writelines(new_data)
    print('remapping written in ', outpath)
                    


if __name__ == '__main__':
    out = mapping2dict('./bio_dl/testdata/human_ids_mapping.tab')
    for gene, databases in out.items():
        print('gene name', gene)
        for db, values in databases.items():
            print('DATABASE::', db)
            print(values)
            print('============')