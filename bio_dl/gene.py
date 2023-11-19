import numpy as np
import os
from tools_dl.tools import debug, norm_shifted_log
from bio_dl.uniprot_api import getGeneFromApi, sequence
from bio_dl.gene_mapping import GenesMapping

isdebug = False
# isdebug = True

class Gene:
    """
    Data structure for storing genocentric info
    with ability to request data from Uniprot API
    """
    def __init__(
        self,
        uniprot_id,
        only_w_values=False
    ):
        self.id_uniprot: str = uniprot_id
        # print('created gene: ', self.id_uniprot)
        self.protein_data = {}
        self.rna_measurements = {}
        self.protein_measurements = {}
        self.gene_name = None
        self.protein_name = None
        self.only_w_values = only_w_values
        self.mapping : GenesMapping.GeneMapping = None
    
    def get_RNA_experiment_value(
        self, 
        experiment_name,
        use_log_norm=True
    ):
        out = self.rna_measurements[experiment_name]
        if use_log_norm:
            return norm_shifted_log(out)
        return out
    
    def get_protein_experiment_value(
        self, 
        experiment_name,
        use_log_norm=True
    ):
        out = self.protein_measurements[experiment_name]
        if use_log_norm:
            return norm_shifted_log(out)
        return out
        
        
    def id(self):
        return self.id_uniprot
        
    @staticmethod
    def proteinAminoAcidsAlphabet():
        """
        protein aminoacids in human body
        """
        return sorted([
            'A', 'C', 'D', 'E', 'F', 
            'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 
            'S', 'T', 'U', 'V', 'W', 'Y'
        ])
    
    def apiData(self, outdir):
        return getGeneFromApi(self.id(), outdir)
    
    def apiSequence(self, outdir):
        return sequence(self.id(), outdir)
    
    def apiSeqOneHot(self, outdir):
        """
        One hot coding of aminoacids sequence
        """
        return Gene.seq2oneHot(self.apiSequence(outdir))
    
    @staticmethod
    def seq2oneHot(seq):
        alphabet = Gene.proteinAminoAcidsAlphabet()
        set_seq = set(seq)
        if len(set_seq) > len(alphabet):
            return None
        try:
            onehot = [[0] * len(alphabet)] * len(seq)
            onehot = np.array(onehot)
            for i in range(len(seq)):
                pos = [
                    j for j in range(len(alphabet)) if alphabet[j] == seq[i]
                ]
                if not len(pos):
                    continue
                onehot[i][pos[0]] = 1 #alphabet[j]
            return onehot
        except:
            print('error in seq2oneHot')
            debug(len(alphabet))
            debug(len(set_seq))
            debug(len(seq))
            return None
        