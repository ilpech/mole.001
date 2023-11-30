from torch.utils.data import DataLoader
from tools_dl.tools import (
    norm_shifted_log
)
from varname.helpers import debug

class BatchIterDistGeneDataLoader(DataLoader):
    """
    Easy batch multi-GPU iteration based on
    Torch DataLoader class and 
    RNA->protein experiments
    
    train/val modes available
    """
    def __init__(
        self, 
        baseloader, 
        mode='train'
    ):
        self.baseloader = baseloader
        self.mode = mode
        self.max_label = baseloader.max_label
    
    def __len__(self):
        """
        returns number of experiments
        based on selected mode
        """
        if self.mode == 'train':
            return len(self.baseloader.trainExpsIds())
        if self.mode == 'val':
            return len(self.baseloader.valExpsIds())
        return 0
    
    def __getitem__(self, i):
        """
        Iter function, returns gene2sample
        for RNA->protein experiment with id i
        """
        if self.mode == 'train':
            ids2use = self.baseloader.trainExpsIds()
        else:
            ids2use = self.baseloader.valExpsIds()
        uid, rna_id, prot_id = ids2use[i]
        gene = self.baseloader.gene(uid)
        databases_alphs = self.baseloader.databases_alphs
        prot_exps_alph = self.baseloader.proteinMeasurementsAlphabet
        prot_exp = prot_exps_alph[prot_id]
        sample = self.baseloader.gene2sample(
            uid,
            databases_alphs,
            rna_exp_name=None,
            protein_exp_name=None,
            rna_exp_id=rna_id,
            protein_exp_id=prot_id,
        )
        if sample is None:
            return None
        label = norm_shifted_log(
            gene.protein_measurements[prot_exp]
        )/self.max_label
        return sample, label
    
class InferenceBatchGeneDataLoader(DataLoader):
    """
    Easy batch multi-GPU iteration based on
    Torch DataLoader class and 
    RNA->protein experiments
    
    train/val modes available
    """
    def __init__(
        self, 
        baseloader, 
        mode='train'
    ):
        self.baseloader = baseloader
        self.mode = mode
        self.max_label = baseloader.max_label
    
    def __len__(self):
        """
        returns number of experiments
        based on selected mode
        """
        if self.mode == 'train':
            return len(self.baseloader.trainExpsIds())
        if self.mode == 'val':
            return len(self.baseloader.valExpsIds())
        return 0
    
    def __getitem__(self, i):
        """
        Iter function, returns gene2sample
        for RNA->protein experiment with id i
        """
        if self.mode == 'train':
            ids2use = self.baseloader.trainExpsIds()
        else:
            ids2use = self.baseloader.valExpsIds()
        uid, rna_id, prot_id = ids2use[i]
        gene = self.baseloader.gene(uid)
        databases_alphs = self.baseloader.databases_alphs
        prot_exps_alph = self.baseloader.proteinMeasurementsAlphabet
        prot_exp = prot_exps_alph[prot_id]
        sample = self.baseloader.gene2sample(
            uid,
            databases_alphs,
            rna_exp_name=None,
            protein_exp_name=None,
            rna_exp_id=rna_id,
            protein_exp_id=prot_id,
        )
        if sample is None:
            return None
        label = norm_shifted_log(
            gene.protein_measurements[prot_exp]
        )/self.max_label
        return sample, label, uid, rna_id, prot_id
