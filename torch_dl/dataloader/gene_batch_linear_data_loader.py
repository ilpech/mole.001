import os
from torch.utils.data import DataLoader
from tools_dl.tools import (
    norm_shifted_log
)
from varname.helpers import debug
import numpy as np
import time

from bio_dl.gene import Gene
from torch.utils.data.distributed import DistributedSampler

from torch_dl.dataloader.gene_data_loader import DistGeneDataLoader
from torch_dl.dataloader.gene_batch_data_loader import BatchIterDistGeneDataLoader

class GeneLinearDataLoader(DistGeneDataLoader):
    
    @staticmethod
    def createDistLoader(
        config_path, 
        rank, 
        batch_size, 
        gpus2use,
        num_workers=os.cpu_count()-1,
        dataset=None, # already used DistGeneDataLoader for fast changing of batch size
        net_config_path=None, # to fast load db mappings
        use_net_experiments=False,
        read_genes_mapping=True,
        use_net_databases=True
    ):
        """
        Creating dataset for training and validation
        based on RNA->protein experiments
        with given size of batch
        """
        t1 = time.time()
        if dataset is None:
            dataset = GeneLinearDataLoader(
                config_path, 
                net_config_path,
                # use_net_experiments,
                # read_genes_mapping,
                # use_net_databases
            )
        train_dataset = TrainGeneLinearDataLoader(
            dataset, 
            'train'
        )
        # multi GPU sampler
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=gpus2use,
            rank=rank
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True, 
            sampler=train_sampler,
            # collate_fn=collate_fn 
        )
        print('createDistLoader::train dataloader created for rank {}'.format(rank))
        val_dataset = TrainGeneLinearDataLoader(
            dataset,
            'val'
        )
        # multi GPU sampler
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=gpus2use,
            rank=rank
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True, 
            sampler=val_sampler,
            # collate_fn=collate_fn 
        )
        print('createDistLoader::val dataloader created for rank {}'.format(rank))
        t2 = time.time()
        print('createDistLoader::time passed {:.3f}'.format(t2-t1))
        return train_dataloader, val_dataloader
    
    def gene2sample1d(
        self, 
        uid, 
        db_dicts2use, 
        rna_exp_name, 
        protein_exp_name, 
        rna_exps_alphabet=None, 
        protein_exps_alphabet=None, 
        rna_exp_id=None, 
        protein_exp_id=None
    ):
        gene = self.gene(uid)
        if not rna_exps_alphabet:
            rna_exps_alphabet = self.rnaMeasurementsAlphabet
        if not protein_exps_alphabet:
            protein_exps_alphabet = self.proteinMeasurementsAlphabet
        if rna_exp_id is None:
            rna_experiment_id = [j for j in range(len(rna_exps_alphabet)) if rna_exps_alphabet[j] == rna_exp_name]
            if not len(rna_experiment_id):
                raise Exception('len(rna_experiment_id) == 0 {}'.format(rna_exp_name)) 
            rna_experiment_id = rna_experiment_id[0]
        else:
            rna_experiment_id = rna_exp_id
            rna_exp_name = rna_exps_alphabet[rna_experiment_id]
        if protein_exp_id is None:
            prot_experiment_id = [j for j in range(len(protein_exps_alphabet)) if protein_exps_alphabet[j] == protein_exp_name]
            if not len(prot_experiment_id):
                raise Exception('len(prot_experiment_id) == 0 {}'.format(protein_exp_name))
            prot_experiment_id = prot_experiment_id[0]
        else:
            prot_experiment_id = protein_exp_id
        try:
            rna_value = gene.rna_measurements[rna_exp_name]
        except KeyError:
            return None
        
        norm_rna_value = norm_shifted_log(rna_value).astype(np.float32)
        norm_rna_value = np.expand_dims(norm_rna_value, 0)

        variable_length_layer_size = self.max_var_layer
        
        #concatenate 1d one hot vectors for each db like in original paper
        #in original paper they have other dims values - check
        annotations_gene_one_hot = None
        for db_name, db_data in db_dicts2use.items():
            db_gene_one_hot = self.mappingDatabase2oneHot(
                db_name, 
                db_data,
                [uid]
            )
            # debug(db_gene_one_hot.shape, db_name)
            if annotations_gene_one_hot is None:
                annotations_gene_one_hot = db_gene_one_hot[0]
            else:
                annotations_gene_one_hot = np.concatenate(
                    [annotations_gene_one_hot, db_gene_one_hot[0]],
                    axis=0, dtype=np.float32
                )
        
        protein_exp_one_hot = np.zeros([len(protein_exps_alphabet)], dtype=np.float32)
        protein_exp_one_hot[protein_exp_id].fill(1.0)
        
        return annotations_gene_one_hot, protein_exp_one_hot, norm_rna_value


class TrainGeneLinearDataLoader(DataLoader):
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
        if self.mode == 'train':
            self.ids2use = self.baseloader.trainExpsIds()
        else:
            self.ids2use = self.baseloader.valExpsIds() 
        uid, rna_id, prot_id = self.ids2use[i]
        gene = self.baseloader.gene(uid)
        databases_alphs = self.baseloader.databases_alphs
        prot_exps_alph = self.baseloader.proteinMeasurementsAlphabet
        prot_exp = prot_exps_alph[prot_id]
        annotations, type_one_hot, norm_rna_val = self.baseloader.gene2sample1d(
            uid,
            databases_alphs,
            rna_exp_name=None,
            protein_exp_name=None,
            rna_exp_id=rna_id,
            protein_exp_id=prot_id,
        )
        if annotations is None:
            return None 
        label = norm_shifted_log(
            gene.protein_measurements[prot_exp]
        )/self.max_label
        return annotations, type_one_hot, norm_rna_val, label
