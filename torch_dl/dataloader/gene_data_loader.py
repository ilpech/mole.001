# python3 torch_dl/dataloader/gene_data_loader.py
import os
import torch
import numpy as np
import random
import json
from tools_dl.tools import (
    denorm_shifted_log,
    ensure_folder,
    roundUp,
    norm_shifted_log,
    curDateTime,
    shuffle,
    find_files,
    boolean_string
)
from tools_dl.heatmap_table import matrix2heatmap
from bio_dl.gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header,
)
from bio_dl.uniprot_api import sequence
from bio_dl.gene import Gene, GenesMapping
import csv

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch_dl.dataloader.gene_batch_data_loader import BatchIterDistGeneDataLoader

from typing import List
import time

from varname.helpers import debug

# isdebug = True
isdebug = False

class DistGeneDataLoader(Dataset):
    """
    Distributive multi-GPU genocentic dataloader
    with genes from uniprot API,
    any dataset multiomics experiment
    could be added to genes info
    """
    def __init__(
        self, 
        config_path, 
        net_config_path=None,
        use_net_experiments=True
    ):
        self.creation_time = curDateTime()
        self.config_path = config_path
        if len(self.config_path) == 0:
           raise Exception('provide path to .yaml train config with data branch') 
        self.net_config_path = net_config_path
        self.use_net_experiments = use_net_experiments
        self.config = DistGeneDataLoader.opt_from_config(self.config_path)
        self.config_data = self.config['data']
        self.config_train = self.config['train']
        self.cash_path = self.config_data['cashPath']
        self.gene2sample_cash_path = None
        if self.cash_path != 'False':
            self.gene2sample_cash_path = os.path.join(
                self.cash_path, 
                'gene2sample'
            )
            ensure_folder(self.gene2sample_cash_path)
        self.max_var_layer = self.config_train['max_var_layer']
        self.gene_mapping_path = self.config_data['geneMapping']
        self.use_zero_prot = self.config_data['useZeroProt']
        self.use_non_zero_prot = self.config_data['useNonZeroProt']
        if not self.use_zero_prot and not self.use_non_zero_prot:
            raise Exception('At least one from useZeroProt and useNonZeroProt flags in {} must be True!'.format(self.config_path))
        self.splitGeneFromModelConfig = str(self.config_data['splitGeneFromModelConfig'])
        self.engs2uniprot_file = self.config_data['engs2uniprot_file']
        self.datasets2use = self.config_data['datasets2use']
        self.gene_ids = []
        self._genes = {} # key - uniprot id, value - Gene object
        self.genes2train = None
        self.genes2val = None
        # fast iterating through experiments ids for easy loader
        self._exps2indxs: List = [] 
        self._valexps2indxs: List = []
        self.genes_mapping_databases = uniprot_mapping_header()
        self.databases_alphs = {}
        self.max_len_db_alph = self.max_var_layer * len(Gene.proteinAminoAcidsAlphabet())
        print('reading mapping', self.gene_mapping_path)
        self.mapping = mapping2dict(self.gene_mapping_path)
        self.max_label = None
        self.net_config = None
        self.net_name = None
        self.proteinMeasurementsAlphabet = None
        self.rnaMeasurementsAlphabet = None
        self.net_proteinMeasurementsAlphabet = None
        self.net_rnaMeasurementsAlphabet = None
        print('ensg2uniprot mapping...')
        if not os.path.isfile(self.engs2uniprot_file):
            print('writing mappings in new file {}'.format(self.engs2uniprot_file))
            self.ensg2uniprot_mapping: GenesMapping = DistGeneDataLoader.ensg2uniprot(
                self.gene_mapping_path,
                self.engs2uniprot_file 
            )
        else:
            print('reading mappings from file {}'.format(self.engs2uniprot_file))
            self.ensg2uniprot_mapping = GenesMapping(self.engs2uniprot_file)
        for k, v in self.datasets2use.items():
            if boolean_string(v):
                self.loadTSVrna2protData(
                        k,
                        self.config_data['{}_rna'.format(k)],
                        self.config_data['{}_prot'.format(k)],
                        create_new_genes=True
                )        
        net_config_dir = self.net_config_path
        if net_config_path and not os.path.isfile(self.net_config_path):
            maybe_config = find_files(net_config_dir, '_config.json', abs_p=True)
            if not len(maybe_config):
                print('warning!!!::net config was not found by path {}'.format(self.net_config_path))
                self.net_config_path = None
            else:
                self.net_config_path = maybe_config[0]
                print(f'found net config::{self.net_config_path}!')
        if self.net_config_path:
            if not os.path.isdir(self.net_config_path):
                net_config_dir = os.path.dirname(self.net_config_path)
            with open(self.net_config_path, 'r') as f:
                self.net_config = json.load(f)
                self.net_name = self.net_config['net_name']
                self.net_genes2val = self.net_config['genes2val']
                self.net_rnaMeasurementsAlphabet = self.net_config['rna_exps_alphabet']
                self.net_proteinMeasurementsAlphabet = self.net_config['protein_exps_alphabet']
        self.fillDatabasesAlphabets(
            self.max_len_db_alph, 
            cash_dir=net_config_dir
        )
        if self.splitGeneFromModelConfig != 'False':
            self.splitGenesFromOtherModelConfig(
                path2model_config=self.splitGeneFromModelConfig
            )
        else:
            self.split_genes2val_train(
                percent_train=float(self.config_data['ifSplitPercentToTrain']), 
                shuffle=True
            )
        print('dataset created! warmup now...')
        self.makeProteinMeasurementsAlphabet()
        self.makeRNAMeasurementsAlphabet()
        if self.rnaMeasurementsAlphabet != self.proteinMeasurementsAlphabet:
            raise Exception('DistGeneDataLoader::rnaMeasurementsAlphabet = self.proteinMeasurementsAlphabet') 
        self._warmupExps()
        debug(len(self._valexps2indxs))
        debug(len(self._exps2indxs))
        debug(self.rnaMeasurementsAlphabet)
        debug(len(self.rnaMeasurementsAlphabet))
        debug(self.proteinMeasurementsAlphabet)
        debug(len(self.proteinMeasurementsAlphabet))
        debug(self.maxProteinMeasurementInData())
        debug(self.maxRNAMeasurementInData())
        
    @staticmethod
    def createDistLoader(
        config_path, 
        rank, 
        batch_size, 
        gpus2use,
        num_workers=os.cpu_count()-1,
        dataset=None, # already used DistGeneDataLoader for fast changing of batch size
        net_config_path=None, # to fast load db mappings
        use_net_experiments=False
    ):
        """
        Creating dataset for training and validation
        based on RNA->protein experiments
        with given size of batch
        """
        t1 = time.time()
        if dataset is None:
            dataset = DistGeneDataLoader(
                config_path, 
                net_config_path,
                use_net_experiments
            )
        train_dataset = BatchIterDistGeneDataLoader(
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
        print('train dataloader created for rank {}'.format(rank))
        val_dataset = BatchIterDistGeneDataLoader(
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
        print('val dataloader created for rank {}'.format(rank))
        t2 = time.time()
        print('time passed {:.3f}'.format(t2-t1))
        return train_dataloader, val_dataloader
        
    def genesCnt(self):
        return len(self.genes())
    
    def _warmupExps(self):
        """
        Create indexes of RNA->protein experiment
        for fast iteration. Call once for created
        dataloader
        """
        t1 = time.time()
        print('warmup:: creating exps idxs...')
        rna_exps_alphabet = self.rnaMeasurementsAlphabet
        protein_exps_alphabet = self.proteinMeasurementsAlphabet
        # debug(rna_exps_alphabet)
        for gene_uid, gene in self.genes().items():
            to_train = False
            if gene_uid in self.genes2train:
                to_train = True
            for rna_exp, value in gene.rna_measurements.items():
                if value == 0.:
                    continue
                rna_exp_ids = [j for j in range(len(rna_exps_alphabet)) if rna_exps_alphabet[j] == rna_exp]
                if not len(rna_exp_ids):
                    continue
                prot_exp_ids = [
                    j for j in range(len(protein_exps_alphabet)) if protein_exps_alphabet[j] == rna_exp
                ]
                if not len(prot_exp_ids):
                    continue
                # random.shuffle(prot_exp_ids)
                for prot_exp_id in prot_exp_ids:
                    prot_exp_name = protein_exps_alphabet[prot_exp_id]
                    try:
                        protein_exp = gene.protein_measurements[prot_exp_name]
                    except KeyError:
                        continue
                    if to_train:
                        self._exps2indxs.append([gene_uid, rna_exp_ids[0], prot_exp_id])
                    else:
                        self._valexps2indxs.append([gene_uid, rna_exp_ids[0], prot_exp_id])
                    break
        print('creating experiments indxs done')
        print('{} train exps in data'.format(len(self._exps2indxs)))
        print('{} val exps in data'.format(len(self._valexps2indxs)))
        t2 = time.time()
        print('time passed {:.3f}'.format(t2-t1))
       
    @staticmethod
    def opt_from_config(config_path):
        """
        read dataloader options from config
        """
        import yaml
        with open(config_path) as c:
            return yaml.load(c, Loader=yaml.FullLoader)
        
    def splitGenesFromOtherModelConfig(
        self,
        path2model_config
    ):
        """
        Split genes to validation based on
        previous model's config
        """
        genes2val = None
        with open(path2model_config, 'r') as f:
            cfg = json.load(f)
            genes2val = cfg['genes2val']
        if genes2val is None or not len(genes2val):
            print('error reading genes2val from config {}'.format(path2model_config))
        self.genes2train = [x for x, _ in self.genes().items() if x not in genes2val]
        self.genes2val = [x for x, _ in self.genes().items() if x in genes2val]
        self.max_label = cfg['denorm_max_label']
        print('genes splitted from file {} \n{} in val \n{} in train'.format(
            path2model_config, 
            len(self.genes2val), 
            len(self.genes2train),
        ))

    def split_genes2val_train(
        self,
        percent_train,
        shuffle=True
    ):
        """
        Split genes to train/validation randomly
        1 - percent_train genes go to validation
        
        Params:
        
        percent_train in (0.0,1.0)
        
        creates self.train_gene_names, self.val_gene_names
        """
        assert percent_train < 1.0
        genes_cnt = len(self.genes())
        gene_ids = [j for j in range(genes_cnt)]
        if shuffle:
            random.shuffle(gene_ids)
        samples2train = int(roundUp(genes_cnt*percent_train))
        genes_names = self.genesKeys()
        self.genes2train = [genes_names[x] for x in gene_ids[:samples2train]]
        self.genes2val = [genes_names[x] for x in gene_ids[samples2train:]]
        print('genes splitted with {} to train \n{} in val \n{} in train'.format(
            percent_train, 
            len(self.genes2val), 
            len(self.genes2train),
        ))
        self.max_label = norm_shifted_log(self.maxProteinMeasurementInData())

    def __len__(self):
        return len(self.trainExpsIds()) + len(self.valExpsIds())

    def shuffleTrain(self):
        self._exps2indxs = shuffle(self._exps2indxs)[0]
        
    def shuffleVal(self):
        self._valexps2indxs = shuffle(self._valexps2indxs)[0]

    def trainExpsIds(self):
        """
        Get indexes of train experiments
        """
        return self._exps2indxs

    def valExpsIds(self):
        """
        Get indexes of val experiments
        """
        return self._valexps2indxs
   
    def genes(self):
        return self._genes
    
    def gene(self, uniprot_gene_id):
        """
        Access gene by uniprot id
        """
        try:
            return self.genes()[uniprot_gene_id]
        except KeyError:
            raise KeyError('Gene::gene {} not found'.format(uniprot_gene_id))
    
    def genesKeys(self):
        """
        Uniprot's ids of loaded genes
        """
        return list(self.genes().keys())

    def rnaExperimentsCount(self):
        return sum([
            len(g.rna_measurements) for g_name, g in self.genes().items()
        ])
        
    def proteinExperimentsCount(self):
        return sum([
            len(g.protein_measurements) for g_name, g in self.genes().items()
        ])
    
    def makeRNAMeasurementsAlphabet(self):
        """
        Names of all of RNA abundance experiments
        of genes in data
        """
        out = []
        for g_name, g in self.genes().items():
            for k, v in g.rna_measurements.items():
                if k not in out:
                    out.append(k)
        out = sorted(list(set(out)))
        if self.net_rnaMeasurementsAlphabet and self.use_net_experiments:
            print('makeRNAMeasurementsAlphabet::experiments added to list from {}'.format(self.net_name))
            out_new = [x for x in out if x not in self.net_rnaMeasurementsAlphabet]
            out = self.net_rnaMeasurementsAlphabet + out_new 
        self.rnaMeasurementsAlphabet = out
         
    def makeProteinMeasurementsAlphabet(self):
        """
        Names of all of protein abundance experiments
        of genes in data
        """
        out = []
        for g_name, g in self.genes().items():
            for k, v in g.protein_measurements.items():
                if k not in out:
                    out.append(k)
        out = sorted(list(set(out)))
        if self.net_proteinMeasurementsAlphabet and self.use_net_experiments:
            print('makeProteinMeasurementsAlphabet::experiments added to list from {}'.format(self.net_name))
            out_new = [x for x in out if x not in self.net_proteinMeasurementsAlphabet]
            out = self.net_proteinMeasurementsAlphabet + out_new 
        self.proteinMeasurementsAlphabet = out
    
    def maxRnaMeasurementsInData(self):
        """
        Max gene uniq RNA experiments count in data
        """
        m = 0
        for g_name, g in self.genes().items():
            exps = len(g.rna_measurements)
            if exps > m:
                m = exps
        return m
    
    def maxProteinMeasurementsInData(self):
        """
        Max gene uniq protein experiments count in data
        """
        m = 0
        for g_name, g in self.genes().items():
            exps = len(g.protein_measurements)
            if exps > m:
                m = exps
        return m
    
    def maxRNAMeasurementInData(self):
        """
        Max protein abundance value in data
        """
        m = 0
        for g_name, g in self.genes().items():
            exps = max([v for _, v in g.rna_measurements.items()])
            if exps > m:
                m = exps
        return m
    
    def maxProteinMeasurementInData(self):
        """
        Max protein abundance value in data
        """
        m = 0
        for g_name, g in self.genes().items():
            exps = max([v for _, v in g.protein_measurements.items()])
            if exps > m:
                m = exps
        return m

    def gene2sample(
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
        '''
        Vectorize gene RNA->protein experiment
        to inference by deep neural network model
        
        4 + len(db_dicts2use) channels::
            - 1-st channel: protein experiment id row filled by ones
            - 2-nd channel: RNA experiment id row filled by ones 
            - 3-rd channel: filled with normed RNA experiment abundance 
            - 4-nd channel: filled by gene aminoacids seq in onehot representation
            - ... channels: onehot coding features from database of db_dicts2use
            
        returned matrix shape:
            (
                4 + len(db_dicts2use), 
                variable_length_layer_size, 
                len(Gene.proteinAminoAcidsAlphabet())
            ) 
             
        Params:
        db_dicts2use is dict with keys - db_name, values - db_alphabet
        
        aminoacids channel is cut by max_var_layer_data
        '''
        if isdebug:
            t1 = time.time()
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
            if isdebug:
                print('gene {} has no rna exp {}'.format(uid, rna_exp_name))
                print(gene.rna_measurements.items())
            return None
        norm_rna_value = norm_shifted_log(rna_value)
        variable_length_layer_size = self.max_var_layer
        t_shape = (
            len(db_dicts2use) + 4, 
            variable_length_layer_size,
            len(Gene.proteinAminoAcidsAlphabet())
        )
        use_cash = False
        gene_t_cash_name = '{}_{}.{}.{}.pt'.format(
            uid,
            *t_shape
        )
        if self.gene2sample_cash_path:
            use_cash = True
            gene_t_cash_path = os.path.join(
                self.gene2sample_cash_path,
                gene_t_cash_name
            )
        if use_cash and os.path.isfile(gene_t_cash_path):
            batch = torch.load(gene_t_cash_path)
            batch[0:3] *= 0.0
            # 1-st channel: protein experiment id row filled by ones
            batch[0, prot_experiment_id].fill(1.0)
            # 2-nd channel: RNA experiment id row filled by ones
            batch[1, rna_experiment_id].fill(1.0)
            # 3-rd channel: filled with normed RNA experiment abundance
            batch[2].fill(norm_rna_value)
            if isdebug: 
                debug(batch.shape)
                cash_t = time.time() - t1
                print('batch loaded from cash {} time {} sec'.format(gene_t_cash_path, cash_t))
                print()
            return batch
        batch = np.zeros(
            t_shape,
            dtype=np.float32
        )
        # 1-st channel: protein experiment id row filled by ones
        batch[0, prot_experiment_id].fill(1.0)
        # 2-nd channel: RNA experiment id row filled by ones
        batch[1, rna_experiment_id].fill(1.0)
        # 3-rd channel: filled with normed RNA experiment abundance
        batch[2].fill(norm_rna_value)
        # 4-nd channel: filled by gene aminoacids seq in onehot representation
        gene_seq_onehot = gene.apiSeqOneHot()
        if gene_seq_onehot is not None and len(gene_seq_onehot):
            onehot_rows = gene_seq_onehot.shape[0]
            if onehot_rows > variable_length_layer_size:
                onehot_rows = variable_length_layer_size
            batch[3][:onehot_rows] = gene_seq_onehot[:onehot_rows]
        k = 0
        for db_name, db_data in db_dicts2use.items():
            id2fill = k + 4
            db_gene_data = self.mappingDatabase2matrix(
                db_name, 
                db_data,
                [uid],
                cols=len(Gene.proteinAminoAcidsAlphabet())
            )
            len_filled = db_gene_data.shape[1]
            if len_filled > len(batch[id2fill]):
                len_filled = len(batch[id2fill])
            batch[id2fill][:len_filled] = db_gene_data[0]
            k += 1
        if use_cash:
            torch.save(batch, gene_t_cash_path)
            if isdebug:
                print('gene tensor created and saved to {} time {} sec'.format(
                    gene_t_cash_path,
                    time.time() - t1
                ))
        return batch
    
    # def augment_sample(self, sample):
    #     if self.with_augm
    #     c, h, w = sample.shape
    #     channels2augment = list(range(4, c))
    #     channel = random.choice(channels2augment)
    #     return
        
    def dataFromMappingDatabase(self, db_name, gene_name):
        '''
        db_name should exist in self.genes_mapping_databases
        '''
        return self.mapping[gene_name][db_name]
    
    def mappingDatabaseAlphabet(self, db_name):
        uniq_data = []
        for i in range(len(self.genes())):
            db_gene_data = self.dataFromMappingDatabase(
                db_name, 
                self.genes()[i].id_uniprot
            )
            for data in db_gene_data:
                if data not in uniq_data:
                    uniq_data.append(data) 
        return uniq_data

    def mappingDatabaseAplhabetSize(self, db_name):
        return len(self.mappingDatabaseAlphabet(db_name))
    
    def fillDatabasesAlphabets(
        self,
        max_len_alph,
        databases=uniq_nonempty_uniprot_mapping_header(),
        cash_dir=None
    ):
        """
        Filling self.databases_alphs with max layer size
        """
        print(f'filling database alphabets with max {max_len_alph}...')
        cash_file = None
        if cash_dir is not None:
            db_cash = find_files(cash_dir, '_databases_', abs_p=True)
            if len(db_cash):
                cash_file = db_cash[0]
        if cash_file:
            with open(cash_file, 'r') as f:
                cash_data = json.load(f)
            self.databases_alphs = cash_data
            print(f'databases alphabets loaded from cash file {cash_file}')
            return self.databases_alphs
        for db_name in databases:
            uniq_data = []
            for uid in self.genes().keys():
                db_gene_data = self.dataFromMappingDatabase(
                    db_name, 
                    uid
                )
                for data in db_gene_data:
                    if data not in uniq_data:
                        uniq_data.append(data)
            print('db {} filled with {} ids'.format(db_name, len(uniq_data)))
            if len(uniq_data) > max_len_alph:
                print('last ids deleted')
                uniq_data = uniq_data[:max_len_alph]
            self.databases_alphs[db_name] = uniq_data
        return self.databases_alphs 
        
    
    def mappingDatabase2matrix(
        self, 
        db_name,
        db_alphabet, 
        gene_names,
        cols=len(Gene.proteinAminoAcidsAlphabet())
    ):
        '''
        Construct channel-shape matrix with
        onehot coding of features from genocentric database
        based on genes names
        '''
        onehot = self.mappingDatabase2oneHot(
            db_name, 
            db_alphabet,
            gene_names
        )
        uniq_size = int(roundUp(onehot.shape[1]/float(cols)))
        reshaped = np.zeros((onehot.shape[0], uniq_size, cols)).flatten()
        for gene_id in range(len(onehot)):
            for value_id in range(len(onehot[gene_id])):
                reshaped[len(onehot[gene_id])*gene_id + value_id] = onehot[gene_id][value_id]
        reshaped = np.reshape(reshaped, (onehot.shape[0], uniq_size, cols))
        return reshaped
    
    def mappingDatabase2oneHot(
        self, 
        db_name,
        db_alphabet,
        gene_names
    ):
        '''
        Construct vector with
        onehot coding of features from genocentric database
        based on genes names
        '''
        npa_len = len(gene_names)
        npa = np.zeros(shape=(npa_len, len(db_alphabet)))
        for i in range(len(gene_names)):
            uid = gene_names[i]
            db_gene_data = self.dataFromMappingDatabase(
                db_name, 
                uid
            )
            for j in range(len(db_alphabet)):
                found = [x for x in db_gene_data if x == db_alphabet[j]]
                if len(found) > 0:
                    npa[i][j] = 1.0
        return npa

    def sequencesMaxAnalys(self):
        max_seq = None
        max_set_seq = None
        for gene_id, gene in self.genes().items():
            seq = sequence(gene_id)
            onehot = gene.apiSeqOneHot()
            set_seq = set(seq)
            if max_seq == None or len(seq) > len(max_seq):
                max_seq = seq
            if max_set_seq == None or len(set_seq) > len(max_set_seq):
                max_set_seq = set_seq
        print('max seq', max_seq, len(max_seq))
        print('max set seq', max_set_seq, len(max_set_seq))
        
    def sequencesAnalys(self, out_path):
        with open(out_path, 'w') as f:
            f.write('id\tseq_l\n')
            for gene_id, gene in self.genes().items():
                seq = sequence(gene_id)
                f.write('{}\t{}\n'.format(gene_id, len(seq)))
        print(f'sequencesAnalys written in {out_path}')

    def info(self):
        """
        Info about loaded genes
        """
        gene_num = 0
        for gene_id, gene in self.genes().items():
            gene_num += 1
            print('====={}======='.format(gene_num))
            print(gene_id)
            # debug(gene.protein_name)
            # debug(gene.nextprot_status)
            # debug(gene.peptide_seq)
            # debug(gene.chromosome_pos)
            print('rna experiments:: len::', len(gene.rna_measurements))
            for m,v in gene.rna_measurements.items():
                print('rna::', m, '::', v)
                prot_m = gene.protein_measurements[m]
                print('protein::', m, '::', prot_m)
            for db_name in uniq_nonempty_uniprot_mapping_header():
                print(f'{db_name} keywords::')
                mappings = self.mapping[gene.id()][db_name]
                print(len(mappings))
                print(mappings)
   
    @staticmethod 
    def ensg2uniprot(
        mapping_path='./bio_dl/testdata/human_18chr_tissue29_ids_mapping.tab',
        out_path='data/mapping_out/engs2uniprot.txt'
    ):
        """
        Construct ensg to uniprot ids mapping based
        on loaded genes
        """
        print('ensg2uniprot::reading {}'.format(mapping_path))
        mapping = mapping2dict(mapping_path)
        mapping_size = len(mapping)
        ensg_db = 'Ensembl'
        uniprot_db = 'UniProtKB-ID'
        out = GenesMapping()
        i = 0
        for gene_name, value in mapping.items():
            if not i % 5000:
                print('{} of {}'.format(i, mapping_size))
            if len(value[ensg_db]): 
                out.add(gene_name, value[ensg_db][0].split('.')[0])
            i += 1
        if len(out_path):
            out.write(out_path)
        return out
    
    def uniprot2ensg(self, uniprot_gene_id):
        """
        Convertion between uniprot id and ENSG... as tissue29 format
        """
        return self.dataFromMappingDatabase('Ensembl', uniprot_gene_id)
    
    def uniprot2db(self, uniprot_gene_id, db2convert_name):
        """
        Convert from uniprot to 
        bio_dl/gene_mapping.py::uniprot_mapping_header() databases
        """
        try:
            return self.dataFromMappingDatabase(
                db2convert_name, 
                uniprot_gene_id
            )
        except:
            raise Exception(
                'error finding mapping Uniprot::{} ==> {} database'.format(
                uniprot_gene_id,
                db2convert_name
            ))
    
    def loadTSVrna2protData(
        self, 
        dataset_name,
        rna_path, 
        prot_path,
        create_new_genes=True,
        max_genes=None
    ):
        """
        Add data from tissue29-like tsv format
        to existing genes or create new if create_new_genes
        
        first column is gene id (should be in Ensembl format)
        """
        rna_tissues = []
        prot_tissues = []
        rna_header = []
        prot_header = []
        rna_data = []
        rna_ensg_ids = []
        rna_id_col_name = None
        prot_data = []
        prot_ensg_ids = []
        prot_id_col_name = None
        ensg2uniprot = self.ensg2uniprot_mapping.mapping()
        with open(rna_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not len(rna_header):
                for rs in reader:
                    for r in rs:
                        rna_header.append(r)
                    break
                rna_id_col_name = rna_header[0]
                rna_tissues = rna_header[1:]
            for ord_dict in reader:
                rna_ensg_ids.append(ord_dict[rna_id_col_name]) 
                rna_data.append(ord_dict)
        print('rna data loaded {} from file {}'.format(
            len(rna_ensg_ids),
            rna_path
        ))
        with open(prot_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not len(prot_header):
                for rs in reader:
                    for r in rs:
                        prot_header.append(r)
                    break
                prot_id_col_name = prot_header[0]
                prot_tissues = prot_header[1:]
            for ord_dict in reader:
                prot_ensg_ids.append(ord_dict[prot_id_col_name]) 
                prot_data.append(ord_dict)
        print('prot data loaded {} from file {}'.format(
            len(prot_ensg_ids),
            prot_path
        ))
        if rna_tissues != prot_tissues:
            raise Exception('rna_tissues != prot_tissues')
        tissues = rna_tissues
        good_data = 0
        ensg2uniprot_errors = []
        for i in range(len(prot_ensg_ids)):
            ensg_id = prot_ensg_ids[i]
            uniprot_id = [
                x.uniprot_id for k, x in ensg2uniprot.items() if x.ensg_id == ensg_id
            ]
            if not len(uniprot_id):
                ensg2uniprot_errors.append(ensg_id)
                print('error while searching ensg {} in uniprot'.format(ensg_id))
                continue
            uniprot_id = uniprot_id[0]
            is_new_gene = False
            try:
                gene = self.genes()[uniprot_id]
                # print('tissue29 data would be added to currently existing gene {}'.format(gene.id()))
            except KeyError:
                if not create_new_genes:
                    continue
                gene = Gene(uniprot_id, only_w_values=True)
                is_new_gene = True
            gene.id_ensg = ensg_id
            is_any_found = 0
            for t in range(len(tissues)):
                tissue = tissues[t]
                prot_value = float(prot_data[i][tissue])
                rna_value = float(rna_data[i][tissue])
                if not self.use_zero_prot:
                    if prot_value == 0.0:
                        continue
                if not self.use_non_zero_prot:
                    if prot_value != 0.0:
                        continue
                if rna_value == 0.0:
                    continue
                is_any_found += 1
                good_data += 1
                measurement_name = '{}_{}'.format(
                    dataset_name,
                    tissue
                )
                gene.rna_measurements[measurement_name] = rna_value
                gene.protein_measurements[measurement_name] = prot_value
            if not is_any_found:
                continue
            if is_new_gene:
                self.genes()[uniprot_id] = gene
        print('{} dataset data uploaded!'.format(dataset_name))
        print('{} good genes experiments added'.format(good_data))
        print('{} genes now'.format(len(self.genes())))
        print('{} rna experiments in data'.format(self.rnaExperimentsCount()))
        print('{} protein experiments in data'.format(self.proteinExperimentsCount()))
        if len(ensg2uniprot_errors):
            print('not found ensg->uniprot::{}'.format(len(ensg2uniprot_errors)))
        return ensg2uniprot_errors

    def viz_sample(
        self,
        gene_id,
        gene2sample,
        outdir,
        layout=(
            'protein_id',
            'rna_id',
            'rna_value',
            'aminoacids',
            'annotations',
        ),
        data2viz=10
    ):
        # if gene_id != 'P35228':
        #     return 0
        dbs = self.databases_alphs
        outd = os.path.join(outdir, 'geneviz', '{}_viz'.format(gene_id))  
        for i, ch in enumerate(gene2sample):
            if np.count_nonzero(ch) < 2:
                return None
        
        ensure_folder(outd)
        for i, ch in enumerate(gene2sample):
            ch_w, ch_h = ch.shape
            vs2viz = data2viz * ch_h
            outp = os.path.join(
                outd,
                '{}_{}.png'.format(gene_id, i)
            )
            ch_f = ch.flatten()
            first_non_zero_chs = [
                i for i in range(len(ch))
                if roundUp(ch[i].sum())
            ]
            first_non_zeros = [
                i for i in range(len(ch_f))
                if roundUp(ch_f[i])
            ]
            last_ch2viz = first_non_zero_chs[0] + data2viz
            v_header = None
            h_header = None
            annotations = False
            if i < 2:
                first_non_zero_chs = [0]
                last_ch2viz = data2viz
                h_header = [x.split('_')[1] for x in self.proteinMeasurementsAlphabet[:data2viz]]
            else:
                h_header = range(first_non_zero_chs[0], last_ch2viz) 
            if last_ch2viz > len(ch):
                raise Exception('viz_sample::not enough data')
            try:
                if layout[i] == 'aminoacids':
                    v_header = Gene.proteinAminoAcidsAlphabet()
                elif layout[i] == 'annotations':
                    raise IndexError
            # db case
            except IndexError:
                annotations = True
                db_name = uniq_nonempty_uniprot_mapping_header()[i-4]
                ch_data = ch_f[first_non_zeros[0]:first_non_zeros[0]+vs2viz]
                db_data_zero = ['0'] * len(ch_data.flatten())
                db_data = self.databases_alphs[db_name][
                    first_non_zeros[0]:first_non_zeros[0]+vs2viz
                ]
                for l in range(len(ch_data)):
                    if roundUp(ch_data[l]):
                        db_data_zero[l] = db_data[l]
                outp = os.path.join(
                    outd,
                    '{}_{}.png'.format(gene_id, db_name)
                )
            m2viz = ch[first_non_zero_chs[0]:last_ch2viz]
            if not annotations:
                m2viz = m2viz.T 
            else:
                v_header = None
                h_header = None
            matrix2heatmap(
                m2viz,
                v_header,
                h_header,
                'rainbow',
                None,
                None,
                out_dir=outp
            )
            print(outp)
        return 0
    
    @staticmethod
    def collate_fn(batch):
        """
        Function for batch creation while train
        """
        batch = list(filter(lambda x: x is not None, batch[1]))
        return torch.utils.data.dataloader.default_collate(batch)
    
    
if __name__ == '__main__':
    train_loader, val_loader = DistGeneDataLoader.createDistLoader(
        './config/gene_expression/train.yaml',
        # './config/gene_expression/train.yaml',
        # './config/gene_expression/gene_train_default.yaml',
        # './config/gene_expression/gene_train_short.yaml',
        rank=0,
        batch_size=10,
        gpus2use=1,
        num_workers=8,
        dataset=None,
        net_config_path='trained/rna2protein_expression_regressor.033/rna2protein_expression_regressor.033_config.json'
    )
    baseloader: DistGeneDataLoader = train_loader.dataset.baseloader
    
    rna_exps_alph = baseloader.rnaMeasurementsAlphabet
    prot_exps_alph = baseloader.proteinMeasurementsAlphabet
    # exit()
    sh_genes = shuffle(list(baseloader.genes().keys()))[0]
    if sh_genes == list(baseloader.genes().keys()):
        raise Exception('che?') 
    for gene_id in sh_genes:
        print(gene_id)
        # continue
        gene = baseloader.genes()[gene_id]
        for experiment in shuffle(rna_exps_alph[:10])[0]:
            sample = baseloader.gene2sample(
                gene_id,
                baseloader.databases_alphs,
                rna_exp_name=experiment,
                protein_exp_name=experiment,
                rna_exps_alphabet=rna_exps_alph,
                protein_exps_alphabet=prot_exps_alph 
            )
            if sample is None:
                # print('{} is none'.format(gene_id))
                continue
            viz = baseloader.viz_sample(
                gene_id,
                sample,
                '/home/ilpech/datasets/tests',
                layout=(
                    'protein_id',
                    'rna_id',
                    'rna_value',
                    'aminoacids',
                    'annotations',
                ),
                data2viz=15
            )
            # if viz is None:
            #     print('{} viz is none'.format(gene_id))
            break
    print('iterating done')
    exit()
            # exit()
    # baseloader.info()
    # databases = uniq_nonempty_uniprot_mapping_header()
    
    # baseloader.sequencesAnalys('bio_dl/tissue29_seqAnalysis.txt')
    # exit()
    databases_alphs = baseloader.databases_alphs
    max_label = norm_shifted_log(baseloader.maxProteinMeasurementInData())
    channels_names = [
        'protein_id',
        'rna_id',
        'rna_value',
        'amino_seq',
        'go',
        'refseq',
        'mim',
        'pubmed',
        'ensembl',
        'pdb',
    ]
    genes_cnt = len(baseloader.genes())
    out_p = 'bio_dl/test/loader/{}_loader.txt'.format(baseloader.creation_time)
    ensure_folder(os.path.dirname(out_p))
    print(f'write in {out_p}')
    exit()
    
    with open(out_p, 'w') as f:
        f.write('created::{} with {} genes\n'.format(baseloader.creation_time, genes_cnt))
        for i, (gene_uid, gene) in enumerate(baseloader.genes().items()):
            print('gene {} of {}'.format(i, genes_cnt))
            f.write('\n+++{}+++\n'.format(gene_uid))
            rna_exp = list(gene.rna_measurements.keys())[0]
            # rna_exp = rna_exps_alph[0]
            prot_value = gene.protein_measurements[rna_exp]
            debug(prot_value)
            print(type(prot_value))
            debug(baseloader.max_label)
            label = norm_shifted_log(
                gene.protein_measurements[rna_exp]
            )/baseloader.max_label
            no_norm_label = norm_shifted_log(
                gene.protein_measurements[rna_exp]
            )
            debug(label)
            debug(no_norm_label)
            denorm_label = denorm_shifted_log(label*baseloader.max_label)
            debug(denorm_label)
            print()
            continue
            exit()
            prot_exp = rna_exp
            sample = baseloader.gene2sample(
                gene_uid,
                databases_alphs,
                rna_exp_name=rna_exp,
                protein_exp_name=prot_exp,
                rna_exps_alphabet=rna_exps_alph,
                protein_exps_alphabet=prot_exps_alph 
            )
            continue
            if sample is None:
                continue
            # print(sample.shape)
            # print(type(sample))
            # print(sample.dtype)
            # exit()
            for j, ch in enumerate(sample):
                f.write('channel::{}::{}\n'.format(channels_names[j], j))
                # print(ch)
                if channels_names[j] == 'rna_value':
                    f.write(f'max::{ch.max()}\n')
                    f.write(f'min::{ch.min()}\n')
                else:
                    f.write(f'nonzero::{np.count_nonzero(ch)}\n')
                # if j == 3:
                #     for row in ch:
                #         print(row)
                
                
            # exit()
    print(f'written in {out_p}')
            

            