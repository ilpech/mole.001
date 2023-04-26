#python3 bio_dl/gene_observer.py --metric_cfg config/metric_inferenced/metric_inferenced.yaml --data_cfg config/gene_expression/train.yaml

import os
import sys
import time
import argparse
from varname.helpers import debug

from bio_dl import metric_inferenced
from torch_dl.dataloader.gene_data_loader import DistGeneDataLoader
from bio_dl.gene import Gene

from tools_dl.tools import (
    curDateTime
)

class GeneObserver:
    def __init__(
        self,
        metric_config_path,
        dataloader_config_path
    ):
        print('=' * 90)
        print('GeneObserver::run...')
        self.start_t = time.time()
        self.start_time = curDateTime()
        self.metric_config_path = metric_config_path
        self.dataloader_config_path = dataloader_config_path
        print('\n' + '*' * 30)
        print('GeneObserver::dataloader...')
        print('*' * 30)
        self.dataloader = DistGeneDataLoader(
            config_path=dataloader_config_path,
            net_config_path='trained/rna2protein_expression_regressor.033/rna2protein_expression_regressor.033_config.json',
            use_net_experiments=False
        )
        print('\n' + '*' * 30)
        print('GeneObserver::metrics...')
        print('*' * 30)
        self.metric_inferenced = metric_inferenced.MetricInferenced(
            self.metric_config_path
        )
        print('\n' + '*' * 30)
        debug(len(self.dataloader.genes()))
        print('*' * 70)
        debug(self.metric_inferenced.net_names())
        print('*' * 70)
        debug(self.metric_inferenced.datasets_names())
        print('*' * 70)
        print('*-' * 35 + '*')
        print('*' * 70 + '\n')
        process_time = time.time() - self.start_t
        self.genes2values()
        print('=' * 90 + '\n')
        print('GeneObserver::done in {:.3f} sec'.format(process_time))
    
    def genes2values(self):
        # !!! add mappings sample
        print('genes2values::...')
        for net_name in self.metric_inferenced.net_names():
            for dataset_name in self.metric_inferenced.datasets_names():
                net_data = self.metric_inferenced.metrics()[net_name][dataset_name]
                exps2process = [
                    x for x in net_data.keys() 
                    if 'uid.' in x and dataset_name not in x
                ]
                for gene_name in exps2process:
                    uniprot_id = gene_name.split('.')[-1]
                    print('#' * 35)
                    debug(uniprot_id)
                    
                    gene: Gene = self.dataloader.genes()[uniprot_id]

                    uniprot_api_data = gene.apiData()
                    # print(uniprot_api_data) 
                    annotation_mappings = self.dataloader.dataFromMappingDatabase(
                        db_name='GO',
                        gene_name=uniprot_id
                    )
                    print(annotation_mappings)
                    seq = gene.apiSequence()
                    print(seq)
                    for gene_exp in net_data[gene_name].keys():
                        rna_from_loader = gene.get_RNA_experiment_value(
                            gene_exp,
                            use_log_norm=self.metric_inferenced.use_norm
                        )
                        protein_from_loader = gene.get_protein_experiment_value(
                            gene_exp,
                            use_log_norm=self.metric_inferenced.use_norm
                        )
                        exp_values = net_data[gene_name][gene_exp]
                        # protein_from_metrics, protein_pred_from_metrics = (
                        #     [x[0] for x in exp_values], 
                        #     [x[1] for x in exp_values]
                        # )
                        protein_gt_from_metrics, protein_prediction_from_metrics = (
                            exp_values[0][0], 
                            exp_values[0][1]
                        )
                        print('-' * 35)
                        print('{}::{}'.format(uniprot_id, gene_exp))
                        debug(self.metric_inferenced.use_norm) 
                        debug(rna_from_loader)
                        debug(protein_gt_from_metrics)
                        debug(protein_prediction_from_metrics)
                        debug(protein_from_loader)
                    return

if __name__ == '__main__':
    # metric_cfg = 'config/metric_inferenced/metric_inferenced.yaml'
    # # data_cfg = 'config/gene_expression/train.yaml'
    # data_cfg = 'config/gene_expression/train_tissue29.yaml'
    # # data_cfg = 'config/gene_expression/train_nci60.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_cfg', type=str)
    parser.add_argument('--data_cfg', type=str)
    opt = parser.parse_args()
    obs = GeneObserver(
        opt.metric_cfg,
        opt.data_cfg
    ) 