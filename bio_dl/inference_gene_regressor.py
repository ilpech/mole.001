#python3 bio_dl/inference_gene_regressor.py --isdebug 0 --net_dir trained/rna2protein_tissue29.ResNet34V2.005 --epoch 4 --data_config config/gene_expression/train_tissue29.yaml --only_val 1 --write_norm False --batch_size 100
#cross val
#python3 bio_dl/inference_gene_regressor.py --isdebug 0 --net_dir trained/cohorts/rna2protein_tissue29.ResNet50V2.c001/rna2protein_tissue29.ResNet50V2.c001.003 --epoch 2 --data_config config/gene_expression/train_tissue29_res50v2.yaml --only_val 1 --write_norm False --batch_size 8 --cross_val_genes trained/cohorts/rna2protein_tissue29.ResNet50V2.c001/rna2protein_tissue29.ResNet50V2.c001.003_val_scheduler_genes.json
import os
import sys
import numpy as np
import torch
from torch_dl.tools.tools_torch import isGPU
from torch_dl.model.model import TorchModel
from torch_dl.dataloader.gene_data_loader import DistGeneDataLoader
from torch_dl.model.regression_model import (
    RegressionResNet2d18,
    RegressionResNet2d34,
    RegressionResNet2dV2_34,
    RegressionResNet2dV2_50
)
import argparse
import yaml
import json
from tools_dl.tools import (
    debug,
    denorm_shifted_log,
    ensure_folder,
    curDateTime,
    find_files,
    boolean_string,
    norm_shifted_log
)
import time
from bio_dl.gene_mapping import uniq_nonempty_uniprot_mapping_header

import argparse

run_start = curDateTime()

parser = argparse.ArgumentParser()
parser.add_argument('--isdebug', type=int)
parser.add_argument('--net_dir', type=str)
parser.add_argument('--data_config', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--only_val', type=int, default=0)
parser.add_argument('--write_norm', type=boolean_string, default=False)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--cross_val_genes', type=str, default='')
opt = parser.parse_args()
epoch = opt.epoch
net_dir_p = opt.net_dir
config_p = find_files(net_dir_p, '_config', abs_p = True)[0]
data_config_p = opt.data_config
isdebug = opt.isdebug
only_val = opt.only_val
write_norm = opt.write_norm
batch_size = opt.batch_size
cross_val_genes = opt.cross_val_genes

dataset_config = DistGeneDataLoader.opt_from_config(data_config_p)
dataset = DistGeneDataLoader(
    dataset_config,
    net_config_path=config_p,
    use_net_experiments=True,
    config_dict=dataset_config
)
if len(cross_val_genes):
    with open(cross_val_genes, 'r') as f:
        scheduler_data = json.load(f)
    val_config_genes = scheduler_data['genes2cross_val']
    print('USING CROSS VAL GENES, ONLY VAL MODE')
    dataset.genes2train = []
    dataset.genes2val = val_config_genes
    debug(val_config_genes)
    dataset._valexps2indxs = []
    dataset._exps2indxs = []
    dataset._warmupExps()
    only_val = True
train_data_loader, val_data_loader = DistGeneDataLoader.createDistLoader(
    data_config_p, 
    rank=0, 
    batch_size=batch_size, 
    gpus2use=1,
    num_workers=8,
    dataset=dataset,
    net_config_path=config_p,
    use_net_experiments=True,
    inference_mode=True
)

with open(data_config_p, 'r') as f:
    # data_config_ = yaml.load(f)
    data_config_ = yaml.load(f, Loader=yaml.SafeLoader)
    data_config = data_config_['data']

params_path = os.path.split(config_p)[0]

with open(config_p, 'r') as f:
    config = json.load(f)
net_name = config['net_name']
width_factor = config['width_factor']
num_layers = config['num_layers']
out_dir = os.path.join(params_path, 'out', run_start)
ensure_folder(out_dir)
debug(config['model_name'])
net_num_channels = len(uniq_nonempty_uniprot_mapping_header()) + 4
if 'V2' not in config['model_name']:
    net = RegressionResNet2d34(
        num_channels=net_num_channels, 
        channels_width_modifier=width_factor
    )
else:
    if num_layers == 34:
        net = RegressionResNet2dV2_34(
            num_channels=net_num_channels, 
            channels_width_modifier=width_factor
        )
    elif num_layers == 50:
        net = RegressionResNet2dV2_50(
            num_channels=net_num_channels, 
            channels_width_modifier=width_factor
        )
model = TorchModel(
    os.path.split(params_path)[0], 
    net_name, 
    epoch,
    model=net,
    load=True
)
model._model.eval()
ctx = isGPU(isdebug)
if 'gpu' in ctx:
    model._model = model._model.cuda()
print('config', config_p)
rna_alph = config['rna_exps_alphabet']
inference_shape = config['inference_shape']
max_label = float(config['denorm_max_label'])
prot_alph = config['protein_exps_alphabet']
databases_names_alph = config['databases']
val_config_genes = config['genes2val']

sample_path = '{}/{}_{:04d}.metrics.txt'
if write_norm:
    sample_path = '{}/{}_{:04d}.norm_metrics.txt'
    
out_path = sample_path.format(
    out_dir, 
    net_name, 
    epoch
)

train_genes_size = len(train_data_loader.dataset.baseloader.genes2train)
val_genes_size = len(val_data_loader.dataset.baseloader.genes2val)
genes_size = train_genes_size + val_genes_size
if only_val:
    print('only validation genes mode applied')
    out_path = '{}/{}_{:04d}.val_metrics.txt'.format(
        out_dir, 
        net_name, 
        epoch
    )
    genes_size = val_genes_size
print('genes', genes_size)
print('data write to', out_path)
print('process...')

# create outfile and write header
with open(out_path, 'w') as fv:
    fv.write('uid\trna_expt\trna_value\tprot_expt\tprot_value\tpredicted_prot_value\n')
j = 0
inference_start = time.time()
batch_inference_ts = []
batch_postprocess_ts = []
calculated_sample_inference_ts = []

dataloaders2process = [train_data_loader, val_data_loader]
if only_val:
    dataloaders2process = [val_data_loader]
    
for data_loader_i, data_loader in enumerate(dataloaders2process):
    batch_passed = 0
    batch_total_count = len(data_loader)
    for batch, prot_vals, uids, rna_ids, prot_ids in data_loader:
        batch_inference = time.time()
        if 'gpu' in ctx:
            batch = batch.cuda()
        out = model(batch)
        batch_inference_ts.append(
            time.time() - batch_inference
        )
        batch_postprocess = time.time()
        out = out.detach().cpu()
        out = out.tolist()
        norm_pred_prot_vals = [x[0] for x in out]
        rna_exps = [
            data_loader.dataset.baseloader.rnaMeasurementsAlphabet[x] for x in rna_ids
        ]
        norm_rna_values = [
            data_loader.dataset.baseloader._genes[uids[x]].get_RNA_experiment_value(
                rna_exps[x], use_log_norm=True
            ) for x in range(len(uids))
        ]
        prot_exps = [
            data_loader.dataset.baseloader.proteinMeasurementsAlphabet[x] for x in prot_ids
        ]
        norm_prot_vals = [
            data_loader.dataset.baseloader._genes[uids[x]].get_protein_experiment_value(
                prot_exps[x], use_log_norm=True
            ) for x in range(len(uids))
        ]
        rna_values2save = norm_rna_values
        prot_values2save = norm_prot_vals
        pred_prot_values2save = norm_pred_prot_vals
        
        if not write_norm:
            denorm_pred_prot_vals = [denorm_shifted_log(x) for x in norm_pred_prot_vals]
            denorm_rna_values = [
                data_loader.dataset.baseloader._genes[uids[x]].get_RNA_experiment_value(
                    rna_exps[x], use_log_norm=False
                ) for x in range(len(uids))
            ]
            denorm_prot_values = [
                data_loader.dataset.baseloader._genes[uids[x]].get_protein_experiment_value(
                    prot_exps[x], use_log_norm=False
                ) for x in range(len(uids))
            ]
            
            rna_values2save = denorm_rna_values
            prot_values2save = denorm_prot_values
            pred_prot_values2save = denorm_pred_prot_vals
        
        for i in range(len(uids)):
            lo = [
                uids[i], 
                rna_exps[i], 
                rna_values2save[i], 
                prot_exps[i], 
                prot_values2save[i], 
                pred_prot_values2save[i]
            ]
            with open(out_path, 'a') as fv:
                fv.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    lo[0],lo[1],
                    lo[2],lo[3],
                    lo[4],lo[5]
                ))
                
        batch_postprocess_ts.append(
            time.time() - batch_postprocess
        )
        calculated_sample_inference = (
            (time.time() - batch_inference) / len(uids)
        )
        calculated_sample_inference_ts.append(calculated_sample_inference)
        
        batch_passed += 1
        
        sys.stdout.write('\rDL [ {}/{}] batch {} of {}'.format(
            data_loader_i+1, len(dataloaders2process), 
            batch_passed, batch_total_count
        ))
        sys.stdout.flush()
        
inference_time = time.time() - inference_start

batch_inference_ts = np.array(batch_inference_ts)
batch_postprocess_ts = np.array(batch_postprocess_ts)
calculated_sample_inference_ts = np.array(calculated_sample_inference_ts)

print('\ndata written to', out_path)
print('======================')
print('inference done in {:.3f} sec'.format(inference_time))
print('average calculated sample inference t {:.3f} sec'.format(
    calculated_sample_inference_ts.mean())
)
print('average batch [ bs:: {} ] inference t {:.3f} sec'.format(
    batch_size, batch_inference_ts.mean())
)
print('average batch postprocess t {:.3f} sec'.format(
    batch_postprocess_ts.mean())
)
print('total batch postprocess t {:.3f} sec'.format(
    batch_postprocess_ts.sum())
)
print('======================')