#python3 bio_dl/inference_gene_regressor.py --isdebug 0 --net_dir trained/rna2protein_tissue29.ResNet34V2.005 --epoch 4 --data_config config/gene_expression/train_tissue29.yaml --only_val 1 --write_norm False
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

import argparse

run_start = curDateTime()

parser = argparse.ArgumentParser()
parser.add_argument('--isdebug', type=int)
parser.add_argument('--net_dir', type=str)
parser.add_argument('--data_config', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--only_val', type=int, default=0)
parser.add_argument('--write_norm', type=boolean_string, default=False)
opt = parser.parse_args()
epoch = opt.epoch
net_dir_p = opt.net_dir
config_p = find_files(net_dir_p, '_config', abs_p = True)[0]
data_config_p = opt.data_config
isdebug = opt.isdebug
only_val = opt.only_val
write_norm = opt.write_norm

data_loader = DistGeneDataLoader(
    data_config_p, 
    net_config_path=config_p
)
with open(data_config_p, 'r') as f:
    data_config_ = yaml.load(f)
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
if 'V2' not in config['model_name']:
    net = RegressionResNet2d34(
        num_channels=10, 
        channels_width_modifier=width_factor
    )
else:
    if num_layers == 34:
        net = RegressionResNet2dV2_34(
            num_channels=10, 
            channels_width_modifier=width_factor
        )
    elif num_layers == 50:
        net = RegressionResNet2dV2_50(
            num_channels=10, 
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

genes_size = len(data_loader.genes())
if only_val:
    print('only validation genes mode applied')
    val_genes = [
        x for x in val_config_genes if x in list(data_loader.genes().keys())
    ]
    out_path = '{}/{}_{:04d}.val_metrics.txt'.format(
        out_dir, 
        net_name, 
        epoch
    )
    genes_size = len(val_genes)
print('genes', genes_size)
print('data write to', out_path)
print('process...')
# create outfile and write header
with open(out_path, 'w') as fv:
    fv.write('uid\trna_expt\trna_value\tprot_expt\tprot_value\tpredicted_prot_value\n')
j = 0
inference_start = time.time()
sample_inference_ts = []
model_inference_ts = []
for uid, gene in data_loader.genes().items():
    if only_val and uid not in val_genes:
        continue
    j += 1
    # if j < 9000 or j > 10000:
    #     continue
    sys.stdout.write('\rgene {} of {}'.format(j, genes_size))
    sys.stdout.flush()
    for rna_exp in rna_alph:
        prot_exp = rna_exp 
        sample_inference = time.time()
        try:
            sample = data_loader.gene2sample(
                uid,
                data_loader.databases_alphs,
                rna_exp,
                prot_exp,
                rna_alph,
                prot_alph
            )
        except Exception as e:
            print(str(e))
            continue
        if sample is None:
            continue
        ens = data_loader.uniprot2ensg(uid)
        if not len(ens):
            ens = ''
        else:
            ens = ens[0]
        batch2inf = torch.Tensor(sample)
        if 'gpu' in ctx:
            batch2inf = batch2inf.cuda()
        batch2inf = batch2inf.unsqueeze(0)
        # rna_expt_id = int(batch2inf[0][1].argmax(0)[0])
        rna_value = gene.rna_measurements[rna_exp]
        if prot_exp not in gene.protein_measurements:
            continue
        prot_value = gene.protein_measurements[prot_exp]
        model_run_start = time.time()
        out = model(batch2inf).detach().cpu()
        model_inference_ts.append(
            time.time() - model_run_start
        )
        out = out.detach().cpu()
        if not write_norm:
            out = denorm_shifted_log(out[0]*max_label)
        else:
            prot_value = norm_shifted_log(prot_value)/max_label
        sample_inference_ts.append(
            time.time() - sample_inference
        )
        lo = [
            uid, 
            rna_exp, 
            rna_value, 
            prot_exp, 
            prot_value, 
            out
        ]
        with open(out_path, 'a') as fv:
            fv.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                lo[0],lo[1],
                lo[2],lo[3],
                lo[4],lo[5]
            ))
            
inference_time = time.time() - inference_start
sample_inference_ts = np.array(sample_inference_ts)
model_inference_ts = np.array(model_inference_ts)
print('\ndata written to', out_path)
print('======================')
print('inference done in {:.3f} sec'.format(inference_time))
print('average sample inference t {:.3f} sec'.format(
    sample_inference_ts.mean())
)
print('average model inference t {:.3f} sec'.format(
    model_inference_ts.mean())
)
print('======================')