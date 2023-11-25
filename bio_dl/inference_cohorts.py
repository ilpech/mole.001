#python3 bio_dl/inference_cohorts.py --cohorts_config config/cohort_gene_expression/cohort_gene_expression.yaml

import os
from bio_dl.rna2protein_train_scheduler import ModelsCohort
from tools_dl.tools import shell
import sys
from subprocess import run, STDOUT, PIPE
import yaml
import json
import argparse

def net_dir_from_metric_path(metric_path):
    head = metric_path
    tail = None
    while tail != 'out':
        head, tail = os.path.split(head)
    return head

def determining_file_type_from_name(file_name):
    file_name_parts = file_name.split('_')
    file_type = file_name_parts[0]
    for part in file_name_parts[1:]:
        if part[0].isdigit():
            break
        else:
            file_type += f'_{part}'
    return file_type

def cohort_inference(path2cohort, batch_size):
    terminal_str2find = 'data written to '
    inferensed_nets = []
    metric_files = []
    params_dir, cohort_name = os.path.split(path2cohort)

    models_cohort = ModelsCohort(
        cohort_name,
        params_dir=params_dir,
        cohort_size=1,
        min_epochs2finish=1,
        stable_validataion_genes_path=None,
        train=False,
        train_config_path=None,
        isdebug=True
    )

    #python3 bio_dl/inference_gene_regressor.py --isdebug 0 --net_dir trained/cohorts/rna2protein_tissue29.ResNet50V2.c001/rna2protein_tissue29.ResNet50V2.c001.003 --epoch 2 --data_config config/gene_expression/train_tissue29_res50v2.yaml --only_val 1 --write_norm False --batch_size 8 --cross_val_genes trained/cohorts/rna2protein_tissue29.ResNet50V2.c001/rna2protein_tissue29.ResNet50V2.c001.003_val_scheduler_genes.json

    for model_name, model in models_cohort.models.items():
        is_model_linear = False
        python_file = 'inference_gene_regressor.py'
        if 'BioPerceptron' in model_name:
            python_file = 'inference_gene_regressor_linear.py'
            is_model_linear = True

        if 'tissue29' in model_name:
            data_config_path = 'config/gene_expression/train_tissue29.yaml'
            if is_model_linear:
                data_config_path = 'config/gene_expression/train_linear_tissue29.yaml'
        if 'nci60' in model_name:
            data_config_path = 'config/gene_expression/train_nci60.yaml'
            if is_model_linear:
                data_config_path = 'config/gene_expression/train_linear_nci60.yaml'

        cross_val = os.path.join(
            '{}'.format(path2cohort),
            '{}_val_scheduler_genes.json'.format(model_name)
        )
        
        # concatenate command for model inference
        inference_command2shell = 'python3 bio_dl/{} \
            --isdebug 0\
            --net_dir {}\
            --epoch {}\
            --data_config {}\
            --only_val 1\
            --write_norm False\
            --batch_size {}\
            --cross_val_genes {}'.format(
                python_file,
                model.net_dir(),
                model.bestEpochFromLog()[1],
                data_config_path,
                batch_size,
                cross_val
            )
        print(f'    Start inference for {model_name}...')
        # print(inference_command2shell)
        # write inference output to inference_output variable
        inference_output = run(
            inference_command2shell.split(), 
            stdout=PIPE, 
            stderr=STDOUT, 
            text=True
        )
        
        # parse inference_output
        inference_output_rows = inference_output.stdout.split('\n')
        for row in inference_output_rows:
            if terminal_str2find in row:
                path2metrics = str(row.replace(terminal_str2find, ''))
                net_name = os.path.splitext(
                    os.path.basename(path2metrics)
                )[0]
                
                inferensed_nets.append(net_name)
                metric_files.append(path2metrics)

                print(f'    Inference finished for {model_name}')
                print(f'    Result saved to {path2metrics}')
                print('   ', 30*'=')
                
    return inferensed_nets, metric_files

def metric_inference_cohort(path2cohort, inferensed_nets, metric_files, default_metric_config=None):
    terminal_str2find = 'log file is opened at '
    metric_files_dict = {}
    if default_metric_config is None:
        default_metric_config = 'config/metric_inferenced/metric_inferenced_cohort.yaml'
        
    with open(default_metric_config, 'r') as f:
        metric_inferenced_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    arr2file = []
    for i in range(len(metric_files)):
        # arr2file.append(f'#{inferensed_nets[i]}')
        arr2file.append(metric_files[i])
    metric_inferenced_config['metric_files'] = arr2file
    
    
    metric_file = os.path.join(
        path2cohort,
        f'{os.path.basename(path2cohort)}.yaml'
    )
    with open(metric_file, 'w') as outfile:
        yaml.dump(
            metric_inferenced_config, 
            outfile, 
            default_flow_style=False
        )
        
    ##python3 bio_dl/metric_inferenced.py --config_path config/metric_inferenced/metric_inferenced.yaml

    command2metric = f'python3 bio_dl/metric_inferenced.py\
        --config_path {metric_file}'
        
    metrics_inference_output = run(
        command2metric.split(), 
        stdout=PIPE, 
        stderr=STDOUT, 
        text=True
    )
    metrics_inference_out = metrics_inference_output.stdout.split('\n')
    for row in metrics_inference_out:
        if terminal_str2find in row:
            log_path = row.replace(terminal_str2find, '')
            net_dir = net_dir_from_metric_path(log_path)
            net_name = os.path.basename(net_dir)
            file_type = determining_file_type_from_name(
                os.path.basename(log_path)
            )
            if net_name not in metric_files_dict.keys():
                metric_files_dict[net_name] = {}
            metric_files_dict[net_name][file_type] = log_path
    
    return metric_files_dict
        
def cohort_models_and_metrics_inference(path2cohort, batch_size):
    cohort_name = os.path.basename(path2cohort)
    print(f'Start inference process for {cohort_name} cohort...\n')
    inferensed_nets, metric_files = cohort_inference(path2cohort=path2cohort, batch_size=batch_size)
    print(inferensed_nets, metric_files)
    # inferensed_nets = ['rna2protein_nci60.ResNet26V2.c001.001_0002.val_metrics', 'rna2protein_nci60.ResNet26V2.c001.002_0001.val_metrics']
    # metric_files = [
    #     '/home/gerzog/repositories/mole.001/trained/cohorts2test/rna2protein_nci60.ResNet26V2.c001/rna2protein_nci60.ResNet26V2.c001.001/out/2023.11.24.19.32.59/rna2protein_nci60.ResNet26V2.c001.001_0002.val_metrics.txt', 
    #     '/home/gerzog/repositories/mole.001/trained/cohorts2test/rna2protein_nci60.ResNet26V2.c001/rna2protein_nci60.ResNet26V2.c001.002/out/2023.11.24.19.36.03/rna2protein_nci60.ResNet26V2.c001.002_0001.val_metrics.txt'                 
    # ]
    metric_cohort_dict = metric_inference_cohort(
        path2cohort=path2cohort,
        inferensed_nets=inferensed_nets,
        metric_files=metric_files
    )
    path2cohort_json = os.path.join(
        path2cohort,
        f'{cohort_name}_metrics.json'
    )
    with open(path2cohort_json, 'w') as f:
        json.dump(metric_cohort_dict, f, indent=4)
    print(f'Metrics paths for {cohort_name} saved to {path2cohort_json}\n')


def cohorts_inference_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohorts_config')
    
    opt = parser.parse_args()
    path2config = opt.cohorts_config
    with open(path2config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    cohorts_paths = config['cohorts_paths']
    batch_size = config['batch_size']
    
    for cohort_path in cohorts_paths:
        cohort_models_and_metrics_inference(cohort_path, batch_size)
        
    
if __name__ == '__main__':
    cohorts_inference_main()
    




