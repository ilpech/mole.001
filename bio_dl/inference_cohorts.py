import os
from bio_dl.rna2protein_train_scheduler import ModelsCohort
from tools_dl.tools import shell
import sys
from subprocess import run, STDOUT, PIPE
import yaml


def cohort_inference(path2cohort):
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
        # print(model_name, model.bestEpochFromLog())
        if 'tissue29' in model_name:
            data_config_path = 'config/gene_expression/train_tissue29.yaml'
        if 'nci60' in model_name:
            data_config_path = 'config/gene_expression/train_nci60.yaml'

        cross_val = os.path.join(
            '{}'.format(path2cohort),
            '{}_val_scheduler_genes.json'.format(model_name)
        )
        
        # concatenate command for model inference
        inference_command2shell = 'python3 bio_dl/inference_gene_regressor.py \
            --isdebug 0\
            --net_dir {}\
            --epoch {}\
            --data_config {}\
            --only_val 1\
            --write_norm False\
            --batch_size 32\
            --cross_val_genes {}'.format(
                model.net_dir(),
                model.bestEpochFromLog()[1],
                data_config_path,
                cross_val
            )
        
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
                path2metrics = row.replace(terminal_str2find, '')
                net_name = os.path.splitext(
                    os.path.basename(path2metrics)
                )[0]
                
                inferensed_nets.append(net_name)
                metric_files.append(path2metrics)
                
                print(f'Inference finished for {net_name}')
                
    return inferensed_nets, metric_files

def metric_inference_cohort(path2cohort, inferensed_nets, metric_files, default_metric_config=None):
    if default_metric_config is None:
        default_metric_config = 'config/metric_inferenced/metric_inferenced_cohort.yaml'
        
    with open(default_metric_config, 'r') as f:
        metric_inferenced_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    arr2file = []
    for i in range(len(metric_files)):
        arr2file.append(f'#{inferensed_nets[i]}')
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

    command2metric = f'bio_dl/metric_inferenced.py\
        --config_path {metric_file}'
        
    metrics_inference_output = run(
        command2metric.split(), 
        stdout=PIPE, 
        stderr=STDOUT, 
        text=True
    )
    print(metrics_inference_output)
    
    
    
                
        
                
        
    # with open('test_cohort_inference.txt', 'w') as sf:
    #     sf.writelines(metric_files)
        
        
if __name__ == '__main__':
    # path2cohort = 'trained/cohorts/rna2protein_nci60.BioPerceptrone.mole_c001'
    # path2cohort = 'trained/cohorts/rna2protein_tissue29.BioPerceptrone.mole_c003'
    # path2cohort = 'trained/cohorts/rna2protein_nci60.BioPerceptrone.mole_c003'
    # path2cohort = '/home/ilpech/repositories/mole.001/trained/cohorts/rna2protein_nci60.ResNet26V2.c001'
    # path2cohort = 'trained/cohorts2test/rna2protein_tissue29.BioPerceptrone.mole_c003'
    path2cohort = 'trained/cohorts2test/rna2protein_nci60.ResNet26V2.c001'
    inferensed_nets, metric_files = cohort_inference(path2cohort=path2cohort)
    print(inferensed_nets)
    print(metric_files)
    
    metric_inference_cohort(
        path2cohort=path2cohort,
        inferensed_nets=inferensed_nets,
        metric_files=metric_files
    )




