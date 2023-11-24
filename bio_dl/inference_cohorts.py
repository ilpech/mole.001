import os
from bio_dl.rna2protein_train_scheduler import ModelsCohort
from tools_dl.tools import shell


def cohort_inference(path2cohort):
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
        if 'tissue29' in model_name:
            data_config_path = 'config/gene_expression/train_tissue29.yaml'
        if 'nci60' in model_name:
            data_config_path = 'config/gene_expression/train_tissue29.yaml'

        cross_val = os.path.join(
            '{}'.format(path2cohort),
            '{}_val_scheduler_genes.json'.format(model_name)
        )
        
        command2shell = 'python3 bio_dl/inference_gene_regressor.py \
            --isdebug 0\
            --net_dir {}\
            --epoch {}\
            --data_config {}\
            --only_val 1\
            --write_norm False\
            --batch_size 8\
            --cross_val_genes {}'.format(
                model.net_dir(),
                model.bestEpochFromLog()[1],
                data_config_path,
                cross_val
            )
        
        os.system(command2shell)
        # shell(command2shell)
        # print(command2shell)
        
        
if __name__ == '__main__':
    # path2cohort = 'trained/cohorts/rna2protein_nci60.BioPerceptrone.mole_c001'
    path2cohort = 'trained/cohorts/rna2protein_tissue29.BioPerceptrone.mole_c003'
    # path2cohort = 'trained/cohorts/rna2protein_nci60.BioPerceptrone.mole_c003'
    path2cohort = '/home/ilpech/repositories/mole.001/trained/cohorts/rna2protein_nci60.ResNet26V2.c001'
    cohort_inference(path2cohort=path2cohort)




