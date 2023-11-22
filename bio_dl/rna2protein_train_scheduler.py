# python3 bio_dl/rna2protein_train_scheduler.py --config config/gene_expression/train_cohorts.yaml --isdebug False
import os
from torch_dl.train.train_gene_regression import DistGeneExpressionTrainer
from torch_dl.dataloader.gene_data_loader import DistGeneDataLoader
from torch_dl.model.model import TorchModel
from typing import List, Dict
from varname.helpers import debug
import numpy as np
import json

from tools_dl.tools import (
    debug,
    str2bool, 
    roundUp,
    norm_shifted_log,
    denorm_shifted_log,
    ensure_folder,
    curDateTime,
    cp_r,
    ls_dir,
    list2chunks,
    flat_list
)
import yaml
from varname.helpers import debug
import argparse

# 1. берем базовый трейн конфиг
# 2. делаем конфиг с распиcанием обучений
#    (имя группы сетей, сколько эпох в обучении, 
#    есть ли настройки графика и прочего, так для каждой группы)
# 3. делаем проверку внутри деревяшки и отдельную папку под когорты моделей
# 4. идем по очереди расписания и проверяем есть ли нужна эпоха, если сохраняем последняю точку остановки
# 5. запускаем обучение с последней найденной точки остановки когорты
# 6. посмотреть, как работает с даталоадером трейн скрипт, возможно нужно ему дать подсунуть даталоадер, чтобы один раз зарядить его для когорты
# 7. разделить гены на трейн и валидацию на уровне файла планирования экспериментов

class ModelsCohort:
    def __init__(
        self,
        cohort_name,
        params_dir,
        cohort_size,
        min_epochs2finish,
        stable_validataion_genes_path,
        train=False,
        train_config_path=None,
        isdebug=False
    ):
        self.isdebug = isdebug
        self.cohort_name = cohort_name
        self.params_dir = params_dir
        self.cohort_size = cohort_size
        self.models: Dict[str, TorchModel] = {}
        self.min_epochs2finish = min_epochs2finish
        self.new_cohort = False
        self.train = train
        self.train_config_path = train_config_path
        self.stable_validataion_genes_path = stable_validataion_genes_path
        try:
            self.loadTorchModels()
        except NotADirectoryError:
            self.new_cohort = True
        if self.train:
            print(f'ModelsCohort::start training process...')
            self.cohortTrain()

    
    def cohortDir(self):
        return os.path.join(
            self.params_dir, self.cohort_name
        )

    def cohortNetNames(self):
        return ls_dir(self.cohortDir())
    
    def generateCohortNetNames(self):
        return ['{}.{:03d}'.format(self.cohort_name, i+1) for i in range(self.cohort_size-len(self.models))]
    
    def loadTorchModels(
        self
    ):
        for net_name in self.cohortNetNames():
            self.models[net_name] = TorchModel(
                self.cohortDir(),
                net_name,
                epoch=0,
                model=None,
                load=False,
                ctx='cpu'
            )
            print(f'loadTorchModels:: model {net_name} loaded, last epoch {self.models[net_name].lastEpoch()}')
        if len(self.models):
            print(f'loadTorchModels::{len(self.models)} models loaded to cohort')

    def cohortTrainQueue(self):
        trainQueue = {}
        for model_name, model in self.models.items():
            last_epoch = model.lastEpoch()
            if not last_epoch:
                trainQueue[model_name] = (-1, self.min_epochs2finish)
            elif last_epoch < self.min_epochs2finish:
                remained_epochs = self.min_epochs2finish - last_epoch
                trainQueue[model_name] = (last_epoch, remained_epochs)
        for model_name in self.generateCohortNetNames():
            if model_name not in trainQueue:
                trainQueue[model_name] = (-1, self.min_epochs2finish)
        return trainQueue
    
    def cohortTrain(self):
        queue = self.cohortTrainQueue()
        if not len(queue):
            raise Exception(f'no queue in {self.cohort_name}')
        if not self.isdebug:
            ensure_folder(self.cohortDir())
            print(f'ModelsCohort::creating new cohort dir {self.cohortDir()}')
        dataset_config = DistGeneDataLoader.opt_from_config(self.train_config_path)
        data_settings = dataset_config['data']
        train_settings = dataset_config['train']
        use_finetune_experiments = train_settings['use_finetune_experiments']
        use_finetune_databases = train_settings['use_finetune_databases']
        finetune_net_name = train_settings['finetune_net_name']
        finetune_params_dir = train_settings['finetune_params_dir']
        net_config_dir = None
        if use_finetune_databases or use_finetune_experiments:
            net_config_dir=os.path.join(
                finetune_params_dir, 
                finetune_net_name
            )
        data_settings['splitGeneFromModelConfig'] = self.stable_validataion_genes_path
        data_settings['ifSplitPercentToTrain'] = False
        cohort_dataset = DistGeneDataLoader(
            self.train_config_path,
            net_config_path=net_config_dir,
            use_net_experiments=use_finetune_experiments,
            use_net_databases=use_finetune_databases,
            # read_genes_mapping=True,
            # read_genes_mapping=False,
            config_dict=dataset_config
        )
        train_genes = cohort_dataset.genes2train
        cohort_chunks = np.array_split(train_genes, self.cohort_size)
        if not self.isdebug:
            cp_r(self.stable_validataion_genes_path, self.cohortDir())
        for i, (model_name, epochs_interval) in enumerate(queue.items()):
            model_cross_validation_genes = cohort_chunks[i]
            model_train_genes = flat_list(
                [cohort_chunks[j] for j in range(len(cohort_chunks)) if j != i]
            )
            print(f'train genes: {len(model_train_genes)}')
            print(f'stabe val genes: {len(cohort_dataset.genes2val)}')
            print(f'cross val genes: {len(model_cross_validation_genes)}')
            cohort_dataset.genes2train = model_train_genes
            model_cross_val_genes_path = os.path.join(
                self.cohortDir(),
                model_name
            ) + '_val_scheduler_genes.json'
            model_cross_validation_data = {
                'genes2val': list(cohort_dataset.genes2val),
                'genes2cross_val': list(model_cross_validation_genes),
                'genes2train': list(model_train_genes),
            } 
            with open(model_cross_val_genes_path, 'w') as f:
                json.dump(model_cross_validation_data, f, indent=4)
            print(f'model cross validation genes written {model_cross_val_genes_path}')
            last_epoch = epochs_interval[0]
            remained_epochs = epochs_interval[1]
            train_settings['epochs'] = self.min_epochs2finish
            train_settings['params_dir'] = self.cohortDir()
            train_settings['net_name'] = model_name
            if remained_epochs:
                print(f'cohortTrain::training with model {model_name}, remained {remained_epochs} epochs')
                trainer = DistGeneExpressionTrainer(
                    self.train_config_path,
                    isdebug=False,
                    config_dict=dataset_config,
                    force_run=False
                )
                trainer.train_loop(
                    dataset=cohort_dataset
                )
        print(f'cohort {self.cohortDir()} trained succesfully')
        return True

class TrainScheduler:
    def __init__(
        self, 
        config_path,
        isdebug=False
    ):
        self.isdebug = isdebug
        self.config_path = config_path
        with open(self.config_path) as f:
            self.config = yaml.load(f, yaml.FullLoader)
            print(f'TrainScheduler::created from config {config_path}')
        self.config_cohortsToTrain = self.config['cohortsToTrain']
        self.cohortsToTrain = self.config_cohortsToTrain.keys()
        self.cohorts: Dict[str, ModelsCohort] = {}
        self.loadCohorts()
        

    def loadCohorts(self):
        for cohort_name in self.cohortsToTrain:
            cohort_settings = self.config_cohortsToTrain[cohort_name]
            train_config = cohort_settings['baseConfigPath']
            models_in_cohort = cohort_settings['crossValidation']['modelsInCohort']
            genes_percent2stable_val = cohort_settings['crossValidation']['stableValidationPercent']
            genes_stable_val_config = cohort_settings['crossValidation']['stableValidationFromConfig']
            min_epochs2finish = cohort_settings['minEpochsToFinish']
            self.cohorts[cohort_name] = ModelsCohort(
                cohort_name,
                params_dir=cohort_settings['cohortParamsDir'],
                cohort_size=models_in_cohort,
                min_epochs2finish=min_epochs2finish,
                train=False,
                train_config_path=train_config,
                stable_validataion_genes_path=genes_stable_val_config,
                isdebug=isdebug
            )
    
    def processCohorts(self):
        for cohort_name, models_cohort in self.cohorts.items():
            models_cohort.cohortTrain()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--isdebug', type=str)
    opt = parser.parse_args()
    config_p = opt.config
    if config_p is None:
        print('add config path with --config param')
        exit()
    isdebug = str2bool(opt.isdebug)
    scheduler = TrainScheduler(config_p, isdebug)
    scheduler.processCohorts()