#usage: python3 torch_dl/train/train_linear_gene_regression.py --config config/gene_expression/train.yaml --isdebug False

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch_dl.dataloader.gene_data_loader import DistGeneDataLoader
from torch_dl.dataloader.gene_batch_linear_data_loader import GeneLinearDataLoader
import argparse
import yaml
import json
import time
from torch_dl.model.regression_model import (
    RegressionResNet2d18,
    RegressionResNet2d34,
    RegressionResNet2dV2_34,
    RegressionResNet2dV2_50
)
from torch_dl.model.regression_bio_perceptron import RegressionBioPerceptron
from torch_dl.model.model import TorchModel
from torch_dl.tools.tools_torch import (
    dist_setup, 
    dist_cleanup,
    inspect_model
)
from tools_dl.tools import (
    debug,
    str2bool, 
    roundUp,
    norm_shifted_log,
    denorm_shifted_log,
    ensure_folder,
    curDateTime,
    cp_r
)
from bio_dl.gene_mapping import (
    uniq_nonempty_uniprot_mapping_header
)
from bio_dl.gene import Gene
from tools_dl.base_trainer import TrainLogger
from torchmetrics import MeanSquaredError
from torchmetrics.regression.pearson import PearsonCorrCoef
from torch_dl.scheduler.cosine_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings

from torch_dl.train.train_gene_regression import DistGeneExpressionTrainer

torch.cuda.empty_cache()

class DistGeneLinearExpressionTrainer:
    '''
    Gene protein abundance regression predictor
    trainer with multi-GPU access 
    '''
    def __init__(
        self, 
        config_path, 
        isdebug,
        config_dict=None,
        force_run=True
    ):
        self.isdebug = isdebug
        if config_dict:
            print('DistGeneExpressionTrainer::using config dict from params not from path')
            self.config = config_dict
        else:
            self.config = DistGeneExpressionTrainer.opt_from_config(config_path)
        # with open(config_path) as f:
        #     self.config = yaml.load(f, yaml.FullLoader)
        self.start_time = curDateTime()
        self.config_path = config_path
        self.data_settings = self.config['data']
        self.train_settings = self.config['train']
        self.net_name = self.train_settings['net_name']
        self.finetune_net_name = self.train_settings['finetune_net_name']
        self.finetune_params_dir = self.train_settings['finetune_params_dir']
        self.epoch_start_from = self.train_settings['epoch_start_from']
        self.use_finetune_experiments = self.train_settings['use_finetune_experiments']
        self.use_finetune_databases = self.train_settings['use_finetune_databases']
        self.params_dir = self.train_settings['params_dir']
        self.params_path = os.path.join(self.params_dir, self.net_name)
        if force_run:
            if os.path.isdir(self.params_path) and not isdebug:
                raise Exception(
                    'error! net out dir {} already exists, choose another net name'.format(
                        self.params_path
                    ))
            if not isdebug:
                warnings.filterwarnings('ignore')
                ensure_folder(self.params_path)
                cp_r(self.config_path, self.params_path)

        self.log_path = '{}/{}_log.txt'.format(self.params_path, self.net_name)
        self.gpus2use = self.train_settings['gpus2use']
        self.logger = TrainLogger(self.log_path, '[MAIN_GPUs::{}]'.format(self.gpus2use))
        self.logger.print('script_start::{}'.format(self.start_time))
        self.databases_alphs_path = '{}/{}_databases_alphs.json'.format(self.params_path, self.net_name)
        self.model_config_path = '{}/{}_config.json'.format(self.params_path, self.net_name)
        self.lr_settings = self.train_settings['lr']
        self.lr_mode = self.lr_settings['mode']
        if len(self.lr_mode) > 1:
            raise Exception(
                'check config, should be only one selected lr mode'
            )
        self.lr_mode = self.lr_mode[0]
        self.lr_scheduler_metric = self.lr_settings['metric2use']
        if len(self.lr_scheduler_metric) > 1:
            raise Exception(
                'check config, should be only one selected lr mode'
            )
        self.lr_scheduler_metric = self.lr_scheduler_metric[0]
        self.lr_mode_settings = self.lr_settings[self.lr_mode]
        self.current_l = None
        self.batch_sizes = self.train_settings['batch_size']
        self.resume_epoch = 0
        self.augm_settings = self.train_settings['augm']
        self.with_augm = self.augm_settings['isEnabled']
        self.epochs = self.train_settings['epochs']

        self.log_interval = self.train_settings['log_interval']
        self.metric_flush_interval = self.train_settings['metric_flush_interval']
        self.wd = self.train_settings['wd']
        self.momentum = self.train_settings['momentum']
        self.optimizer = self.train_settings['optimizer']
        drop_rate = 0.0
        self.model_settings = self.config['model']
        self.model2use = self.model_settings['model2use']
        if len(self.model2use) > 1:
            raise Exception(
                'check config, should be only one selected model'
            )
        self.model2use = self.model2use[0]
        # self.width_factor = None
        # self.num_layers = None
        
        self.hidden_size = None
        self.annotations_dropout = None
        self.hidden_dropout = None
        
        if self.model2use == 'BioPerceptron':
            self.input_features_hidden_size = int(self.model_settings[self.model2use]['input_features_hidden_size'])
            self.hidden_size = int(self.model_settings[self.model2use]['hidden_size'])
            self.annotations_dropout = float(self.model_settings[self.model2use]['annotations_dropout'])
            self.hidden_dropout = float(self.model_settings[self.model2use]['hidden_dropout'])
        
        drop_rate = 0.0
        self.dist_cpu_backend = dist.Backend.GLOO
        self.dist_gpu_backend = dist.Backend.NCCL
        self.dist_backend = self.dist_cpu_backend
        self.avail_gpus = torch.cuda.device_count()
        if self.gpus2use > self.avail_gpus:
            print('avail gpus {} less than gpus2use param from config {}'.
            format(
                self.avail_gpus, self.gpus2use
                )
            )
            self.gpus2use = self.avail_gpus
        if torch.cuda.is_available():
            self.ctx = 'gpu'
            self.logger.print('successfully created gpu array -> using gpu')
            self.dist_backend = self.dist_gpu_backend
        else:
            self.ctx = 'cpu'
            if self.isdebug:
                self.logger.print('debug mode -> using cpu')
            else:
                self.logger.print('cannot create gpu array -> using cpu')
                self.logger.print(
                    'Error! dist train can not work without gpu, use usual train instead'
                )
                exit()
        
        self.current_lr = None
        self.current_batch_size = None
        
        if force_run:
            if self.gpus2use == 1 or self.isdebug:
                self.train_loop()
            # #=========DIST
            elif dist.is_available() and self.gpus2use > 0:
                print(f'cuda available -> mp spawn on {self.gpus2use} GPUs')
                mp.spawn(self.train_loop, nprocs=self.gpus2use)
        
    def create_model(self, num_channels):
        '''
        Create model or load epoch if self.epoch_start_from is set 
        '''
        if self.model2use == 'wideResNet':
            model_settings = self.model_settings[self.model2use]
            self.num_layers = model_settings['num_layers']
            self.width_factor = model_settings['width_factor']
            if self.num_layers == 34:
                model = RegressionResNet2d34(
                    num_channels=num_channels, 
                    channels_width_modifier=self.width_factor
                )
            elif self.num_layers == 18:
                model = RegressionResNet2d18(
                    num_channels=num_channels, 
                    channels_width_modifier=self.width_factor
                )
            else:
                raise Exception(
                    'no model {} with {} layers and {} width factor found'
                ).format(
                    self.model2use,
                    self.num_layers,
                    self.width_factor
                )
        elif self.model2use == 'ResNetV2':
            model_settings = self.model_settings[self.model2use]
            self.num_layers = model_settings['num_layers']
            self.width_factor = model_settings['width_factor']
            if self.num_layers == 34:
                model = RegressionResNet2dV2_34(
                    num_channels=num_channels, 
                    channels_width_modifier=self.width_factor
                )
            elif self.num_layers == 50:
                model = RegressionResNet2dV2_50(
                    num_channels=num_channels, 
                    channels_width_modifier=self.width_factor
                )
            else:
                raise Exception(
                    'no model {} with {} layers and {} width factor found'
                ).format(
                    self.model2use,
                    self.num_layers,
                    self.width_factor
                )
        else:
            raise Exception(
                f'no model {self.model2use} found in settings, use from model.model2use'
            )
            
        if self.epoch_start_from > 0:
            self.logger.print('finetune from {}!'.format(self.epoch_start_from))
            return TorchModel(
                new_params_dir=self.params_dir, 
                new_net_name=self.net_name,
                net_name=self.finetune_net_name,
                params_dir=self.finetune_params_dir,
                model=model,
                load=True,
                epoch=self.epoch_start_from,
                test_mode=self.isdebug
            )
        return TorchModel(
            params_dir=self.params_dir, 
            net_name=self.net_name,
            model=model,
            test_mode=self.isdebug
        )
        
    def train_loop(
        self, 
        gpu_id=0, 
        dataset=None
    ):
        
        '''
        Multi-GPU train loop, 
        each loop function is created for
        selected GPU id
        
        At the end of each epoch weights
        from each GPU sync and validate only once
        '''
        start_time = curDateTime()
        gpu_log_path = '{}/{}_gpu.{:03d}_log.txt'.format(self.params_path, self.net_name, gpu_id)
        logger = TrainLogger(gpu_log_path, 'GPU[{:03d}]'.format(gpu_id))
        # rank = args.nr * args.gpus + gpu
        databases = uniq_nonempty_uniprot_mapping_header()
        net_config_dir=os.path.join(
            self.finetune_params_dir, 
            self.finetune_net_name
        )

        print('loading data...')
        self.each_train_has_own_dataset = self.config[
            'train'
        ]['each_train_has_own_dataset']
        self.current_batch_size = self.batch_sizes[0]      
        # train_loader, val_loader = DistGeneDataLoader.createDistLoader(
        train_loader, val_loader = GeneLinearDataLoader.createDistLoader(
            self.config_path,
            gpu_id,
            self.current_batch_size,
            self.gpus2use,
            dataset=dataset,
            # num_workers=os.cpu_count()-1,
            num_workers=self.gpus2use*2,
            net_config_path=net_config_dir,
            use_net_databases=self.use_finetune_databases,
            crop_db_alph=False
        )
        data_loader = train_loader.dataset.baseloader
        self.max_label = data_loader.max_label
        print(f'dist dataloader created for gpu {gpu_id}')
        
        annotations_len = len(databases) * data_loader.max_var_layer
        # inference_shape = (
        #     1, 
        #     4+len(databases), 
        #     data_loader.max_var_layer, 
        #     len(Gene.proteinAminoAcidsAlphabet())
        # )
        logger.print('creating linear regression model...'.format(gpu_id))
        # logger.print('creating resnset regression model...'.format(gpu_id))
        # model: TorchModel = self.create_model(
        #     num_channels=10
        # )
        
        # inspect_model(
        #     model.model(),
        #     torch.zeros(size=inference_shape)
        # )
        
        network = RegressionBioPerceptron(
            input_annotations_len=annotations_len, 
            input_type_len=len(data_loader.proteinMeasurementsAlphabet), 
            input_gene_seq_bow_len=len(Gene.proteinAminoAcidsAlphabet()),
            input_features_hidden_size=self.input_features_hidden_size,
            hidden_size=self.hidden_size,
            annotation_dropout=self.annotations_dropout,
            hidden_dropout=self.hidden_dropout
        )
        
        if self.epoch_start_from > 0:
            self.logger.print('finetune from {}!'.format(self.epoch_start_from))
            model = TorchModel(
                new_params_dir=self.params_dir, 
                new_net_name=self.net_name,
                net_name=self.finetune_net_name,
                params_dir=self.finetune_params_dir,
                model=network,
                load=True,
                epoch=self.epoch_start_from,
                test_mode=self.isdebug
            )
        model = TorchModel(
            params_dir=self.params_dir, 
            net_name=self.net_name,
            model=network,
            test_mode=self.isdebug
        )
        
        if dist.is_available() and gpu_id >= 0:
            model._model.cuda(gpu_id)
            # wrap the model to current gpu
            if gpu_id > 0:
                dist_setup(gpu_id, self.dist_backend, self.gpus2use)
                model._model = DDP(model._model, device_ids=[gpu_id])
        else:
            print('train adopted only for gpu, exit...')
            exit()
          
        logger.print('train started at::{}'.format(start_time))
        if gpu_id == 0:
            self.logger.print('{} genes2train'.format(len(data_loader.genes2train)))
            self.logger.print('{} genes2val'.format(len(data_loader.genes2val)))
            self.logger.print('{} epxs in data'.format(len(data_loader)))
        
        max_eps = data_loader.maxProteinMeasurementsInData()
        
        data_cnt = len(data_loader)
        L = torch.nn.MSELoss()
        # L = torch.nn.SmoothL1Loss(beta=0.5)
        # L = torch.nn.HuberLoss(delta=0.5)
        scheduler = None
        if self.lr_mode == 'byhand':
            lr_dict = self.lr_settings['byHand']
            self.current_lr = lr_dict[0]
        elif self.lr_mode == 'byPlateu':
            lr_dict = self.lr_settings['byPlateu']
            self.current_lr = lr_dict['start_value']
        elif self.lr_mode == 'cosineWarmup':
            lr_dict = self.lr_settings['cosineWarmup']
            self.current_lr = lr_dict['min_lr']
            
        optimizer = torch.optim.SGD(
            model.model().parameters(), 
            lr=self.current_lr,
            momentum=self.config['train']['momentum'],
            weight_decay=self.config['train']['wd']
        )
        if self.lr_mode == 'byPlateu':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_dict['mode'],
                factor=lr_dict['factor'],
                patience=lr_dict['patience'],
                threshold=lr_dict['threshold'],
                cooldown=lr_dict['cooldown'],
                min_lr=lr_dict['min_lr']
            )
        elif self.lr_mode == 'cosineWarmup':
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                max_lr=lr_dict['max_lr'],
                min_lr=lr_dict['min_lr'],
                warmup_steps=lr_dict['warmup_steps'],
                first_cycle_steps=lr_dict['first_cycle_steps'],
                cycle_mult=lr_dict['cycle_mult'],
                gamma=lr_dict['gamma']
            )
        num_batch = roundUp(data_cnt/self.current_batch_size)
        best_epoch = 0
        max_val_p2n = None
        assert len(data_loader.databases_alphs)
        with open(self.databases_alphs_path, 'w') as f:
            json.dump(data_loader.databases_alphs, f, indent=4)
            print('databases info written', self.databases_alphs_path)
        # if gpu_id == 0:
        #     self.logger.print('batch shape::{}'.format(inference_shape))
        rna_exps_alphabet = data_loader.rnaMeasurementsAlphabet
        if self.max_label is None:
            self.max_label = 1.0
            # self.max_label = norm_shifted_log(
            #     data_loader.maxProteinMeasurementInData()
            # )
        max_label = self.max_label
        protein_exps_alphabet = data_loader.proteinMeasurementsAlphabet
        config_data = {
            'net_name': self.net_name,
            'model_name': self.model2use,
            # 'num_layers': self.num_layers,
            # 'width_factor': self.width_factor,
            # 'inference_shape': inference_shape,
            'denorm_max_label': float(max_label),
            'rna_exps_alphabet': rna_exps_alphabet,
            'protein_exps_alphabet': protein_exps_alphabet,
            'databases': databases,
            'genes2val': data_loader.genes2val
        }
        if gpu_id == 0:
            with open(self.model_config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                print('config info written', self.model_config_path)
            if self.each_train_has_own_dataset:
                print('train...')
            else:
                print('train with common dataset...')
        epoch_from = model.epoch
        objs2train = len(train_loader)
        train_metric_mse = MeanSquaredError().cuda(gpu_id)
        val_metric_mse = MeanSquaredError().cuda(gpu_id)
        train_metric_p = PearsonCorrCoef().cuda(gpu_id)
        val_metric_p = PearsonCorrCoef().cuda(gpu_id)
        train_metric_p_n = PearsonCorrCoef().cuda(gpu_id)
        val_metric_p_n = PearsonCorrCoef().cuda(gpu_id)
        train_denorm_metric_rmse = MeanSquaredError(
            squared=False
        ).cuda(gpu_id) 
        val_denorm_metric_rmse = MeanSquaredError(
            squared=False
        ).cuda(gpu_id)
        train_norm_metric_rmse = MeanSquaredError(
            squared=False
        ).cuda(gpu_id) 
        val_norm_metric_rmse = MeanSquaredError(
            squared=False
        ).cuda(gpu_id)
        logger.print('start training {}...'.format(self.net_name))
        all_passed = 0
        lr_scheduler_step = 1
        scheduler_step = 0
        for i in range(self.epochs):
            model.model().train()  
            data_loader.shuffleTrain()
            epoch = i
            epoch_tic = time.time()
            if self.lr_mode == 'byHand':
                if epoch in lr_dict and epoch > 0:
                    self.current_lr = lr_dict[epoch] 
                    optimizer.param_groups[0]['lr'] = self.current_lr 
                    self.logger.print(
                        '\nlrScheduler::byHand::{}::Current learning rate is:{:.8f}'.format(
                            lr_scheduler_step,
                            self.current_lr
                        )
                    )
                    lr_scheduler_step += 1
            
            if epoch in self.batch_sizes and epoch > 0:
                self.current_batch_size = self.batch_sizes[epoch]
                self.logger.print('Current batch size is:{}'.format(
                    self.current_batch_size)
                )
                # train_loader, _ = DistGeneDataLoader.createDistLoader(
                train_loader, _ = GeneLinearDataLoader.createDistLoader(
                    self.config_path,
                    gpu_id,
                    self.current_batch_size,
                    self.gpus2use,
                    # num_workers=os.cpu_count()-1,
                    num_workers=self.gpus2use*2,
                    net_config_path=os.path.join(
                        self.finetune_params_dir, 
                        self.finetune_net_name
                    ),
                    use_net_databases=self.use_finetune_databases,
                    crop_db_alph=False
                )
                objs2train = len(train_loader)
            train_tic = time.time()
            train_loss = 0
            passed = 0 # train batches passed
            valpassed = 0 # val batches passed
            # for data, labels in train_loader:
            for annotations, types_one_hot, norm_rna_vals, gene_seq_bow, labels in train_loader:
                if norm_rna_vals is None or labels is None:
                    continue
                # labels_gpu, data_gpu = (
                #     labels.unsqueeze(1).float().cuda(gpu_id),
                #     data.float().cuda(gpu_id)
                # )
                annotations_gpu, types_one_hot_gpu, norm_rna_vals_gpu, gene_seq_bow_gpu, labels_gpu = (
                    annotations.float().cuda(gpu_id),
                    types_one_hot.float().cuda(gpu_id),
                    norm_rna_vals.float().cuda(gpu_id),
                    gene_seq_bow.float().cuda(gpu_id),
                    labels.unsqueeze(1).float().cuda(gpu_id)
                )
                
                # out = model(data_gpu)
                out = model(
                    annotations_gpu, 
                    types_one_hot_gpu, 
                    norm_rna_vals_gpu, 
                    gene_seq_bow_gpu
                )
                loss = L(
                    out, 
                    labels_gpu
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if passed and not passed % self.metric_flush_interval:
                    sys.stdout.write(
                        '\rEPOCH::{} {}::{}:{} bs::{} lr({})::{:.8f} P2_norm_train::{:.3f}'
                            .format(
                                epoch,
                                all_passed+1,
                                passed+1, 
                                objs2train,
                                self.current_batch_size,
                                lr_scheduler_step,
                                self.current_lr,
                                train_metric_p_n.compute() ** 2
                            )
                    )
                passed += 1
                all_passed += 1
                if scheduler and self.lr_mode == 'cosineWarmup':
                    step_each = self.lr_mode_settings['step_each_batch']
                    if not all_passed % step_each:
                        scheduler_step += 1
                        try:
                            scheduler.step(scheduler_step)
                            # scheduler.step(passed)
                        except ValueError:
                            logger.print(
                                'cosine value error on {} step'.format(scheduler_step)
                            )
                            scheduler_step = int(scheduler_step/2)
                    self.current_lr = scheduler.get_lr()[0]
                    lr_scheduler_step = scheduler.cycle
                sys.stdout.flush()
                train_loss += loss.item()
                train_metric_mse.update(out, labels_gpu)
                train_metric_p_n.update(out, labels_gpu)
                train_norm_metric_rmse.update(out, labels_gpu)
                denorm_labels = torch.Tensor(
                    [denorm_shifted_log(x*max_label) for x in labels_gpu]
                ).cuda(gpu_id)
                denorm_out = torch.Tensor(
                    [denorm_shifted_log(x*max_label) for x in out]
                ).cuda(gpu_id)
                train_denorm_metric_rmse.update(denorm_out, denorm_labels)
                train_metric_p.update(denorm_out, denorm_labels)
            train_mse = train_metric_mse.compute()
            train_denorm_rmse = train_denorm_metric_rmse.compute()
            train_norm_rmse = train_norm_metric_rmse.compute()
            train_p2 = train_metric_p.compute() ** 2
            train_p2n = train_metric_p_n.compute() ** 2
            train_metric_mse.reset()
            train_norm_metric_rmse.reset()
            train_denorm_metric_rmse.reset()
            train_metric_p.reset()
            train_metric_p_n.reset()
            del annotations_gpu
            del types_one_hot_gpu
            del norm_rna_vals_gpu
            del labels_gpu
            del gene_seq_bow_gpu
            torch.cuda.empty_cache()
            train_time = time.time() - train_tic
            #val
            model.model().eval() # turn off train mode while validation
            val_tic = time.time()
            print('\n{}::\nval started...'.format(curDateTime()))
            # for data, labels in val_loader:
            for annotations, types_one_hot, norm_rna_vals, gene_seq_bow, labels in val_loader:
                # labels_gpu, data_gpu = labels.unsqueeze(1).float().cuda(gpu_id), \
                #                 data.float().cuda(gpu_id)
                annotations_gpu, types_one_hot_gpu, norm_rna_vals_gpu, gene_seq_bow_gpu, labels_gpu = (
                    annotations.float().cuda(gpu_id),
                    types_one_hot.float().cuda(gpu_id),
                    norm_rna_vals.float().cuda(gpu_id),
                    gene_seq_bow.float().cuda(gpu_id),
                    labels.unsqueeze(1).float().cuda(gpu_id)
                )
                # out = model(data_gpu)
                out = model(annotations_gpu, types_one_hot_gpu, norm_rna_vals_gpu, gene_seq_bow_gpu)
                
                valpassed += 1
                val_metric_mse.update(out, labels_gpu)
                val_metric_p_n.update(out, labels_gpu)
                val_norm_metric_rmse.update(out, labels_gpu)
                denorm_labels = torch.Tensor(
                    [denorm_shifted_log(x*max_label) for x in labels_gpu]
                ).cuda(gpu_id)
                denorm_out = torch.Tensor(
                    [denorm_shifted_log(x*max_label) for x in out]
                ).cuda(gpu_id)
                val_denorm_metric_rmse.update(denorm_out, denorm_labels)
                val_metric_p.update(denorm_out, denorm_labels)
            val_time = time.time() - val_tic
            #========= EPOCH FINISHED =========
            val_mse = val_metric_mse.compute()
            val_denorm_rmse = val_denorm_metric_rmse.compute()
            val_norm_rmse = val_norm_metric_rmse.compute()
            val_p2 = val_metric_p.compute() ** 2
            val_p2n = val_metric_p_n.compute() ** 2
            if scheduler and self.lr_mode == 'byPlateu':
                scheduler.step(val_p2n)
                self.current_lr = [
                    group['lr'] for group in optimizer.param_groups
                ][0]
                self.logger.print(
                    '\nlrScheduler::byPlateu::{}::Current learning rate is:{:.8f}'.format(
                        lr_scheduler_step,
                        self.current_lr
                    )
                )
                
            val_metric_mse.reset()
            val_norm_metric_rmse.reset()
            val_denorm_metric_rmse.reset()
            val_metric_p.reset()
            val_metric_p_n.reset()
            del annotations_gpu
            del types_one_hot_gpu
            del norm_rna_vals_gpu
            del labels_gpu
            torch.cuda.empty_cache()
            new_best_val = False
            if not max_val_p2n:
                max_val_p2n = val_p2n
            else:
                if val_p2n > max_val_p2n:
                    max_val_p2n = val_p2n
                    new_best_val = True
                    best_epoch = i
            train_loss /= num_batch
            epoch_time = time.time() - epoch_tic
            device_msg = '[Epoch::{:03d}] GPU::{} bs::{} \n'\
                  '|-------------------------------------\n'\
                  '|  (time[sec])::epoch::{:.2f}\n'\
                  '|  (time[sec])::train::{:.2f}\n'\
                  '|  (time[sec])::val::{:.2f}\n'\
                  '|-------------------------------------\n'.format(
                  i, 
                  gpu_id,
                  self.current_batch_size,
                  epoch_time,
                  train_time,
                  val_time,
            )
            logger.print(device_msg)
            model.epoch = i
            TorchModel.checkpoint(model._model, gpu_id, model.pt_path())
            if gpu_id == 0:
                msg = '[Epoch::{:03d}] GPU::{} bs::{} lr({})::{:.10f} \n'\
                    '|-------------------------------------\n'\
                    '|  (val)::P^2_metric::{:.6f} \n'\
                    '|  (val)::P^2_norm_metric::{:.6f} \n'\
                    '|  (val)::MSE_metric::{:.6f} \n'\
                    '|  (val)::RMSE_denorm_metric::{:.6f} \n'\
                    '|  (val)::RMSE_norm_metric::{:.6f} \n'\
                    '|-------------------------------------\n'\
                    '|  (train)::P^2_metric::{:.6f} \n'\
                    '|  (train)::P^2_norm_metric::{:.6f} \n'\
                    '|  (train)::MSE_metric::{:.6f} \n'\
                    '|  (train)::RMSE_denorm_metric::{:.6f} \n'\
                    '|  (train)::RMSE_norm_metric::{:.6f} \n'\
                    '|  (train)::MSE_loss::{:.6f}\n'\
                    '|-------------------------------------\n'.format(
                    i, 
                    gpu_id,
                    self.current_batch_size,
                    lr_scheduler_step,
                    self.current_lr,
                    val_p2,
                    val_p2n,
                    val_mse,
                    val_denorm_rmse,
                    val_norm_rmse,
                    train_p2,
                    train_p2n,
                    train_mse,
                    train_denorm_rmse,
                    train_norm_rmse,
                    train_loss
                )
                self.logger.print(msg)
                
                if new_best_val:
                    self.logger.print('new best val!::{:.6f}'.format(max_val_p2n))
                if i and not new_best_val:
                    self.logger.print(
                        'best val was at epoch({})::{:.6f}'.format(
                            best_epoch, 
                            max_val_p2n
                        )
                    )
                self.logger.print('='*50)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--isdebug', type=str)
    opt = parser.parse_args()
    if opt.config is None:
        print('add config path with --config param')
        exit()
    isdebug = str2bool(opt.isdebug)
    # trainer = DistGeneExpressionTrainer(opt.config, isdebug)
    trainer = DistGeneLinearExpressionTrainer(opt.config, isdebug)
    # trainer.data_loader.info()
    trainer.train_loop()
    
    
    
    

