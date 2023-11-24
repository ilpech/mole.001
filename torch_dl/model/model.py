# python3 torch_dl/model/model.py
import os
import torch
import torch.distributed as dist
from torch import nn
import json
from typing import List
from collections import OrderedDict
import yaml

from tools_dl.tools import (
    ensure_folder, 
    curDateTime, 
    ls, 
    cp_r,
    find_files
)
from torch_dl.tools.tools_torch import model_params_cnt
from torch.nn.parallel import DistributedDataParallel as DDP

from argparse import Namespace

from varname.helpers import debug

# from torchsummary import summary


class TorchModel:
    def __init__(
        self, 
        params_dir, 
        net_name, 
        epoch=0,
        model: torch.nn.Module=None,
        load=True,
        test_mode=False,
        ctx='cpu',
        gpu_id=0,
        new_params_dir=None,
        new_net_name=None
    ):
        self.create_time = curDateTime()
        self._model: torch.nn.Module = model
        self.net_name = net_name
        self.params_dir = os.path.abspath(params_dir)
        self.epoch = epoch
        self.loaded = False
        self.test_mode = test_mode
        self.load_from_params = load
        self.ctx = ctx
        self.gpu_id = gpu_id
        self.config = None
        self.finetuned = None
        self.parent_net_name = None
        self.parent_training_config = None
        self.history = None
        if self.load_from_params and self.epoch > 0:
            self.load(self.ctx)
            if self.test_mode and ctx != 'cpu':
                if dist.is_available() and torch.cuda.is_available():
                    self._model.cuda(self.gpu_id)
            if new_net_name is not None and new_params_dir is not None:
                self.parent_net_name = self.net_name
                self.parent_params_dir = self.params_dir
                self.parent_epoch = epoch
                print('net name update')
                print('{} -> {}'.format(self.net_name, new_net_name))
                self.net_name = new_net_name
                print('params dir update')
                print('{} -> {}'.format(self.params_dir, new_params_dir))
                self.params_dir = new_params_dir
                self.finetuned = True
        if not self.test_mode:
            ensure_folder(self.params_dir)
            ensure_folder(self.pt_dir())

    def with_model_created(self):
        return self._model is None
    
    def history2str(self):
        if not self.history:
            return None
        for i, k in enumerate(self.history.keys()):
            child_name = os.path.basename(k)
            parent_name = os.path.basename(self.history[k])
            child_train_cfg = self.read_train_config(
                k
            )
            parent_epoch = child_train_cfg['train']['epoch_start_from']
            if not i:
                out = 'history({})::\n'.format(self.net_name)
                out += '=' * 75 + '\n'
                out += '{}_{:04d}->\n'.format(child_name, self.epoch)
            if i != len(self.history.keys())-1:
                out += '->{}_{:04d}->\n'.format(parent_name, parent_epoch)
            else:
                out += '->{}_{:04d}...\n'.format(parent_name, parent_epoch)
        out += '=' * 75 + '\n'
        return out
    
    def history2epochs(self):
        if not self.history:
            return None
        out = []
        for i, k in enumerate(self.history.keys()):
            child_name = os.path.basename(k)
            parent_name = os.path.basename(self.history[k])
            child_train_cfg = self.read_train_config(
                k
            )
            parent_epoch = child_train_cfg['train']['epoch_start_from']
            if not i:
                out.append('{}_{:04d}'.format(child_name, self.epoch))
            out.append('{}_{:04d}'.format(parent_name, parent_epoch))
        return out
    
    def parent_config_path(self):
        if not self.parent_net_name:
            return None
        ifparents = find_files(
            self.parent_params_dir, 
            '_config', 
            abs_p=True
        )
        if not len(ifparents):
            return None
        if len(ifparents) > 1:
            print('parent_config::found more 1 parent config')
        return ifparents[0]
    
    def train_config_path(self, net_dir):
        iftrain = find_files(
            net_dir, 
            'train.yaml', 
            abs_p=True
        )
        if not len(iftrain):
            return None
        return iftrain[0]
    
    def read_train_config(self, net_dir):
        parent_train_cfg_path = self.train_config_path(net_dir)
        with open(parent_train_cfg_path) as f:
            # print('read train config from path {}'.format(parent_train_cfg_path))
            return yaml.load(f, yaml.FullLoader)
        
    def parent_train_config_path_from_train(self, net_dir):
        train_cfg = self.read_train_config(net_dir)['train']
        if train_cfg['epoch_start_from'] == 0:
            return None
        parent_dir = os.path.join(
            train_cfg['finetune_params_dir'],
            train_cfg['finetune_net_name']
        )
        return self.train_config_path(parent_dir)
        
    def browse_parents(self):
        if self.history is None:
            self.history = {}
        parent_config_dir = self.config_dir()
        if self.parent_net_name:
            parent_config_p = self.parent_config_path()
            parent_config_dir = os.dirname(parent_config_p)
            self.history[
                os.path.dirname(self.config_path())
            ] = parent_config_dir
        while parent_config_dir is not None:
            older_parent_config_path = self.parent_train_config_path_from_train(parent_config_dir)
            if not older_parent_config_path:
                parent_config_dir = None
                break 
            older_parent_config_dir = os.path.dirname(older_parent_config_path) 
            self.history[parent_config_dir] = older_parent_config_dir
            parent_config_dir = older_parent_config_dir
        return self.history 

    def name(self):
        return '{}_{:04d}'.format(self.net_name, self.epoch) 
    
    # def net_summary(self, inp_size=(3, 224, 224)):
    #     return summary(self._model, inp_size, device='cpu')
    
    def pt_name(self):
        return '{}.pt'.format(self.name()) 
    
    def config_name(self):
        return '{}_config.json'.format(self.net_name)
    
    def net_dir(self):
        return os.path.join(self.params_dir, self.net_name)
     
    def config_path(self):
        return os.path.join(self.net_dir(), self.config_name())

    def config_dir(self):
        return self.params_dir 
    
    def pt_path(self):
        return os.path.join(self.pt_dir(), self.pt_name())

    def pt_dir(self):
        return os.path.join(self.net_dir(), self.net_name)
    
    def log_path(self):
        return os.path.join(self.net_dir(), f'{self.net_name}_log.txt')
        
    def load(self, ctx='cpu'):
        if self._model is None:
            self._model = torch.nn.Module()
        loaded_state_dict = torch.load(
            self.pt_path(), 
            map_location=torch.device(ctx)
        )
        new_state_dict = OrderedDict()
        for k, v in loaded_state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        self._model.load_state_dict(
            new_state_dict
        )
        maybe_cfg_path = self.config_path()
        if os.path.isfile(maybe_cfg_path):
            with open(maybe_cfg_path, 'r') as f:
                self.config = json.load(f)
        self.loaded = True

    def save(self):
        if self.test_mode:
            print('WARNING!!! save with def path disabled in test mode')
            return
        cwd = os.getcwd()
        os.chdir(self.net_dir())
        torch.save(self._model.state_dict(), self.pt_path())
        print('model saved {}'.format(self.pt_path()))
        # !realize
        # self.write_config('.')
        os.chdir(cwd)

    @staticmethod
    def checkpoint(
        model: nn.Module, 
        gpu: int, 
        pt_path,
        
    ):
        """
        saves the model in master process and loads it everywhere else.
        """
        if gpu == 0:
            # all processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # saving it in one process is sufficient.
            torch.save(model.state_dict(), pt_path)
        else:
            # use a barrier() to make sure that process 1 loads the model 
            # after process 0 saves it.
            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            model.load_state_dict(
            torch.load(pt_path, map_location=map_location))

    def save2path(self, params_dir):
        cwd = os.getcwd()
        os.chdir(params_dir)
        torch.save(self._model.state_dict(), self.pt_path())
        # !realize
        # self.write_config('.')
        os.chdir(cwd)
    
    def model(self):
        return self._model

    def __call__(self, *args, **kwds) -> torch.Tensor:
        if self._model is not None:
            return self._model(*args, **kwds)
    
    def exportEpochs(self, export_dir: str, epochs2export: List[int]):
        ls_configs = ls(self.net_dir())
        ls_params = ls(self.pt_dir())
        export_nn_dir = os.path.join(
            os.path.abspath(export_dir), 
            self.net_name
        )
        ensure_folder(export_nn_dir)
        print('created export folder for nn:: {}'.format(export_nn_dir))
        for param_file in ls_configs:
            do_export = False
            if 'out' == param_file:
                do_export = True
            ext = os.path.splitext(param_file)[1]
            if not len(ext) and not do_export:
                continue
            if '.json' == ext:
                do_export = True
            if '.yaml' == ext:
                do_export = True
            if '.txt' == ext:
                do_export = True
            if not do_export:
                continue
            params_from_path = os.path.join(self.net_dir(), param_file)
            params_out_path = os.path.join(export_nn_dir, param_file)
            cp_r(params_from_path, params_out_path)
            if os.path.isfile(params_out_path) or os.path.isdir(params_out_path):
                print()
                print(params_from_path, ' copied to')
                print(params_out_path)
                print()
            else:
                print()
                print('ERROR WHILE COPYING')
                print(params_from_path, ' to')
                print(params_out_path)
                print()
        for param_file in ls_params:
            do_export = False
            ext = os.path.splitext(param_file)[1]
            if '.pt' == ext:
                for epoch in epochs2export:
                    if '_{:04d}'.format(epoch) in param_file:
                        do_export = True
                        break
            if not do_export:
                continue
            params_from_path = os.path.join(self.pt_dir(), param_file)
            params_out_path = os.path.join(export_nn_dir, self.net_name, param_file)
            cp_r(params_from_path, params_out_path)
            if os.path.isfile(params_out_path) or os.path.isdir(params_out_path):
                print()
                print(params_from_path, ' copied to')
                print(params_out_path)
                print()
            else:
                print()
                print('ERROR WHILE COPYING')
                print(params_from_path, ' to')
                print(params_out_path)
                print()
        print('EXPORT DONE!')
        
    def getParamsNumber(self):
        return model_params_cnt(self._model)

    @staticmethod
    def netWeights2epoch(param_file):
        '''
        check if file have format of exported epoch 
        like rna2protein_nci60.ResNet34V2.005_0012.pt,
        info after last _ should be epoch number
        
        return None if epoch not found
        '''
        fname, ext = os.path.splitext(param_file)
        if '.pt' != ext:
            return None
        last_split = fname.split('_')[-1]
        if len(last_split) == 4:
            try:
                epoch = int(last_split)
                return epoch
            except ValueError:
                return None 

    def lastEpoch(self):
        ls_params = ls(self.pt_dir())
        epochs = sorted([TorchModel.netWeights2epoch(x) for x in ls_params if x])
        if not len(epochs):
            return None
        return epochs[-1]
    
    
    def bestEpochFromLog(self, non_zero=True):
        metric2checkstr = '|  (val)::P^2_norm_metric::'
        with open(self.log_path(), 'r') as f:
            log_data = f.readlines()
        f.close()
        epochs = []
        p2_metric_vals = []
        
        for row in log_data:
            if '[Epoch::' in row:
                epoch = int(row[8:11])
                epochs.append(epoch)
            if metric2checkstr in row:
                row.replace('\n', '')
                val = float(row.replace(
                    metric2checkstr, ''
                ))
                p2_metric_vals.append(val)
        
        # we cannot use 0 epoch
        if non_zero:
            epochs = epochs[1:]
            p2_metric_vals = p2_metric_vals[1:]
            
        best_val = max(p2_metric_vals)
        best_epoch = epochs[p2_metric_vals.index(best_val)]
        return best_val, best_epoch
    
    
def export_regression():
    from torch_dl.model.regression_model import RegressionResNet2d18 
    sample = torch.randn((1,10,256,20))
    model = TorchModel(
        params_dir='trained/r_prot_abundance_regressor.011', 
        net_name='r_prot_abundance_regressor.011',
        model=RegressionResNet2d18(num_channels=sample.shape[1]),
        epoch=82
    )
    debug(model.get_n_params())
    # model.exportEpochs('trained/2export/', [10, 55, 80])

            
            
if __name__ == '__main__':
    # net_name = 'rna2protein_nci60.002'
    # epoch = 4
    # net_name = 'rna2protein_nci60.ResNet34V2.001'
    # epoch = 1
    # net_name = 'rna2protein_nci60.ResNet50V2.001'
    # epoch = 2
    # net_name = 'rna2protein_expression_regressor.033'
    # epoch = 6
    # net_name = 'rna2protein_tissue29.ResNet34V2.001'
    # epoch = 2
    net_name = 'rna2protein.ResNet34V2.002'
    epoch = 4
    export_dir = 'trained' 
    model = TorchModel(
        params_dir=export_dir, 
        net_name=net_name,
        model=None,
        epoch=epoch,
        load=False
    )
    model.exportEpochs('trained/2export', [epoch])
    exit()
    history = model.browse_parents()
    historystr = model.history2str()
    print(historystr)
    history_w_epochs = model.history2epochs()
    for generation in history_w_epochs:
        dir_path_from_hist = [
            v for k,v in history.items()  
            if os.path.basename(v) in generation
        ]
        
        debug(generation)
        debug(dir_path_from_hist)
        if len(dir_path_from_hist):
            metric_files = find_files(dir_path_from_hist[0], generation, abs_p=True)
            metric_files = [x for x in metric_files if 'out' in x]
            debug(metric_files)

