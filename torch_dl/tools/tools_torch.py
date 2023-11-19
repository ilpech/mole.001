from varname.helpers import debug
import os
import sys

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import time

# from sklearn.metrics import confusion_matrix

from tools_dl.tools import isiter, file_size

def isGPU(isdebug=False):
    if torch.cuda.is_available():
        ctx = 'gpu'
        print('successfully created gpu array -> using gpu')
    else:
        ctx = 'cpu'
        if isdebug:
            print('debug mode -> using cpu')
        else:
            print('cannot create gpu array -> using cpu')
    return ctx

def dist_setup(rank: int, backend, world_size, host='localhost', port='5000'):
    """Initializes a distributed training process group.
    
    Args:
        rank: a unique identifier for the process
        args: user defined arguments
    """
    if host is not None:
        os.environ['MASTER_ADDR'] = host
    if port is not None:
        os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group(backend=backend, 
                            init_method='env://', 
                            rank=rank, 
                            world_size=world_size)
    
def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
def dist_cleanup():
    dist.destroy_process_group()

def model_params_cnt(
    model: torch.nn.Module,
    convert2millions=True
):
    if model is None:
        return
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    if not convert2millions:
        return pp
    return '{:.2f} million params'.format(pp/10**6)

def inspect_model(
    model,
    sample_batch,
    model_name='',
    with_pt_temp_save=True
):
    print('*' * 70)
    print('inspect_model::{}'.format(model_name))
    print('*' * 35)
    debug(model_params_cnt(model))
    print('*' * 35)
    outs = model(sample_batch)
    debug(sample_batch.shape)
    if not isinstance(outs, list):
        outs = [outs]
    for i, out in enumerate(outs):
        print('out[{}] has type {}'.format(i, type(out)))
        try:
            debug(out.shape)
        except AttributeError:
            pass
    if with_pt_temp_save:
        print('*' * 35)
        out_p = 'temp_{}.pt'.format(model_name)
        torch.save(model.state_dict(), out_p)
        print('model params file weights::{}'.format(
            file_size(out_p))
        )
        os.remove(out_p)
        print('*' * 35)
    print('*' * 70)

def classify_preds2stat(preds: np.array, labels: np.array, classes_len):
    '''
    conf_m, tn, fp, fn, tp = pred2tn_fp_fn_tp(pred, label) 
    '''
    tp = np.zeros(shape=(1,classes_len), dtype='int32')
    fp = np.zeros(shape=(1,classes_len), dtype='int32')
    fn = np.zeros(shape=(1,classes_len), dtype='int32')
    error_matrix = np.zeros(shape=(classes_len,classes_len), dtype='int32')
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        if pred == label:
            tp[0, pred] += 1
        else:
            fn[0, label] += 1
            fp[0, pred] += 1
        error_matrix[label, pred] += 1
    return error_matrix, tp,  fp, fn

def precision_recall_f2(error_matrix):
    classes_len = len(error_matrix)
    precision = np.zeros(shape=(1,classes_len), dtype=np.float16)
    recall = np.zeros(shape=(1,classes_len), dtype=np.float16)
    fmetric = np.zeros(shape=(1,classes_len), dtype=np.float16)
    for i in range(classes_len):
        cls_cnt_gt = error_matrix[i, :].sum() # samples of class in gt
        cls_cnt_pred = error_matrix[:,i].sum() # predicted samples for class
        if cls_cnt_pred:
            precision[0][i] = error_matrix[i, i] / cls_cnt_pred
        recall[0][i] = error_matrix[i, i] / cls_cnt_gt
        if cls_cnt_gt and cls_cnt_pred:
            fmetric[0][i] = 2.0*precision[0][i]*recall[0][i]/(
                precision[0][i]+recall[0][i]
            )
    return precision, recall, fmetric


def plot_scheduler(scheduler, iter_num, save_path):
    lrs = []
    for i in range(iter_num):
        scheduler.optimizer.step()
        lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(range(iter_num),lrs)
    plt.grid()
    
    plt.savefig(
        save_path,
        bbox_inches ="tight",
        pad_inches = 1,
        transparent = True,
        facecolor ="w",
        edgecolor ='b',
        orientation ='landscape')
    print(f'plot_scheduler::saved to {save_path}')
    
    
def measure_model_batch(model, img_size, iter_num, ctx='cpu'):
    # measure time for one image validation
    iters_results = []
    for iter in range(iter_num):
        print(iter, '/', iter_num)
        test_image = torch.rand(img_size)
        
        if ctx == 'gpu':
            test_image = test_image.cuda()
        start_time = time.time()
        output = model(test_image)
        iter_result = time.time() - start_time
        iters_results.append(iter_result)
    one_img_val_time = np.mean(
        np.array(iters_results[int(0.1 * len(iters_results)):])
    )
    print('Time for one image validation is {:.03f} sec'.format(one_img_val_time))
    return one_img_val_time, iters_results


def list_from_prettytable(pr_table):
    # return info from PrettyTable format in list format
    table_list = [
        [item.strip() for item in line.split('|') if item]  # maintain the number of columns in rows.
        for line in pr_table.strip().split('\n')
        if '+-' not in line  # discard +-
        ]
    return table_list
    

if __name__ == '__main__':
    p = np.array([0,1,0,0,2,1])
    l = np.array([0,2,0,2,0,1])
    error_matrix, tp,  fp, fn = classify_preds2stat(p,l, 3)
    debug(p) 
    debug(l) 
    print(error_matrix) 
    precision, recall, fmetric = precision_recall_f2(error_matrix)
    print(tp) 
    print(fp) 
    print(fn)
    debug(precision) 
    debug(recall) 
    debug(fmetric) 