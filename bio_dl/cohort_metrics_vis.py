import os
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def metrics2vis_dict_from_path2config(path2config, metric2compare):
    metric2compare = 'table_tissues_metric'
    metrics2vis = {}
    with open(path2config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    for cohort_path in config['cohorts_paths']:
        cohort_name = os.path.basename(cohort_path)
        files = os.listdir(cohort_path)
        for f in files:
            if '_metrics' in f:
                f_path = os.path.join(cohort_path, f)
                
                with open(f_path, 'r') as metrics_f:
                    metrics_dict = json.load(metrics_f)
                if metrics_dict != {}:
                    for net_name in metrics_dict.keys():
                        net_metrics = metrics_dict[net_name]
                        net_file2compare = net_metrics[metric2compare]
                        try:
                            metric2compare_df = pd.read_csv(net_file2compare)
                            metric2compare_dict = metric2compare_df.to_dict(orient='list')
                            metric2compare_arr = [float(x[0]) for x in metric2compare_dict.values()]
                            if cohort_name not in metrics2vis.keys():
                                metrics2vis[cohort_name] = {}
                            metrics2vis[cohort_name][net_name] = metric2compare_arr
                        except pd.errors.EmptyDataError:
                            print(f'{net_file2compare} in process...')
                else:
                    print(f'{cohort_name} has not weights for {f}')
    return metrics2vis


def cohort_violins_from_path2config(path2config, outpath):
    metrics2vis = metrics2vis_dict_from_path2config(path2config, metric2compare='table_tissues_metric')
    cohort_num = len(metrics2vis.keys())
    model_per_cohort = [len(x) for x in metrics2vis.values()]

    height = cohort_num
    width = max(model_per_cohort)

    all_values = []
    for cohort_name in metrics2vis.keys():
        for model_name in metrics2vis[cohort_name].keys():
            model_vals = metrics2vis[cohort_name][model_name]
            all_values.append(max(model_vals))
    uniq_vals = list(set(all_values))
    color_map = plt.get_cmap('seismic')(uniq_vals)
    color_map = color_map[:, :3]

    fig, axs = plt.subplots(nrows=height, ncols=width, figsize=(20, 20))
    for cohort_num, cohort_name in enumerate(metrics2vis.keys()):
        for model_num, model_name in enumerate(metrics2vis[cohort_name].keys()):
            ax2process = axs[cohort_num, model_num]
            
            parts = ax2process.violinplot(
                metrics2vis[cohort_name][model_name],
                showmeans=True,
                showmedians=True,
                showextrema=False,
                # widths=0.55
            )
            max_val = max(metrics2vis[cohort_name][model_name])
            color_id = uniq_vals.index(max_val)
            color_rgb = color_map[color_id]
            color_hex = mpl.colors.to_hex(color_rgb, keep_alpha=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color_hex)
                # pc.set_facecolor('#D43F3A')
                # pc.set_facecolor('#D91C9B')
                pc.set_edgecolor('black')
                pc.set_alpha(1)
            
            parts['cmedians'].set_color('red')
            
            ax2process.yaxis.grid(True)
            ax2process.set_xticks([])
            ax2process.set_title(model_name)
            
            ax2process.set_ylim([0, max(uniq_vals)])
    fig.tight_layout()
    plt.savefig(outpath, dpi=1024)
    
def cohort_metrics(path2config, metric2compare):
    metrics2vis = metrics2vis_dict_from_path2config(path2config, metric2compare)
    cohort_mean_metrics_dict = {} 
    for cohort_name in metrics2vis.keys():
        cohort_mean_metrics_dict[cohort_name] = None
        # print(cohort_name)
        for model in metrics2vis[cohort_name].keys():
            model_values = metrics2vis[cohort_name][model]
            model_values_np = np.array(model_values)
            if cohort_mean_metrics_dict[cohort_name] is None:
                cohort_mean_metrics_dict[cohort_name] = model_values_np
            else:
                cohort_mean_metrics_dict[cohort_name] = np.vstack([
                    cohort_mean_metrics_dict[cohort_name],
                    model_values_np
                ])
    return cohort_mean_metrics_dict

def cohort_mean_table_tissues_metric_metrics(path2config, is_debug=False):
    cohort_metrics_dict = cohort_metrics(path2config, 'table_tissues_metric')
    cohort_mean_metrics_dict = {}
    for k, v in cohort_metrics_dict.items():
        if len(v.shape) == 2:
            cohort_mean_metrics_dict[k] = np.mean(v, axis=0)
        else:
            cohort_mean_metrics_dict[k] = v
    if is_debug:
        header = ['Net name', 'min P^2', 'mean P^2', 'median P^2','max P^2']
        header[0] += (50 - len(header[0])) * ' '
        info = 121*'-'
        info += '\n' 
        header_row = '|'
        for i in range(len(header)):
            header_row += f' {header[i]} \t|'
        info += header_row + '\n'
        info += 121*'-'
        info += '\n' 
        for k , v in cohort_mean_metrics_dict.items():
            cohortname2print = k + ((50-len(k))*' ')
            row_info = '| {} \t| {:.03f} \t| {:.03f} \t| {:.03f} \t| {:.03f} \t|\n'.format(
                cohortname2print,
                np.min(v), np.mean(v), np.median(v), np.max(v)
            )
            info += row_info
            info += 121*'-'
            info += '\n' 
        print(info)
    return cohort_mean_metrics_dict


def cohort_mean_violin(cohort_mean_metrics_dict, outpath):
    all_values = []
    for cohort_name in cohort_mean_metrics_dict.keys():
        model_vals = cohort_mean_metrics_dict[cohort_name]
        all_values.append(max(model_vals))
    uniq_vals = list(set(all_values))
    uniq_vals_np = np.array(uniq_vals)
    
    #for simple collor mapping
    uniq_vals_np = np.concatenate([uniq_vals_np, np.array([0, 1])]) 
    # uniq_vals_mapped = np.array(uniq_vals) / max(uniq_vals)
    uniq_vals_mapped = np.interp(uniq_vals_np, (uniq_vals_np.min(), uniq_vals_np.max()), (0, 1))
    # uniq_vals_mapped *= 0.5
    # print(uniq_vals_mapped)
    # exit()
    color_map = plt.get_cmap('viridis')(uniq_vals_mapped)
    color_map = color_map[:, :3]
    width = len(cohort_mean_metrics_dict.keys())
    fig, axs = plt.subplots(nrows=1, ncols=width, figsize=(25, 5))
    for cohort_num, cohort_name in enumerate(cohort_mean_metrics_dict.keys()):
        ax2process = axs[cohort_num]
        
        parts = ax2process.violinplot(
            cohort_mean_metrics_dict[cohort_name],
            showmeans=True,
            showmedians=True,
            showextrema=False,
            # widths=0.55
        )
        max_val = max(cohort_mean_metrics_dict[cohort_name])
        color_id = uniq_vals.index(max_val)
        color_rgb = color_map[color_id]
        color_hex = mpl.colors.to_hex(color_rgb, keep_alpha=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color_hex)
            # pc.set_facecolor('#D43F3A')
            # pc.set_facecolor('#D91C9B')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        
        parts['cmedians'].set_color('red')
        
        ax2process.yaxis.grid(True)
        ax2process.set_xticks([])
        ax2process.set_title(cohort_name)
            
        ax2process.set_ylim([0, 0.8])
    fig.tight_layout(pad=2, h_pad=1, w_pad=2)
    plt.savefig(outpath, dpi=1024)
    
    # print(cohort_mean_metrics_dict)
    



path2config = '/home/gerzog/repositories/mole.001/config/cohort_gene_expression/cohort_gene_expression.yaml'
mean_vals_dict = cohort_mean_table_tissues_metric_metrics(path2config, is_debug=True)
# cohort_mean_violin(mean_vals_dict, 'test_cohort_mean_violin.png')
