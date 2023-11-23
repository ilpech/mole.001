#python3 bio_dl/metric_inferenced.py --config_path config/metric_inferenced/metric_inferenced.yaml
import os
import numpy as np
import argparse
from tools_dl.tools import (
    boolean_string,
    debug,
    curDateTime,
    list2chunks,
    flat_list,
    is_number
)
import json
import csv
from scipy.stats import spearmanr, pearsonr 
import argparse
from tools_dl.base_trainer import TrainLogger
from tools_dl.heatmap_table import (
    matrix2heatmap, 
    violin_plot_exps_gt, 
    violin
)
from tools_dl.tools import norm_shifted_log, find_files
import yaml

# from bio_dl.tissue_mapping import ExptsMapping, MatchingExpt

def tensor_str2float(tensor_str):
    try:
        return float(tensor_str)
    except ValueError:
        # return float(tensor_str.split('[')[1].split(']')[0])
        return float(tensor_str.split('[[')[1].split(']')[0])
    

def is_measurement_negative(measurement):
    out = measurement
    if not isinstance(out, float):
        out = tensor_str2float(measurement)
    return out < 0

class MetricInferenced:
    def __init__(
        self,
        config_path
    ):
        self.st_time = curDateTime()
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        self.isdebug = self.config['isdebug']
        self.metric_files = self.config['metric_files']
        self.select_genes = boolean_string(self.config['select_genes'])
        self.use_norm = boolean_string(self.config['use_norm'])
        self.use_half = self.config['use_half']
        self.writeCSV = self.config['writeCSV']
        self.drawDatasetsGT2predictedHeatmap = self.config['drawDatasetsGT2predictedHeatmap']
        self.drawDatasetsViolin = self.config['drawDatasetsViolin']
        self.drawGenesGT2predictedViolin = self.config['drawGenesGT2predictedViolin']
        self.changeNegativeProts2Zero = self.config['changeNegativeProts2Zero']
        self.only_val_by_net_config = boolean_string(self.config['only_val_by_net_config'])
        if self.only_val_by_net_config:
            print('only val genes mode')
        self._metrics = {}
        self._rna_values = {}
        
        for metric_file in self.metric_files:
            self.process_metric_file(
                metric_file, only_val=self.only_val_by_net_config
            )
       
        if self.drawDatasetsGT2predictedHeatmap:
            print('drawDatasetsGT2predictedHeatmap...') 
            self.datasets_heatmap()
        if self.drawDatasetsViolin:
            print('drawDatasetsViolin...') 
            self.datasets2violin()
        if self.drawGenesGT2predictedViolin: 
            print('drawGenesGT2predictedViolin...') 
            self.genes2violin()
        
    
    def metrics(self):
        return self._metrics

    def net_names(self):
        return list(self._metrics.keys())
    
    def datasets_names(self):
        out = []
        for k, net_name in enumerate(self.net_names()):
            for dataset_name in self._metrics[net_name].keys():
                if dataset_name not in out:
                    out.append(dataset_name)
        return out
    
    def experiments_names(self):
        out = []
        for k, net_name in enumerate(self.net_names()):
            for dataset_name in self._metrics[net_name].keys():
                net_data = self._metrics[net_name][dataset_name]
                for x in net_data.keys():
                    if x not in out:
                        out.append(x)
        return out
    
    def gene_experiments_names(self):
        out = []
        for k, net_name in enumerate(self.net_names()):
            for dataset_name in self._metrics[net_name].keys():
                net_data = self._metrics[net_name][dataset_name]
                for x in net_data.keys():
                    if 'uid.' in x and x not in out:
                        out.append(x)
        return out
    
    def rna_values(self):
        return self._rna_values
    
    def genes2violin(
        self
    ):
        net_names = self.net_names()
        exps = [None] * len(net_names)
        dataset_header_name = None
        for k, net_name in enumerate(net_names):
            for dataset_name in self._metrics[net_name].keys():
                dataset_header_name = dataset_name
                
                net_data = self._metrics[net_name][dataset_name]
                keys2process = [
                    x for x in net_data.keys() 
                    if 'uid.' in x and dataset_name not in x
                ]
                for gene_name in keys2process:
                    gene_exps = flat_list([
                        net_data[gene_name][x] for x in net_data[gene_name].keys()
                    ])
                    # if len(gene_exps) <= 5:
                    if len(gene_exps) <= 10:
                        continue
                    gts, preds = (
                        [x[0] for x in gene_exps], 
                        [x[1] for x in gene_exps]
                    )
                    try:
                        pearson_scipy_v, _ = pearsonr(preds, gts)
                    except ValueError:
                        continue
                    if is_number(pearson_scipy_v):
                        if exps[k] is None:
                            exps[k] = []
                        exps[k].append(pearson_scipy_v**2)
                        # debug(k)
                        # debug(pearson_scipy_v**2)
                        # debug(preds)
                        # debug(gts)
        clr = self.dataset2clr(dataset_header_name)
        violin(
            exps, 
            [
                'ResNet34V1.3',
                'ResNet34V2',
                'ResNet50V2'
            ],
            facecolor=clr,
        )

    def dataset2clr(self, dataset_header_name):
        if 'tissue29' in dataset_header_name:
            clr = '#D43F3A'
        elif 'nci60' in dataset_header_name:
            clr = '#D91C9B'
        else:
            clr = 'green'
        return clr
                
    def datasets2violin(self):
        for net_name in self._metrics.keys():
            for dataset_name in self._metrics[net_name].keys():
                clr = self.dataset2clr(dataset_name)
                violin_plot_exps_gt(
                    self._metrics,
                    dataset_name,
                    max_in_one=12,
                    facecolor=clr,
                    key2process='protein_gt_'
                )
                violin_plot_exps_gt(
                    self._metrics,
                    dataset_name,
                    max_in_one=12,
                    # facecolor='green',
                    facecolor='orange',
                    key2process='rna_gt_'
                )
                # violin_plot_exps_gt(
                #     self._metrics,
                #     dataset_name,
                #     max_in_one=12,
                #     # facecolor='green',
                #     facecolor='orange',
                #     key2process='_gt_',
                #     sum2rna_prot=True
                # )
    
    def datasets_heatmap(self):
        datasets = []
        for k, v in self._metrics.items():
            datasets += list(v.keys())
        datasets = set(datasets)
        for dataset_name in datasets:
            self.dataset_heatmap(dataset_name)
    
    def dataset_heatmap(
        self, 
        dataset_name,
        max_in_one=None
    ):
        if not max_in_one:
            if dataset_name == 'tissue29':
                max_in_one = 8
            else:
                max_in_one = 10
        net_names = list(self._metrics.keys())
        exps = []
        for net_name in net_names:
            net_data = self._metrics[net_name][dataset_name]
            keys2process = [
                x for x in net_data.keys() 
                if dataset_name in x and '_gt_' not in x
            ]
            for key in keys2process:
                if key not in exps:
                    exps.append(key)
        exps_patches = list2chunks(exps, max_in_one)
        for patch in exps_patches:
            matrix = np.zeros(shape=(len(net_names), len(patch)))
            for j, net_name in enumerate(net_names):
                net_data = self._metrics[net_name][dataset_name]
                for i, exp_name in enumerate(patch):
                    matrix[j][i] = net_data[exp_name]
            patch2print = [x.replace(dataset_name+'_', '') for x in patch]
            matrix2heatmap(
                matrix=matrix,
                # v_header=net_names,
                v_header=[
                    'ResNet34V1.3',
                    'ResNet26V2',
                    'ResNet50V2',
                    # 'ResNet26V2_U',
                ],
                h_header=patch2print,
                # cmap='RdYlGn',
                cmap='rainbow',
                # cmap='PuOr',
                cbarlabel='R^2 regression coef'
            )
                    
    def process_metric_file(self, metric_file, only_val=False):
        metric_dir = os.path.dirname(metric_file)
        name_parts = os.path.basename(metric_file).split('_')
        dataset_name = None
        net_name_parts = name_parts[:-2]
        net_name = ''
        if only_val:
            # metrics dir: MODEL_NAME/out/METRICS_FILE
            # net_dir:     MODEL_NAME/
            net_dir = os.path.split(
                os.path.split(metric_dir)[0]
            )[0]
            config_p = find_files(net_dir, '_config', abs_p = True)[0]
            print(f'ONLY VAL MODE. config: {config_p}')
            with open(config_p, 'r') as f:
                config = json.load(f)
                f.close()
            genes2val = config['genes2val']
            # print(f'genes2val:: {genes2val}')

        for i, p in enumerate(net_name_parts):
            if i != len(net_name_parts) -1:
                net_name += p + '_'
            else:
                net_name += p
        
        if self.writeCSV:
            table_filepath = os.path.join(
                metric_dir, 
                'table_metric_{}.csv'.format(self.st_time)
            )
            table_tissues_filepath = os.path.join(
                metric_dir, 
                'table_tissues_metric_{}.csv'.format(self.st_time)
            )
            table_tissues_overall_filepath = os.path.join(
                metric_dir, 
                'table_tissues_overall_metric_{}.csv'.format(self.st_time)
            )
        if self.use_norm:
            metric_filepath = os.path.join(
                metric_dir, 
                'metric_norm_{}.txt'.format(self.st_time)
            )
        else:
            metric_filepath = os.path.join(
                metric_dir, 
                'metric_{}.txt'.format(self.st_time)
            )
            

        logger = TrainLogger(metric_filepath)
        if self.writeCSV:
            table_logger = TrainLogger(table_filepath)
            table_tissues_logger = TrainLogger(table_tissues_filepath)
            table_tissues_overall_logger = TrainLogger(table_tissues_overall_filepath)
        logger.print(metric_file)
        if self.select_genes:
            selected_genes_filepath = os.path.join(
                metric_dir, 
                'metric_{}_selected.json'.format(self.st_time)
            )
            selected_genes = {}
        uids = []
        rna_exps = []
        rna_values = []
        prot_exps = []
        labels = []
        preds = []
        with open(metric_file, 'r') as f:
            f.readline() # skip header
            ds = f.readlines()
        z_prots = 0
        lz_prots_pred = 0
        data_cnt = len(ds)
        for j, d in enumerate(ds):
            if self.use_half:
                if self.use_half == 1:
                    if j > int(data_cnt/2):
                        break
                else:
                    if j < int(data_cnt/2):
                        continue
                    
            d = d[:-1]
            data = d.split('\t')
            pred_str = data[5]
            uid = data[0]
            if only_val:
                if uid not in genes2val:
                    continue
            rna_value = float(data[2])
            if self.use_norm:
                pred_value = norm_shifted_log(tensor_str2float(pred_str))
                prot_value = norm_shifted_log(float(data[4]))
                rna_value = norm_shifted_log(rna_value)
            else:
                pred_value = tensor_str2float(pred_str)
                prot_value = float(data[4])
            prot_exp = data[3]
            rna_exp = data[1]

            if self.select_genes:
                if is_measurement_negative(pred_str):
                    if uid not in selected_genes:
                        selected_genes[uid] = []
                    selected_genes[uid].append(
                        '{}->{}::is_measurement_negative'.format(
                            rna_exp,
                            prot_exp
                        )
                    ) 
            if prot_value == 0.0:
                z_prots += 1 
                continue

            if pred_value < 0.0:
                if self.changeNegativeProts2Zero:
                    pred_value = 0.0
                    # print(pred_value, '-> 0.0')
                else:
                    lz_prots_pred += 1 
                
            preds.append(pred_value)
            uids.append(uid)
            rna_exps.append(rna_exp)
            prot_exps.append(prot_exp)
            rna_values.append(rna_value)
            labels.append(prot_value)
        
            if self.isdebug and len(uids) > 1000:
                break

        if self.select_genes:
            with open(selected_genes_filepath, 'w') as f:
                json.dump(selected_genes, f, indent=4)
                logger.print('selected genes dumped::')
                logger.print('{}'.format(selected_genes_filepath))

        logger.print('{} genes'.format(len(set(uids))))
        logger.print('{} experiments read'.format(len(labels)))
        logger.print('{} zero prots experiments'.format(z_prots))
        logger.print('{} less zero prots predicted'.format(lz_prots_pred))

        # # =========== RAW VIZ
        # from pylab import *
        # figure(1)
        # # plot(preds, labels, color='red', marker='o', linestyle='', s=0.3)
        # scatter(preds, labels, color='red', marker='o', s=0.3)
        # xlabel('predicted')
        # ylabel('gt')
        # grid()
        # out = '/home/ilpech/datasets/test/test_regr.png'
        # savefig(out, dpi=72)
        # print(f'saved in {out}')
        # exit()

        pearson_scipy_v, pearson_square_scipy_v, spearman_scipy_v = self.calculate_net_metrics(
            preds, labels
        )
        logger.print('pearson_scipy_v == {}'.format(pearson_scipy_v))
        logger.print('pearson_scipy_v^2 == {}'.format(pearson_square_scipy_v))
        logger.print('spearman_scipy_v == {}'.format(spearman_scipy_v))
        
        header = [
            'Genes_cnt',
            'Experiments',
            'R^2',
            'Spearman'
        ]
        metrics = (
            len(set(uids)), 
            len(preds),
            pearson_square_scipy_v, 
            spearman_scipy_v
        )
        if self.writeCSV:
            table_logger.writeAsCSV(header, [metrics])
        labels_list = labels
        preds_list = preds
        labels = np.array(labels)
        preds = np.array(preds)
        
        overall_metric = pearson_scipy_v ** 2
        logger.print('=============')
        logger.print('overall(mx.metric.PearsonCorrelation^2) {:.4f}'.format(overall_metric))
        self.genes_gt2predicted = {}
        
        experiment_names, exps_p_square_metrics, exps_sperman_metrics, num_genes_per_experiments, net_metrics, net_rna_values = \
            self.calculate_net_metrics_per_tissue(
                net_name=net_name,
                rna_values = rna_values, 
                uids = uids, 
                prot_exps = prot_exps, 
                labels = labels, 
                preds = preds
            )
            
        self._metrics[net_name] = net_metrics[net_name]
        self._rna_values[net_name] = net_rna_values[net_name]
        
        logger.print('=============')
        logger.print('experiments metrics(PearsonCorrelation^2 | SpearmanCorrelation)::')
        for i in range(len(experiment_names)):
            logger.print('. ({}){} = {:.4f} | {:.4f}'.format(
                num_genes_per_experiments[i], 
                experiment_names[i], 
                exps_p_square_metrics[i],
                exps_sperman_metrics[i]
            ))
        
        print('{} experiments found'.format(len(experiment_names)))
        if self.writeCSV:
            table_tissues_logger.writeAsCSV(
                experiment_names,
                [exps_p_square_metrics]
            )
        
        # not sure
        if dataset_name is None:
            dataset_name = prot_exps[0].split('_')[0]
            
        for i, exp in enumerate(header):
            self._metrics[net_name][dataset_name][exp] = metrics[i]
        for i, exp in enumerate(experiment_names):
            dataset_name = exp.split('_')[0]
            self._metrics[net_name][dataset_name][exp] = exps_p_square_metrics[i]
        
        #=================
        # sample_matrix = np.array(exps_metrics * 3)
        # sample_matrix = sample_matrix.reshape((3, len(exps_metrics)))

        # matrix2heatmap(
        #     matrix=sample_matrix,
        #     v_header=['m1', 'm2', 'm3'],
        #     h_header=experiment_names,
        #     # cmap='PuOr',
        #     cmap='seismic',
        #     cbarlabel='R^2 regression coef'
        # )
        exps_p_square_metrics = np.array(exps_p_square_metrics)
        logger.print('min(PearsonCorrelation^2) {:.4f}'.format(exps_p_square_metrics.min()))
        logger.print('max(PearsonCorrelation^2) {:.4f}'.format(exps_p_square_metrics.max()))
        logger.print('mean(PearsonCorrelation^2) {:.4f}'.format(exps_p_square_metrics.mean()))
        logger.print('=============')
        if self.writeCSV:
            table_tissues_overall_logger.writeAsCSV(
                [
                    'min(R^2(tissues))',
                    'max(R^2(tissues))',
                    'mean(R^2(tissues))'
                ],
                [
                    [
                        exps_p_square_metrics.min(),
                        exps_p_square_metrics.max(),
                        exps_p_square_metrics.mean(),
                    ]
                ]
            )
            
    @staticmethod
    def calculate_net_metrics(preds, labels):
        pearson_scipy_v, _ = pearsonr(preds, labels)
        spearman_scipy_v, _ = spearmanr(preds, labels)
        return pearson_scipy_v, pearson_scipy_v**2, spearman_scipy_v
        
            
    @staticmethod
    def calculate_net_metrics_per_tissue(
        net_name, 
        uids: list, 
        rna_values: list, 
        prot_exps: list, 
        labels: np.ndarray, 
        preds: np.ndarray
    ):
        import time
        prot_exps_alph = sorted(set(prot_exps))
        exps_pearson_r_square_metrics = []
        exps_spearman_metrics = []
        experiment_names = []
        experiment_names_no_cnt = []
        num_genes_per_experiments = []
        
        net_metrics = {}
        net_rna_values = {}
        
        if net_name not in net_metrics:
            net_metrics[net_name] = {}
            
        if net_name not in net_rna_values:
            net_rna_values[net_name] = {}
        
        for l, exp in enumerate(prot_exps_alph):
            dataset_name = exp.split('_')[0]
            net_metrics[net_name][dataset_name] = {}
            net_rna_values[net_name][dataset_name] = {}

            ids2use = [
                i for i in range(len(labels)) if prot_exps[i] == exp
            ]

            # exp_labels = [
            #     labels[i] for i in range(len(labels)) if i in ids2use
            # ]   
            # exp_labels_m = np.array(exp_labels)
            exp_labels_m = np.take(labels, np.array(ids2use))
            exp_labels = exp_labels_m.tolist()

            # exp_preds = [
            #     preds[i] for i in range(len(preds)) if i in ids2use
            # ]
            # exp_preds_m = np.array(exp_preds)
            exp_preds_m = np.take(preds, np.array(ids2use))
            
            uids2use = np.take(
                np.array(uids), np.array(ids2use)
            )
            gene_names = [
                f'uid.{i}' for i in uids2use
            ]
            start_t = time.time()
            exp_rna = np.take(
                np.array(rna_values),
                np.array(ids2use)
            ).tolist()

            for d, gene_name in enumerate(gene_names):
                if gene_name not in net_metrics[net_name][dataset_name]:
                    net_metrics[net_name][dataset_name][gene_name] = {}
                
                if gene_name not in net_rna_values[net_name][dataset_name]:
                    net_rna_values[net_name][dataset_name][gene_name] = {}
                    
                if exp not in net_metrics[net_name][dataset_name][gene_name]:
                    net_metrics[net_name][dataset_name][gene_name][exp] = []
                    
                if exp not in net_rna_values[net_name][dataset_name][gene_name]:
                    net_rna_values[net_name][dataset_name][gene_name][exp] = []
                    
                net_metrics[net_name][dataset_name][gene_name][exp].append((
                    exp_labels_m[d],
                    exp_preds_m[d]
                ))
                
                net_rna_values[net_name][dataset_name][gene_name][exp].append((
                    exp_rna[d]
                ))

            try:
                pears_v, _ = pearsonr(exp_labels_m, exp_preds_m)
                pears_square_v = pears_v ** 2
                spearm, _ = spearmanr(exp_labels_m, exp_preds_m)
            except:
                pears_v, pears_square_v, spearm = None, None, None
            
            # experiment_names.append(
            #     '{}({})'.format(
            #         exp, 
            #         len(exp_labels), 
            #     )
            # ) 
            experiment_names.append(exp) 
            experiment_names_no_cnt.append(
                    exp
            ) 
            exps_pearson_r_square_metrics.append(pears_square_v)
            exps_spearman_metrics.append(spearm)
            num_genes_per_experiments.append(len(exp_labels_m))
            if len(exp_rna):
                exp_name = 'rna_gt_{}'.format(exp)
                if exp_name not in net_metrics[net_name][dataset_name]:
                   net_metrics[net_name][dataset_name][exp_name] = []
                net_metrics[net_name][dataset_name][exp_name] += exp_rna
            if len(exp_labels):
                exp_name = 'protein_gt_{}'.format(exp)
                if exp_name not in net_metrics[net_name][dataset_name]:
                   net_metrics[net_name][dataset_name][exp_name] = []
                net_metrics[net_name][dataset_name][exp_name] += exp_labels
        return experiment_names, exps_pearson_r_square_metrics, exps_spearman_metrics, num_genes_per_experiments, net_metrics, net_rna_values
        
            
    # def tissue29_nci60_mapping(self, tissue_mapping_file_path, gene_match=True):
    #     self.mapping_processor = ExptsMapping()
        
    #     with open(tissue_mapping_file_path) as json_file:
    #         mapping_dict = json.load(json_file)
        
    #     net_names = self.net_names()

    #     tissue29_prots_dict = None
    #     tissue29_rna_dict = None
    #     nci60_prots_dict = None
    #     nci60_rna_dict = None

    #     for net_name in net_names:
    #         prots_dict2process = self._metrics[net_name]
    #         rna_dict2process = self._rna_values[net_name]

    #         try:
    #             tissue29_prots_dict = prots_dict2process['tissue29']
    #             tissue29_rna_dict = rna_dict2process['tissue29']
    #         except:
    #             print('{} have not measurements for tissue29'.format(net_name))

    #         try:
    #             nci60_prots_dict = prots_dict2process['nci60']
    #             nci60_rna_dict = rna_dict2process['nci60']
    #         except:
    #             print('{} have not measurements for nci60'.format(net_name))

    #     for gene in tissue29_prots_dict.keys():
    #         if 'uid.' in gene:
    #             tissue29_gene2check_dict = tissue29_prots_dict[gene]
    #             for tissue in tissue29_gene2check_dict.keys():
    #                 if tissue in mapping_dict.keys():
    #                     nci_tissues2check = mapping_dict[tissue]
    #                     if len(nci_tissues2check) != 0:
    #                         if gene in nci60_prots_dict.keys():
    #                             nci60_gene2check_dict = nci60_prots_dict[gene]
    #                             for nci_tissue in nci_tissues2check:
    #                                 if nci_tissue in nci60_gene2check_dict.keys():
    #                                     matching_expt = MatchingExpt(
    #                                         tissue, 
    #                                         tissue29_rna_dict[gene][tissue][0],
    #                                         tissue, 
    #                                         tissue29_prots_dict[gene][tissue][0][0], 
    #                                         tissue29_prots_dict[gene][tissue][0][1],
    #                                         gene, 
                                            
    #                                         nci_tissue, 
    #                                         nci60_rna_dict[gene][nci_tissue][0],
    #                                         nci_tissue, 
    #                                         nci60_prots_dict[gene][nci_tissue][0][0], 
    #                                         nci60_prots_dict[gene][nci_tissue][0][1],
    #                                         gene,     
    #                                     )     
    #                                     self.mapping_processor.add_tissue_uid_match_expt(matching_expt)
    #     print(self.mapping_processor.uids_matching_info()) 
               
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)

    opt = parser.parse_args()

    metric_processor = MetricInferenced(opt.config_path)