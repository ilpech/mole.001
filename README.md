# Installation

1. clone project
2. download file with Uniprot databases 
    [annotations mappings](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz)


3. unzip file and move to ./bio_dl/testdata/human_ids_mapping.tab path inside project repository root

4. create virtual environment:

    we offer you two ways to create virtual environment: venv or conda.

    **venv**
    
    create virtual environment:

    ```console
    $ python3 -m venv {PATH_TO_VENV}/mole.001
    ```
    activate venv:

 
    bash/zsh (POSIX):
    ```console  
    $ source {PATH_TO_VENV}/mole.001/bin/activate
    ```

    commands to activate virtual environment on other platforms and shells using venv you can find 
    [here](https://docs.python.org/3/library/venv.html):

    
    **conda**

    create virtual environment:

    ```console
    $ conda create -n mole.001 python=3.8
    ```

    activate venv:  

    ```console
    $ conda activate mole.001
    ```

    **after activation of virtual environment install libraries from requirements:**
    
    ```console 
    (mole.001) $ python3 -m pip install -r requirements.txt 
    ```

# Data analysis 

## ./bio_dl/gene.py

Data structure to store gene information with ability to add RNA and protein abundance experiments. 

Through Gene class object you can access Uniprot API data with apiData() functions and amino acids sequence with apiSequence(). 

Identification is done by Uniprot IDs.

## ./bio_dl/gene_mapping.py

Script using to process databases annotations mapping from Uniprot website. 

Note function uniprot_mapping_header() to get full list of databases names and function uniq_nonempty_uniprot_mapping_header() to use selected for model inference databases with many-to-many relations.

## ./bio_dl/inference_gene_regressor.py

Script to inference trained model with given epoch. Results are saved in file inside the model directory.

Run command example::
```console
$ python3 bio_dl/inference_gene_regressor.py --isdebug 0 --net_dir trained/rna2protein_nci60.002 --epoch 6 --data_config config/gene_expression/train_nci60.yaml --only_val 0 
```

## ./bio_dl/metric_inferenced.py

Run command example::
```console
$ python3 bio_dl/metric_inferenced.py --config_path config/metric_inferenced/metric_inferenced.yaml
```

## ./bio_dl/gene_observer.py

Sandbox script for observing experiments result and comparing it with actual dataset data. 

To run the script bio_dl/inference_gene_regressor.py you have firstly to prepare a file with model inference on data. 

MetricInferenced and DistGeneDataLoader objects are used to access model inference and data information.

Run command example::
```console
$ python3 bio_dl/gene_observer.py --metric_cfg config/metric_inferenced/metric_inferenced.yaml --data_cfg config/gene_expression/train.yaml
```
Argparse keys::
1. --metric_cfg - file resulting script bio_dl/inference_gene_regressor.py model inference 
2. --data_cfg - same as train config (used only config.data)

## ./torch_dl/dataloader/gene_data_loader.py

Genecentric RNA and protein experiments data loader launching by config from the folder config/gene_expression. 

Genes are loaded from .tab data files with ENSG IDs converting name to Uniprot ID if possible. 

Model inference data vectorization and caching is realized through gene2sample() function.

Data from annotations mapping for gene is accessed with dataFromMappingDatabase() function.

## ./torch_dl/dataloader/gene_batch_data_loader.py

Multiprocessing dataloader with batch iteration.

## ./torch_dl/train/train_gene_regression.py

Train script. Creating or finetuning model based on config from the folder ./config/gene_expression.

To turn on dataset use config.data.datasets2use, select True for corresponding dataset key. 

Used datasets are in .tab format with ENSG identifier and columns named by experiments like ./bio_dl/testdata/tissue29_rna.tsv.

You can prepare a predicting model for your own dataset. To do it, add a specific key with its name in config.data.datasets2use and in config.data. 

Models finetuning is also available through config.train.finetune_net_name and config.train.finetune_params_dir. To select epoch to start finetuning from use config.train.epoch_start_from. If this param equals zero, finetuning is off.

Run command example::
```console
$ python3 torch_dl/train/train_gene_regression.py --config config/gene_expression/train.yaml --isdebug False
```

Argparse keys::
1. --config - path to data and train config
2. --isdebug - run test mode (less experiments)  

# Config files 

## ./config/gene_expression/train.yaml

Data keys(config.data)::

1. cashPath - 
2. datasets2use - 
3. *_rna - 
4. *_prot - 
1. geneMapping - 
2. engs2uniprot_file - 
3. splitGeneFromModelConfig - 
4. ifSplitPercentToTrain - 
5. useZeroProt - 
6. useNonZeroProt -

Train keys(config.train)::

1. params_dir - 
2. max_var_layer - 
3. gpus2use - 
4. each_train_has_own_dataset - 
5. net_name - 
6. finetune_net_name - 
7. finetune_params_dir - 
8. use_finetune_experiments - 
9. epoch_start_from - 
10. epochs - 
11. log_interval - 
12. metric_flush_interval - 
13. wd - 
14. momentum - 
15. lr.mode - 
16. lr.metric2use - 
17. lr.byHand - 
18. lr.byPlateu - 
19. lr.byPlateu.mode - 
20. lr.cosineWarmup - 
21. lr.cosineWarmup.warmup_steps - 
22. lr.cosineWarmup.first_cycle_steps - 
23. lr.cosineWarmup.cycle_mult - 
24. lr.cosineWarmup.gamma - 
25. batch_size - 
26. model.model2use - 
27. model.model2use.wideResNet - 
28. model.model2use.ResNetV2 - 

## ./config/gene_expression/train_tissue29.yaml

Train config prepared for Tissue29 dataset.

## ./config/gene_expression/train_nci60.yaml

Train config prepared for NCI60 dataset.

## ./config/gene_expression/val.yaml

Same config as ./config/gene_expression/train.yaml for validation data settings

## ./config/metric_inferenced/metric_inferenced.yaml

Settings to run ./bio_dl/metric_inferenced.py script for metrics gathering on run results
