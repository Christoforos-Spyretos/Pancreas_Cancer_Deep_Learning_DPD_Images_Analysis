# imports
from __future__ import print_function
import argparse
import pdb
import os
import math
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import warnings

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

warnings.filterwarnings("ignore", category=UserWarning)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_experiment_name(cfg):
    return '_'.join([str(cfg.seed),
                    str(cfg.stain_modality.description),
                    str(cfg.n_classes),
                    str(cfg.label_dict),
                    str(cfg.data_root_dir),
                    str(cfg.csv_path),
                    str(cfg.patient_strat),
                    str(cfg.shuffle),
                    str(cfg.print_info), 
                    str(cfg.embed_dim), 
                    str(cfg.max_epochs), 
                    str(cfg.lr), 
                    str(cfg.lr_scheduler),
                    str(cfg.label_frac), 
                    str(cfg.reg),
                    str(cfg.k),
                    str(cfg.k_start),
                    str(cfg.k_end),
                    str(cfg.results_dir),
                    str(cfg.split_dir),
                    str(cfg.log_data),
                    str(cfg.testing),
                    str(cfg.early_stopping),
                    str(cfg.opt),
                    str(cfg.drop_out),
                    str(cfg.bag_loss),
                    str(cfg.model_type),
                    str(cfg.exp_code),
                    str(cfg.weighted_sample),
                    str(cfg.use_class_weights),
                    str(cfg.model_size),
                    str(cfg.feature_type),
                    str(cfg.task),
                    str(cfg.no_inst_cluster),
                    str(cfg.inst_loss),
                    str(cfg.subtyping),
                    str(cfg.bag_weight),
                    str(cfg.B),
                    str(cfg.ignore)])

@hydra.main(version_base="1.3.2", 
			config_path= '/local/data1/chrsp39/Pancreas_Cancer_Deep_Learning_DPD_Images_Analysis/configs/evaluation',
			config_name='run_model')

def main(cfg:DictConfig):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(cfg.seed)

    experiment_name = build_experiment_name(cfg)

    settings = {'num_splits': cfg.k, 
            'k_start': cfg.k_start,
            'k_end': cfg.k_end,
            'task': cfg.task,
            'max_epochs': cfg.max_epochs, 
            'results_dir': cfg.results_dir, 
            'lr': cfg.lr,
            'lr_scheduler':cfg.lr_scheduler,
            'experiment': cfg.exp_code,
            'reg': cfg.reg,
            'label_frac': cfg.label_frac,
            'bag_loss': cfg.bag_loss,
            'seed': cfg.seed,
            'model_type': cfg.model_type,
            'model_size': cfg.model_size,
            "use_drop_out": cfg.drop_out,
            'weighted_sample': cfg.weighted_sample,
            'feature_type': cfg.feature_type,
            'use_class_weights':cfg.use_class_weights,
            'opt': cfg.opt}

    if cfg.ignore is None:
        cfg.ignore = [] # Set to an empty list if None

    # create results directory if necessary
    if not os.path.isdir(cfg.results_dir):
        os.mkdir(cfg.results_dir)

    if cfg.model_type in ['abmil','clam_sb','clam_mb']:
        settings.update({'bag_weight': cfg.bag_weight,
                    'inst_loss': cfg.inst_loss,
                    'B': cfg.B})

    with open_dict(cfg):
        cfg.n_classes = cfg.n_classes
        cfg.subtyping = cfg.subtyping

    print('\nLoad Dataset')
    if cfg.task:
        print(f'Task description: {cfg.task}')

    dataset = Generic_MIL_Dataset(csv_path = cfg.csv_path,
                                data_dir= os.path.join(cfg.stain_modality.data_root_dir),
                                shuffle = cfg.shuffle, 
                                seed = cfg.seed, 
                                print_info = cfg.print_info,
                                label_dict = cfg.label_dict,
                                patient_strat=cfg.patient_strat,
                                ignore=cfg.ignore)

    if cfg.model_type in ['abmil','clam_sb','clam_mb', 'mil']:
        assert cfg.subtyping 

    if not os.path.isdir(cfg.results_dir):
        os.mkdir(cfg.results_dir)

    cfg.results_dir = os.path.join(cfg.results_dir, str(cfg.stain_modality.exp_code) + '_s{}'.format(cfg.seed))
    if not os.path.isdir(cfg.results_dir):
        os.mkdir(cfg.results_dir)

    if cfg.split_dir is None:
        cfg.split_dir = os.path.join('splits', cfg.task+'_{}'.format(int(cfg.label_frac*100)))
    else:
        cfg.split_dir = os.path.join('splits', cfg.split_dir)

    print('split_dir: ', cfg.split_dir)
    assert os.path.isdir(cfg.split_dir)

    settings.update({'split_dir': cfg.split_dir})

    with open(cfg.results_dir + '/experiment_{}.txt'.format(cfg.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val)) 

    if cfg.k_start == -1:
        start = 0
    else:
        start = cfg.k_start
    if cfg.k_end == -1:
        end = cfg.k
    else:
        end = cfg.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)

    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(cfg.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, cfg)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(cfg.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != cfg.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(cfg.results_dir, save_name))

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")