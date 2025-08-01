import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

import torch
from torch_geometric.loader import DataLoader
from data.dataset import SUMDataset
from torchmetrics.classification import (Accuracy, 
                                         JaccardIndex)
from model.utils.loss import CrossEntropyWithLabelWeight
from model.net import LMSegNet
from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/sum/sum_lmseg_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/SUM',
                        help='path to dataset folder')
    parser.add_argument('--path', type=str,  metavar='N',
                        default=None,
                        help='path to save model')
    args = parser.parse_args()
    
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)   
         
        if args.path is not None:
            cfg['path'] = args.path
            
        print("\nLoaded Configuration:\n" + "="*25)
        print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
        print("="*25 + "\n")
        
        train_set = SUMDataset(root=args.root, 
                               split='train', 
                               load_feature=cfg['load_feature'])
        val_set = SUMDataset(root=args.root, 
                             split='validate', 
                             load_feature=cfg['load_feature'])
        test_set = SUMDataset(root=args.root, 
                              split='test', 
                              load_feature=cfg['load_feature'])
                    
        train_loader = DataLoader(train_set, 
                                  batch_size=cfg['batch'], 
                                  shuffle=True, 
                                  num_workers=cfg['workers'])   
        val_loader = DataLoader(val_set, 
                                batch_size=cfg['batch'], 
                                shuffle=False, 
                                num_workers=cfg['workers'])
        test_loader = DataLoader(test_set, 
                                 batch_size=cfg['batch'], 
                                 shuffle=False, 
                                 num_workers=cfg['workers'])
        
        model = LMSegNet(cfg['in_channels'], cfg['out_channels'],
                         cfg['hid_channels'], 
                         cfg['num_convs'], 
                         cfg['pool_factors'], 
                         cfg['num_nbrs'],
                         cfg['num_block'],
                         cfg['alpha'], 
                         cfg['beta'])
        
        all_labels = torch.cat([data.y for data in train_set])
        class_freq = torch.bincount(all_labels, minlength=cfg['out_channels']).float()
        class_weight = 1.0 / torch.log(1.2 + (class_freq / class_freq.sum()))
        class_weight[0] = 0.0

        loss = CrossEntropyWithLabelWeight(ignore_index=0, 
                                           label_smoothing=0.1, 
                                           label_weights=class_weight)
        
        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    criterion=loss,
                    train_loader=train_loader, 
                    val_loader=val_loader)
        
        metric_dict = {
            'OA': Accuracy(task="multiclass", 
                           num_classes=cfg['out_channels'], 
                           ignore_index=0,
                           average='micro'),
            'mAcc': Accuracy(task="multiclass", 
                             num_classes=cfg['out_channels'], 
                             ignore_index=0,
                             average='macro'),
            'mIoU': JaccardIndex(task="multiclass", 
                                 num_classes=cfg['out_channels'], 
                                 ignore_index=0,
                                 average='macro'),
            'IoU': JaccardIndex(task="multiclass", 
                                num_classes=cfg['out_channels'], 
                                ignore_index=0,
                                average=None),
        }
        trainer.eval(model, 
                     test_loader, 
                     metric=metric_dict, 
                     ckpt="best_val_epoch.pth",
                     verbose=True)