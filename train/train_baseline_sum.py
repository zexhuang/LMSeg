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
from model.net import GANet, HGAPNet, LGAPNet
from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/sum/sum_hgap_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/SUM',
                        help='path to dataset folder')
    args = parser.parse_args()
    
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)    
        
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
        
        if cfg['model'] == 'GA':
            model = GANet(cfg['in_channels'], cfg['out_channels'],
                          cfg['hid_channels'], 
                          cfg['num_convs'], 
                          cfg['pool_factors'], 
                          cfg['num_nbrs'],
                          cfg['num_block'])
        elif cfg['model'] == 'HGAP':
            model = HGAPNet(cfg['in_channels'], cfg['out_channels'],
                            cfg['hid_channels'], 
                            cfg['num_convs'], 
                            cfg['pool_factors'], 
                            cfg['num_nbrs'],
                            cfg['num_block'],
                            cfg['alpha'], 
                            cfg['beta'])
        elif cfg['model'] == 'LGAP':
            model = LGAPNet(cfg['in_channels'], cfg['out_channels'],
                            cfg['hid_channels'], 
                            cfg['num_convs'], 
                            cfg['pool_factors'], 
                            cfg['num_block'],
                            cfg['alpha'], 
                            cfg['beta'])
        else:
            raise Exception("select correct model")
        
        labels = torch.hstack([data.y for data in train_loader])    
        weight = torch.bincount(labels) / torch.sum(labels)
        loss = CrossEntropyWithLabelWeight(ignore_index=0, 
                                           label_smoothing=0.1, 
                                           label_weights=1 / torch.log(1.2 + weight))
    
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
        trainer.device = 'cpu'
        trainer.eval(model, 
                     test_loader, 
                     metric=metric_dict, 
                     ckpt=f"epoch{cfg['epoch']}",
                     verbose=True)