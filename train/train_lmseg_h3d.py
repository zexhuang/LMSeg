import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

import torch
from torch_geometric.loader import DataLoader
from data.dataset import H3DDataset
from torchmetrics.classification import Accuracy, F1Score

from model.utils.loss import CrossEntropyWithLabelWeight
from model.net import GANet, HGAPNet, LGAPNet, LMSegNet
from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/h3d/lmseg_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/H3DBenchmark/',
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
        
        epochs = ['Epoch_March2019', 'Epoch_November2018', 'Epoch_March2018']
        train_data = [H3DDataset(root=args.root, split='train', epoch=ep) for ep in epochs]
        train_set = [data for dataset in train_data for data in dataset]
        
        val_set = H3DDataset(root=args.root, split='val', epoch='Epoch_March2018')
             
        train_loader = DataLoader(train_set, 
                                  batch_size=cfg['batch'], 
                                  shuffle=True, 
                                  pin_memory=True,
                                  num_workers=cfg['workers'])   
        val_loader = DataLoader(val_set, 
                                batch_size=cfg['batch'], 
                                shuffle=False, 
                                pin_memory=True,
                                num_workers=cfg['workers'])
        test_loader = DataLoader(val_set, 
                                 batch_size=cfg['batch'], 
                                 shuffle=False, 
                                 pin_memory=True,
                                 num_workers=cfg['workers'])
        
        if 'model' in cfg:
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
            model = LMSegNet(cfg['in_channels'], cfg['out_channels'],
                             cfg['hid_channels'], 
                             cfg['num_convs'], 
                             cfg['pool_factors'], 
                             cfg['num_nbrs'],
                             cfg['num_block'],
                             cfg['alpha'], 
                             cfg['beta'],
                             cfg['load_feature'])
        ignore_index = 11
        all_labels = torch.cat([data.y for data in train_set])
        class_freq = torch.bincount(all_labels, minlength=cfg['out_channels']).float()
        class_weight = 1.0 / torch.log(1.2 + (class_freq / class_freq.sum()))
        class_weight[ignore_index] = 0.0

        loss = CrossEntropyWithLabelWeight(ignore_index=ignore_index, 
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
                           ignore_index=ignore_index,
                           average='micro'),
            'mF1': F1Score(task="multiclass", 
                           num_classes=cfg['out_channels'], 
                           ignore_index=ignore_index,
                           average='macro'),
            'F1': F1Score(task="multiclass", 
                          num_classes=cfg['out_channels'], 
                          ignore_index=ignore_index,
                          average=None),
        }
        trainer.eval(model, 
                     test_loader, 
                     metric=metric_dict, 
                     ckpt=f"epoch{cfg['epoch']}.pth",
                     verbose=True)