import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from data.dataset import BudjBimWallMeshDataset
from model.GraphUNet.net import GraphUNet
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/gunet_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/BBW',
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
        
        areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
        for area in areas:            
            train_set = BudjBimWallMeshDataset(root=args.root, split='train', test_area=area)
            val_set = BudjBimWallMeshDataset(root=args.root, split='val', test_area=area)
            test_set = BudjBimWallMeshDataset(root=args.root, split='test', test_area=area)
            
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
            test_loader = DataLoader(test_set, 
                                     batch_size=cfg['batch'], 
                                     shuffle=False, 
                                     pin_memory=True,
                                     num_workers=cfg['workers'])    
        
            model = GraphUNet(cfg['in_channels'], 
                              cfg['hid_channels'], 
                              cfg['out_channels'], 
                              cfg['depth'],
                              cfg['pool_ratios'],
                              cfg['sum_res'],
                              cfg['act'])

            trainer = Trainer(cfg=cfg)
            trainer.path = cfg['path'] + f'/{area}'
            trainer.fit(model, 
                        criterion=BCELogitsSmoothingLoss(),
                        train_loader=train_loader, 
                        val_loader=val_loader)
            trainer.eval(model, 
                         test_loader, 
                         metric={'f1': BinaryF1Score(), 'mIoU': BinaryJaccardIndex()},
                         ckpt=f"epoch{cfg['epoch']}.pth",
                         verbose=True)