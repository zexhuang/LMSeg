import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

from torch_geometric.loader import DataLoader
from data.dataset import BudjBimWallMeshDataset

from model.net import GANet, HGAPNet, LGAPNet, LMSegNet
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/lmseg_feature.yaml',
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
            trainer = Trainer(cfg=cfg)
            
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
                
            trainer.path = cfg['path'] + f'/{area}'
            trainer.fit(model, 
                        criterion=BCELogitsSmoothingLoss(),
                        train_loader=train_loader, 
                        val_loader=val_loader)
            trainer.eval(model, 
                         test_loader, 
                         ckpt=f"epoch{cfg['epoch']}.pth",
                         verbose=True)