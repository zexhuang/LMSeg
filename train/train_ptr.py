import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

from torch_geometric.loader import DataLoader
from data.dataset import BudjBimWallMeshDataset

from model.PointTransformer.net import PointTransformer
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/ptr_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/BudjBimWall',
                        help='path to dataset folder')
    args = parser.parse_args()
    
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)    
        
        print("\nLoaded Configuration:\n" + "="*25)
        print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
        print("="*25 + "\n")
        
        train_set = BudjBimWallMeshDataset(root=args.root, split='train')
        val_set = BudjBimWallMeshDataset(root=args.root, split='val')
        test_set = BudjBimWallMeshDataset(root=args.root, split='test')
            
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
        
        model = PointTransformer(cfg['in_channels'], 
                                 cfg['out_channels'], 
                                 cfg['hid_channels'], 
                                 cfg['pool_ratio'], 
                                 cfg['num_nbrs'])

        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    criterion=BCELogitsSmoothingLoss(),
                    train_loader=train_loader, 
                    val_loader=val_loader)
        trainer.eval(model, 
                     test_loader, 
                     ckpt=f"epoch{cfg['epoch']}.pth",
                     verbose=True)