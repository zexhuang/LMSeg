import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

from torch_geometric.loader import DataLoader
from data.dataset import BudjBimWallMeshDataset

from model.DeeperGCN.net import DeeperGCN
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/bbw_deepergcn_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/BudjBimWall',
                        help='path to dataset folder')
    args = parser.parse_args()
    
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)    
        
        train_set = BudjBimWallMeshDataset(root=args.root, 
                                           split='train', 
                                           load_feature=cfg['load_feature'])
        val_set = BudjBimWallMeshDataset(root=args.root, 
                                         split='val', 
                                         load_feature=cfg['load_feature'])
        test_set = BudjBimWallMeshDataset(root=args.root, 
                                          split='test', 
                                          load_feature=cfg['load_feature'])
            
        train_loader = DataLoader(train_set, 
                                  batch_size=cfg['batch'], 
                                  shuffle=True, 
                                  num_workers=cfg['workers'])   
        val_loader = DataLoader(val_set, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=cfg['workers'])
        test_loader = DataLoader(test_set, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=cfg['workers'])
        
        model = DeeperGCN(cfg['in_channels'], 
                          cfg['out_channels'],
                          cfg['hid_channels'], 
                          cfg['num_layers'])

        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    criterion=BCELogitsSmoothingLoss(),
                    train_loader=train_loader, 
                    val_loader=val_loader)
        trainer.eval(model, 
                     test_loader, 
                     ckpt=f"epoch{cfg['epoch']}",
                     verbose=True)