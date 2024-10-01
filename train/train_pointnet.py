import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from data.dataset import BudjBimWallMeshDataset

from model.PointNet.net import PointNetSeg
from model.PointNet.pointnet_utils import BCERegLoss

from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/bbw_pointnet_feature.yaml',
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
        train_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
        
        val_set = BudjBimWallMeshDataset(root=args.root, 
                                         split='val', 
                                         load_feature=cfg['load_feature'])
        val_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
        
        test_set = BudjBimWallMeshDataset(root=args.root, 
                                          split='test', 
                                          load_feature=cfg['load_feature'])
        test_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
            
        train_loader = DataLoader(train_set, 
                                  batch_size=cfg['batch'], 
                                  shuffle=True, 
                                  num_workers=cfg['workers'],
                                  drop_last=True)   
        val_loader = DataLoader(val_set, 
                                batch_size=cfg['batch'], 
                                shuffle=False, 
                                num_workers=cfg['workers'],
                                drop_last=False)
        test_loader = DataLoader(test_set, 
                                 batch_size=cfg['batch'], 
                                 shuffle=False, 
                                 num_workers=cfg['workers'],
                                 drop_last=False)
        
        model = PointNetSeg(cfg['in_channels'], 
                            cfg['out_channels'],
                            get_trans_feat=True)
        loss = BCERegLoss()

        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    criterion=loss,
                    train_loader=train_loader, 
                    val_loader=val_loader)
        
        model.get_trans_feat = False
        trainer.eval(model, 
                     test_loader, 
                     ckpt=f"epoch{cfg['epoch']}",
                     verbose=True)