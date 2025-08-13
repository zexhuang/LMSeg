import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path.cwd() / "KPConv"))

import yaml
import argparse
import torch
import torch_geometric.transforms as T
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from KPConv.utils.config import Config
from KPConv.models.KPFCNN import KPFCNN
from KPConv.datasets.dataloader import get_dataloader
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import KPConvTrainer
from data.dataset import BBWPointDataset


class KPFCNNConfig(Config):    
    # https://github.com/XuyangBai/KPConv.pytorch#
    # model
    architecture = [
                    "simple",
                    "resnetb",
                    "resnetb_strided",
                    "resnetb",
                    "resnetb_strided",
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary'
                ]
    dropout = 0.5
    resume = None
    use_batch_norm = True
    batch_norm_momentum = 0.02
    # https://github.com/pytorch/examples/issues/289 pytorch bn momentum 0.02 == tensorflow bn momentum 0.98

    # kernel point convolution
    KP_influence = 'linear'
    KP_extent = 1.2
    convolution_mode = 'sum'
    
    def load(self, path):
        return
    
    def save(self, path):
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/kpconv_feature.yaml',
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
        
        kp_config = KPFCNNConfig()
        kp_config.dataset = cfg['dataset']
        kp_config.num_classes = cfg['out_channels']
        kp_config.in_features_dim = cfg['in_channels']
        kp_config.num_points = cfg['num_points']
        kp_config.train_batch_size = cfg['batch']
        kp_config.test_batch_size = cfg['batch']
        
        areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
        for area in areas:
            train_set = BBWPointDataset(root=args.root, split='train', test_area=area, config=kp_config)
            train_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
            
            val_set = BBWPointDataset(root=args.root, split='val', test_area=area, config=kp_config)
            val_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
            
            test_set = BBWPointDataset(root=args.root, split='test', test_area=area, config=kp_config)
            test_set.transform.transforms.append(T.FixedPoints(cfg['num_points']))
        
            train_loader = get_dataloader(train_set, 
                                          batch_size=cfg['batch'], 
                                          shuffle=True, 
                                          num_workers=cfg['workers'])   
            val_loader = get_dataloader(val_set, 
                                        batch_size=cfg['batch'], 
                                        shuffle=False, 
                                        num_workers=cfg['workers'])
            test_loader = get_dataloader(test_set, 
                                        batch_size=cfg['batch'], 
                                        shuffle=False, 
                                        num_workers=cfg['workers'])
            
            model = KPFCNN(kp_config)
            
            trainer = KPConvTrainer(cfg) 
            trainer.path = cfg['path'] + f'/{area}'
            trainer.fit(model, 
                        criterion=BCELogitsSmoothingLoss(),
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        optimizer=torch.optim.SGD(
                            model.parameters(),
                            lr=cfg['lr'],
                            momentum=0.9,
                            weight_decay=cfg['w_decay']
                        )
            )
            trainer.eval(model, 
                         test_loader, 
                         metric={'f1': BinaryF1Score(), 'mIoU': BinaryJaccardIndex()},
                         ckpt=f"epoch{cfg['epoch']}.pth",
                         verbose=True)
