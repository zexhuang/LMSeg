import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('/data/proj/LMSeg/KPConv')

import yaml
import argparse
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from KPConv.utils.config import Config
from KPConv.models.KPFCNN import KPFCNN
from KPConv.datasets.dataloader import get_dataloader
from model.utils.loss import BCELogitsSmoothingLoss

from train.trainer import KPConvTrainer
from data.dataset import BBWPointDataset


class KPFCNNConfig(Config):    
    # model
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    dropout = 0.5
    resume = None
    use_batch_norm = True
    batch_norm_momentum = 0.02
    # https://github.com/pytorch/examples/issues/289 pytorch bn momentum 0.02 == tensorflow bn momentum 0.98

    # kernel point convolution
    KP_influence = 'linear'
    KP_extent = 1.0
    convolution_mode = 'sum'
    
    def load(self, path):
        return
    
    def save(self, path):
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for model train val')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/bbw_kpconv_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/BudjBimWall',
                        help='path to dataset folder')
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)    
        
        kp_config = KPFCNNConfig()
        kp_config.dataset = cfg['dataset']
        kp_config.num_classes = cfg['out_channels']
        kp_config.in_features_dim = cfg['in_channels']
        kp_config.train_batch_size = cfg['batch']
        kp_config.test_batch_size = cfg['batch']
    
        train_set = BBWPointDataset(root=args.root, 
                                    split='train', 
                                    config=kp_config,
                                    load_feature=cfg['load_feature'])
        val_set = BBWPointDataset(root=args.root, 
                                  split='val', 
                                  config=kp_config,
                                  load_feature=cfg['load_feature'])
        test_set = BBWPointDataset(root=args.root, 
                                   split='test', 
                                   config=kp_config,
                                   load_feature=cfg['load_feature'])
    
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
        
        trainer.fit(model, 
                    criterion=BCELogitsSmoothingLoss(),
                    train_loader=train_loader, 
                    val_loader=val_loader)
        trainer.eval(model, 
                     test_loader, 
                     metric={'f1': BinaryF1Score(), 
                             'mIoU': BinaryJaccardIndex()},
                     ckpt=f"epoch{cfg['epoch']}",
                     verbose=True)
