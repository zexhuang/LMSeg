import sys
from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse
import torch

from torch_geometric.loader import DataLoader
from data.dataset import BudjBimLandscapeMeshDataset

from model.net import LMSegNet
from train.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for prediction')
    parser.add_argument('--cfg', type=str,  metavar='N',
                        default='cfg/bbw/bbw_lmseg_feature.yaml',
                        help='path to config file')
    parser.add_argument('--root', type=str,  metavar='N',
                        default='data/BudjBimLandscape',
                        help='path to dataset folder')
    parser.add_argument('--out_dir', type=str,  metavar='N',
                        default='data/BudjBimLandscape/processed/pred',
                        help='path to model output folder')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)    
        
        dataset = BudjBimLandscapeMeshDataset(root=args.root)        
        dataloader = DataLoader(dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=8)
        
        model = LMSegNet(cfg['in_channels'], cfg['out_channels'],
                         cfg['hid_channels'], 
                         cfg['num_convs'], 
                         cfg['pool_factors'], 
                         cfg['num_nbrs'],
                         cfg['num_block'],
                         cfg['alpha'], 
                         cfg['beta'],
                         cfg['load_feature'])
        
        trainer = Trainer(cfg=cfg) 
        trainer.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = trainer.load_weights(model, f"epoch{cfg['epoch']}")
        
        for idx, data in enumerate(tqdm(dataloader, desc="Predicting")):
            data = data.to(trainer.device)
            with torch.no_grad():
                out = model(data)
                pred = torch.sigmoid(out['y']).detach().cpu()
                torch.save(pred, out_dir / f"{data.name[0]}")