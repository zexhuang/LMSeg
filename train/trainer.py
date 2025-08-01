import yaml
import logging
import random
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchinfo import summary
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, cfg: Union[str, Path, dict]):
        if isinstance(cfg, (str, Path)):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)
        elif isinstance(cfg, dict):
            self.cfg = cfg
        else:
            raise ValueError("Config must be a string, Path, or dictionary.")

        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.num_cls = self.cfg['out_channels']
        self.device = self.cfg['device']

        self.set_seed()

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        ckpt = Path(ckpt) if isinstance(ckpt, str) else ckpt
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])
        model.eval().to(self.device)
        return model

    def _save_ckpt(self, model, ckpt_name):
        ckpt_dir = Path(self.path) / 'ckpt'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'params': model.state_dict()}, ckpt_dir / f'{ckpt_name}.pth')
        logging.info(f'Model checkpoint saved: {ckpt_name}.pth')

    def _load_ckpt(self, ckpt_name, device):
        ckpt_path = Path(self.path) / 'ckpt' / ckpt_name
        return torch.load(ckpt_path, map_location=device, weights_only=True)


class Trainer(BaseTrainer):
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],
            criterion: Optional[Callable] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            ckpt: Union[str, Path, None] = None,
            save_period: int = 10):

        summary(model, depth=3)
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        criterion = criterion or torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch, eta_min=1e-5)

        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)

        for ep in tqdm(range(1, self.epoch + 1)):
            train_loss = self._train_one_epoch(model, optimizer, criterion, train_loader, self.device)
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            lr_scheduler.step()

            if ep % save_period == 0:
                if val_loader:
                    val_loss = self._evaluate(model, criterion, val_loader, self.device)
                    self.writer.add_scalar('Loss/val', val_loss, ep)
                    early_stopping(val_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping at epoch {ep}.")
                        break
                else:
                    self._save_ckpt(model, ckpt_name=f'epoch{ep}')

    def _train_one_epoch(self, model, optimizer, criterion, dataloader, device):
        model.train().to(device)
        total_loss = 0.0

        for data in dataloader:
            data = data.to(self.device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out['y'], data['y'])
            loss.backward()
            optimizer.step()
            total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def _evaluate(self, model, criterion, dataloader, device):
        model.eval().to(device)
        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                data = data.to('cpu')
                out = model(data)
                loss = criterion(out['y'], data['y'])
                total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing],
             dataloader: DataLoader,
             metric: Optional[Dict[str, Metric]] = None,
             ckpt: Union[str, Path, None] = None,
             verbose: bool = False):

        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        metric = metric or {
            'OA': BinaryAccuracy(),
            'mIoU': BinaryJaccardIndex()
        }

        model.eval().to(self.device)
        for cm in metric.values():
            cm.to(self.device)

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                out = model(data)
                for name, cm in metric.items():
                    cm.update(out['y'], data['y'].long())

        if verbose:
            for name, cm in metric.items():
                value = cm.compute().cpu()
                if value.numel() == 1:
                    print(f"{name}: {value.item():.4f}")
                else:
                    formatted_values = ", ".join(f"{v:.4f}" for v in value.numpy())
                    print(f"{name}: [{formatted_values}]")

        return metric
    
    
class KPConvTrainer(BaseTrainer):
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],
            criterion: Optional[Callable] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            ckpt: Union[str, Path, None] = None,
            save_period: int = 10):

        summary(model, depth=3)
        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        criterion = criterion or torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch, eta_min=1e-5)

        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
        early_stopping = EarlyStopping(path=self.path, patience=self.patience)

        for ep in tqdm(range(1, self.epoch + 1)):
            train_loss = self._train_one_epoch(model, optimizer, criterion, train_loader, self.device)
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            lr_scheduler.step()

            if ep % save_period == 0:
                if val_loader:
                    val_loss = self._evaluate(model, criterion, val_loader, self.device)
                    self.writer.add_scalar('Loss/val', val_loss, ep)
                    early_stopping(val_loss, model, optimizer, ep, lr_scheduler.get_last_lr())
                    if early_stopping.early_stop:
                        logging.info(f"Early stopping at epoch {ep}.")
                        break
                else:
                    self._save_ckpt(model, ckpt_name=f'epoch{ep}')

    def _train_one_epoch(self, model, optimizer, criterion, dataloader, device):
        model.train().to(device)
        total_loss = 0.0

        for data in dataloader:
            data = self._to_device(data, device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data['labels'])
            loss.backward()
            optimizer.step()
            total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def _evaluate(self, model, criterion, dataloader, device):
        model.eval().to(device)
        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                data = self._to_device(data, device)
                out = model(data)
                loss = criterion(out, data['labels'])
                total_loss += len(data) * loss.item()

        return total_loss / len(dataloader.dataset)

    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing],
             dataloader: DataLoader,
             metric: Optional[Dict[str, Metric]] = None,
             ckpt: Union[str, Path, None] = None,
             verbose: bool = False):

        if ckpt:
            model.load_state_dict(self._load_ckpt(ckpt, self.device)['params'])

        metric = metric or {
            'OA': BinaryAccuracy(),
            'mIoU': BinaryJaccardIndex()
        }

        model.eval().to(self.device)
        for cm in metric.values():
            cm.to(self.device)

        with torch.no_grad():
            for data in dataloader:
                data = self._to_device(data)
                out = model(data)
                for name, cm in metric.items():
                    cm.update(out, data['labels'].long())

        if verbose:
            for name, cm in metric.items():
                value = cm.compute().cpu()
                if value.numel() == 1:
                    print(f"{name}: {value.item():.4f}")
                else:
                    formatted_values = ", ".join(f"{v:.4f}" for v in value.numpy())
                    print(f"{name}: [{formatted_values}]")

        return metric

    def _to_device(self, data, device=None):
        device = device or self.device
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = [item.to(device) for item in v]
            else:
                data[k] = v.to(device)
        return data


# Code adapoted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation monitor doesn't improve after a given patience."""
    def __init__(self, path, best_score=None, patience=10, delta=0.0, verbose=False, trace_func=print):
        """
        Args:
            path (str): Path for the checkpoint to be saved to.
            best_score (flaot or none): Value of metric of the best model.
                            Default: None
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            trace_func (function): trace print function.
                            Default: print            
        """
        self.path = Path(path)
        self.best_score = best_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model, optimizer, epoch, last_lr, cm=None):
        score = loss 
        
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'params': model_state,
            'optimizer': optimizer_state,
            'lr': last_lr[0],
            'cm': cm
        }
    
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(checkpoint, score)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\n Validation loss does not improve ({self.best_score} --> {score}). \n EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(checkpoint, score)
            self.counter = 0

    def save_checkpoint(self, checkpoint, loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\n Validation loss decrease ({self.best_score:.6f} --> {loss:.6f}). \n Saving model ...')
            
        checkpoint_path = self.path / 'ckpt'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path.joinpath('best_val_epoch.pth'))
        
        self.best_score = loss