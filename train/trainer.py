import yaml
import logging
import torch

from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, Callable, Dict

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchinfo import summary


class Trainer:
    def __init__(self, cfg:Union[str, Path, dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, dict):
            self.cfg = cfg
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.num_cls = self.cfg['out_channels']
        self.device = self.cfg['device']
            
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],  
            criterion: Optional[Callable]=None,  
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=10):
        summary(model, depth=100)
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) if ckpt else model  
        criterion = torch.nn.CrossEntropyLoss() if criterion == None else criterion
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=self.lr, 
                                      weight_decay=self.w_decay) 
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                  T_max=self.epoch,
                                                                  eta_min=1e-5)
        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
         
        for ep in tqdm(range(1, self.epoch+1)):
            self.device = self.cfg['device']
            t_ls = self._fit_impl(model, optimizer, criterion, train_loader, self.device)
            self.writer.add_scalar('Loss/train', t_ls, ep)
            torch.cuda.empty_cache()
                
            # Adjust learning rate 
            self.writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            lr_scheduler.step()
                
            if ep % save_period == 0: # save model at every n epoch
                if val_loader:
                    # self.device = 'cpu'
                    v_ls = self._val_impl(model, criterion, val_loader, self.device) 
                    self.writer.add_scalar('Loss/val', v_ls, ep)
                    torch.cuda.empty_cache()
                    
                self._save_ckpt(model, ckpt_name=f'epoch{ep}')
            
    def _fit_impl(self, model, optimizer, criterion, dataloader, device):
        model.to(device)  
        model.train()
        ls = 0.0
        for _, data in enumerate(dataloader):
            data.to(device)
            optimizer.zero_grad()         # Clear gradients
            out = model(data) 
            loss = criterion(out['y'], data['y'])   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            # Loss dim reduction="mean"
            ls += len(data) * loss.detach().clone()  
        return ls / len(dataloader.dataset)
    
    def _val_impl(self, model, criterion, dataloader, device):
        model.to(device)  
        model.eval()
        
        ls = 0.0
        
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                data.to(device)
                out = model(data) 
                loss = criterion(out['y'], data['y'])   
                ls += len(data) * loss.detach().clone()
        return ls / len(dataloader.dataset)

    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing], 
             dataloader: Optional[DataLoader]=None, 
             metric: Optional[Dict[str, Metric]]=None,
             ckpt: Union[str, Path, None]=None,
             verbose: bool=False):
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) if ckpt else model  
        
        if metric is None:
            metric = {'OA': BinaryAccuracy(), 
                      'mIoU': BinaryJaccardIndex()}

        metric = self._eval_impl(model, metric, dataloader)
        
        if verbose:
            for name, cm in metric.items(): print(f"{name}: {cm.compute().cpu().numpy().tolist()}")    
        return metric
    
    def _eval_impl(self, model, metric, dataloader):
        model.to(self.device)
        model.eval()
        
        for name, cm in metric.items(): 
            cm.to(self.device)
        
        with torch.no_grad():
            for data in dataloader:
                data.to(self.device)
                out = model(data) 
                for name, cm in metric.items():
                    cm.update(out['y'], data['y'].long())
        return metric  
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) 
        model.eval()
        model.to(self.device)
        return model
    
    def _save_ckpt(self, model, ckpt_name):
        path = Path(self.path) / 'ckpt'
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({'params':model.state_dict()}, path.joinpath(ckpt_name))
        logging.info('model ckpt is saved.')
    
    def _load_ckpt(self, ckpt_name, device):
        path = Path(self.path) / 'ckpt'
        return torch.load(path.joinpath(ckpt_name), map_location=device, weights_only=True) # {'params': Tensor}
    
    
class KPConvTrainer:
    def __init__(self, cfg:Union[str, Path, dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f:
                self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, dict):
            self.cfg = cfg
            
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.num_cls = self.cfg['out_channels']
        self.device = self.cfg['device']
            
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],  
            criterion: Optional[Callable]=None,  
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=10):
        summary(model, depth=100)
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) if ckpt else model  
        criterion = torch.nn.CrossEntropyLoss() if criterion == None else criterion
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=self.lr, 
                                      weight_decay=self.w_decay) 
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                  T_max=self.epoch,
                                                                  eta_min=1e-5)
        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
         
        for ep in tqdm(range(1, self.epoch+1)):
            t_ls = self._fit_impl(model, optimizer, criterion, train_loader, self.device)
            self.writer.add_scalar('Loss/train', t_ls, ep)
            torch.cuda.empty_cache()
                
            # Adjust learning rate 
            self.writer.add_scalar('LRate/train', lr_scheduler.get_last_lr()[0], ep)
            lr_scheduler.step()
                
            if ep % save_period == 0: # save model at every n epoch
                if val_loader:
                    v_ls = self._val_impl(model, criterion, val_loader, self.device) 
                    self.writer.add_scalar('Loss/val', v_ls, ep)
                    torch.cuda.empty_cache()
                    
                self._save_ckpt(model, ckpt_name=f'epoch{ep}')
            
    def _fit_impl(self, model, optimizer, criterion, dataloader, device):
        model.to(device)  
        model.train()
        ls = 0.0
        for _, data in enumerate(dataloader):
            for k, v in data.items():  # load inputs to device.
                if type(v) == list:
                    data[k] = [item.to(self.device) for item in v]
                else:
                    data[k] = v.to(self.device)
            optimizer.zero_grad()         # Clear gradients
            out = model(data) 
            loss = criterion(out, data['labels'])   # Compute gradients
            loss.backward()               # Backward pass 
            optimizer.step()              # Update model parameters                                                       
            # Loss dim reduction="mean"
            ls += len(data) * loss.detach().clone()  
        return ls / len(dataloader.dataset)
    
    def _val_impl(self, model, criterion, dataloader, device):
        model.to(device)  
        model.eval()
        
        ls = 0.0
        
        with torch.no_grad():
            for _, data in enumerate(dataloader):       
                for k, v in data.items():  # load inputs to device.
                    if type(v) == list:
                        data[k] = [item.to(self.device) for item in v]
                    else:
                        data[k] = v.to(self.device)
                out = model(data) 
                loss = criterion(out, data['labels'])   
                ls += len(data) * loss.detach().clone()
        return ls / len(dataloader.dataset)

    def eval(self, 
             model: Union[torch.nn.Module, MessagePassing], 
             dataloader: Optional[DataLoader]=None, 
             metric: Optional[Dict[str, Metric]]=None,
             ckpt: Union[str, Path, None]=None,
             verbose: bool=False):
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) if ckpt else model  
        
        if metric is None:
            metric = {'OA': BinaryAccuracy(), 
                      'mIoU': BinaryJaccardIndex()}

        metric = self._eval_impl(model, metric, dataloader)
        
        if verbose:
            for name, cm in metric.items(): print(f"{name}: {cm.compute().cpu().numpy().tolist()}")    
        return metric
    
    def _eval_impl(self, model, metric, dataloader):
        model.to(self.device)
        model.eval()
        
        for name, cm in metric.items(): 
            cm.to(self.device)
        
        with torch.no_grad():
            for data in dataloader:
                for k, v in data.items():  # load inputs to device.
                    if type(v) == list:
                        data[k] = [item.to(self.device) for item in v]
                    else:
                        data[k] = v.to(self.device)
                out = model(data) 
                for name, cm in metric.items():
                    cm.update(out, data['labels'].long())
        return metric  
    
    def load_weights(self, model: torch.nn.Module, ckpt: Union[str, Path]):
        if isinstance(ckpt, str): ckpt = Path(ckpt)
        model.load_state_dict(self._load_ckpt(ckpt, self.device)['params']) 
        model.eval()
        model.to(self.device)
        return model
    
    def _save_ckpt(self, model, ckpt_name):
        path = Path(self.path) / 'ckpt'
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({'params':model.state_dict()}, path.joinpath(ckpt_name))
        logging.info('model ckpt is saved.')
    
    def _load_ckpt(self, ckpt_name, device):
        path = Path(self.path) / 'ckpt'
        return torch.load(path.joinpath(ckpt_name), map_location=device, weights_only=True) # {'params': Tensor}