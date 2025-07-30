from typing import Callable, Optional, Union, Dict, Any

import torch
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import scatter
from torch_geometric.nn.resolver import (activation_resolver,
                                         normalization_resolver)


class ResMLP(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 bias: bool = True,
                 norm: Union[str, Callable, None] = "batch_norm",
                 norm_kwargs: Optional[Dict[str, Any]] = None,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.conv1 = MLP([in_channels, in_channels], 
                         norm=norm, norm_kwargs=norm_kwargs,
                         act=act, act_kwargs=act_kwargs,
                         plain_last=False,
                         bias=bias)
        self.conv2 = MLP([in_channels, in_channels], 
                         norm=None, norm_kwargs=norm_kwargs,
                         act=None, act_kwargs=act_kwargs,
                         plain_last=True,
                         bias=bias)
        self.norm = normalization_resolver(norm, in_channels, **(norm_kwargs or {}))
        self.act = activation_resolver(act, **(act_kwargs or {}))
        
    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))) + x)
    

class PosEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 alpha: float,
                 beta: float):
        super().__init__()        
        feat_dim = out_channels // in_channels
        self.encoding = PositionalEncoding(feat_dim, alpha, beta)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, pos):          
        x = self.encoding(pos)
        x = x.view(pos.shape[0], self.out_channels)
        return x


class GAL(nn.Module):
    def __init__(self, 
                 embedding_channels: int, 
                 num_block: int):
        super().__init__()
        self.embedding_channels = embedding_channels        
        
        self.affine_w = nn.Parameter(torch.ones(embedding_channels + 3))
        self.affine_b = nn.Parameter(torch.zeros(embedding_channels + 3))

        self.shared_lin = MLP([(embedding_channels + 3) * 2, embedding_channels], plain_last=False)
        self.shared_res_mlp = nn.Sequential(*[ResMLP(embedding_channels) for _ in range(num_block)])
        
    def forward(self, pos, x, edge_index, pos_c=None, x_c=None, batch=None):
        pos_c = pos.clone() if pos_c is None else pos_c
        x_c = x.clone() if x_c is None else x_c
        
        assert pos.ndim == 2 and pos.shape[-1] == 3
        assert pos_c.ndim == 2 and pos_c.shape[-1] == 3
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2
        
        idx_i, idx_j = edge_index[0], edge_index[1]
        pos_i, pos_j = pos_c[idx_i], pos[idx_j]

        x_i, x_j = x_c[idx_i], x[idx_j]
        x_i, x_j = torch.cat([x_i, pos_i], dim=-1), torch.cat([x_j, pos_j], dim=-1)

        # Normalize Node Feature
        std_x = torch.std(x_j - x_i)
        x_rel = (x_j - x_i) / (std_x + 1e-5)
        x_rel = self.affine_w * x_rel + self.affine_b
        
        # Feature Expansion
        x_w = torch.cat([x_i, x_rel], dim=-1)
        x_w = self.shared_lin(x_w)
        
        # Shared MLP
        out_x = self.shared_res_mlp(x_w)
        
        # Aggregations
        dim_size = None if batch is None else batch.shape[0] 
        msg = scatter(out_x, idx_i, dim=0, dim_size=dim_size, reduce='max') \
            + scatter(out_x, idx_i, dim=0, dim_size=dim_size, reduce='mean') 
        return msg


class GAPL(nn.Module):
    def __init__(self, 
                 embedding_channels: int, 
                 alpha: float, 
                 beta: float,
                 num_block: int,
                 t_max: float = 1.0,
                 t_avg: float = 0.0):
        super().__init__()
        self.embedding_channels = embedding_channels        
        self.alpha = alpha
        self.beta = beta    
        
        self.affine_w = nn.Parameter(torch.ones(embedding_channels + 3))
        self.affine_b = nn.Parameter(torch.zeros(embedding_channels + 3))
    
        self.shared_lin = MLP([(embedding_channels + 3) * 2, embedding_channels], plain_last=False)
        self.embedding = PosEmbedding(3, embedding_channels, alpha, beta)
        self.shared_res_mlp = nn.Sequential(*[ResMLP(embedding_channels) for _ in range(num_block)])
                        
        self.gen_aggr_max = SoftmaxAggregation(t=t_max, learn=True, channels=embedding_channels)  
        self.gen_aggr_avg = SoftmaxAggregation(t=t_avg, learn=True, channels=embedding_channels)       
        
    def forward(self, pos, x, edge_index, pos_c=None, x_c=None, batch=None):
        pos_c = pos.clone() if pos_c is None else pos_c
        x_c = x.clone() if x_c is None else x_c
        
        assert pos.ndim == 2 and pos.shape[-1] == 3
        assert pos_c.ndim == 2 and pos_c.shape[-1] == 3
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2
        
        idx_i, idx_j = edge_index[0], edge_index[1]
        pos_i, pos_j = pos_c[idx_i], pos[idx_j]
        
        x_i, x_j = x_c[idx_i], x[idx_j]
        x_i, x_j = torch.cat([x_i, pos_i], dim=-1), torch.cat([x_j, pos_j], dim=-1)
                
        # Normalize Node Feature
        std_x = torch.std(x_j - x_i)
        x_rel = (x_j - x_i) / (std_x + 1e-5)
        x_rel = self.affine_w * x_rel + self.affine_b
        
        # Feature Expansion
        x_w = torch.cat([x_i, x_rel], dim=-1)
        x_w = self.shared_lin(x_w)
        
        # Normalize Node Pos 
        std_pos = torch.std(pos_j - pos_i)
        pos_rel = (pos_j - pos_i) / (std_pos + 1e-5)
        
        # Geometry Extraction        
        pos_embedding = self.embedding(pos_rel)
        x_w = pos_embedding * (x_w + pos_embedding)
        
        # Shared Residual MLP
        out_x = self.shared_res_mlp(x_w)
        
        # Aggregations
        dim_size = None if batch is None else batch.shape[0] 
        msg = self.gen_aggr_max(x=out_x, index=idx_i, dim_size=dim_size, dim=0) \
            + self.gen_aggr_avg(x=out_x, index=idx_i, dim_size=dim_size, dim=0)
        return msg