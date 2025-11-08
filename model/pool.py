import torch
import torch.nn as nn

from torch_geometric.nn import fps, knn_graph
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.utils import (subgraph, 
                                   coalesce, 
                                   dropout_edge,
                                   remove_isolated_nodes,
                                   is_undirected, 
                                   to_undirected)


class FPSPooling(nn.Module):  
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, pool_ratio=0.5, ptr=None, batch=None):
        assert x.ndim == 2, 'node matrix need to be in shape of (N, D)'
        N = x.shape[0]
        
        pool_ratio = 1 / pool_ratio if pool_ratio > 1 else pool_ratio
        
        node_index = fps(x, batch, pool_ratio)
        N_pool = node_index.shape[0]
        
        edge_index_pool, _ = subgraph(node_index, edge_index, relabel_nodes=True, num_nodes=N)
        edge_index_pool = coalesce(edge_index_pool, num_nodes=N_pool)
        
        if not is_undirected(edge_index_pool, num_nodes=N_pool):
            edge_index_pool = to_undirected(edge_index_pool, num_nodes=N_pool)
            
        return edge_index_pool, node_index, ptr
    

class RandomPooling(nn.Module):  
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, pool_factor=1, ptr=None, batch=None):
        assert x.ndim == 2, 'node matrix need to be in shape of (N, D)'
        N = x.shape[0]
                
        ptr = torch.tensor([0, N], dtype=torch.long, device=x.device) if ptr is None else ptr
                
        node_index, ptr_pool = decimation_indices(ptr, pool_factor)
        N_pool = node_index.shape[0]
        
        edge_index_pool, _ = subgraph(node_index, edge_index, relabel_nodes=True, num_nodes=N)
        edge_index_pool = coalesce(edge_index_pool, num_nodes=N_pool)
                        
        if not is_undirected(edge_index_pool, num_nodes=N_pool):
            edge_index_pool = to_undirected(edge_index_pool, num_nodes=N_pool)
            
        return edge_index_pool, node_index, ptr_pool

    
class EdgeSimilarityPooling(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.cos_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, pos, edge_index, batch=None, **kwargs):
        assert x.ndim == 2, 'node matrix need to be in shape of (N, D)'
        assert pos.ndim == 2, 'position matrix need to be in shape of (N, 2) or (N, 3)'
        N = x.shape[0]        
        
        new_edge_index = knn_graph(x=pos, 
                                   k=3,
                                   batch=batch,
                                   loop=False, 
                                   flow='target_to_source',
                                   **kwargs)
        edge_index_pool = coalesce(torch.cat([new_edge_index, edge_index], dim=-1), num_nodes=N)
        
        src, dst = edge_index_pool
        edge_mask = self.cos_sim(x[src], x[dst]) >= self.threshold
 
        edge_index_pool = edge_index_pool[:, edge_mask]
        if not is_undirected(edge_index_pool, num_nodes=N):
            edge_index_pool = to_undirected(edge_index_pool, num_nodes=N)
            
        return edge_index_pool
    
      
class EdgeRandomPooling(nn.Module):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        assert x.ndim == 2, 'node matrix need to be in shape of (N, D)'
        N = x.shape[0]
        
        edge_index_pool, _ = dropout_edge(edge_index, p=self.dropout_rate, force_undirected=True, training=True)
        edge_index_pool = coalesce(edge_index_pool, num_nodes=N)
        
        edge_index_pool, _, node_index = remove_isolated_nodes(edge_index_pool, num_nodes=N)
        
        N_pool = node_index.nonzero().shape[0]
        edge_index_pool = coalesce(edge_index_pool, num_nodes=N_pool)
        
        assert is_undirected(edge_index_pool, num_nodes=N_pool)
        
        return edge_index_pool, node_index