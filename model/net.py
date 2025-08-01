import torch
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.nn import knn_interpolate, knn
from torch_geometric.nn.pool.decimation import decimation_indices

from .layer import GAL, GAPL, ResMLP
from .pool  import RandomPooling, EdgeSimilarityPooling


class GAEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hid_channels: int, 
                 num_convs: int,  
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        
        self.num_convs = num_convs
        self.pool_factors = pool_factors
        self.num_nbrs = num_nbrs
            
        self.embedding = MLP([in_channels, hid_channels], plain_last=False)
        self.local_embedding = GAL(embedding_channels=hid_channels, num_block=num_block)
        self.res_mlp = nn.Sequential(*[ResMLP(hid_channels) for _ in range(num_block)])
        
        self.node_pool = RandomPooling()
        self.edge_pool = EdgeSimilarityPooling()
        
        self.down_convs_hier = nn.ModuleList()
        self.down_convs_local = nn.ModuleList()
        self.res_convs = nn.ModuleList()
        
        for _ in range(num_convs):                       
            self.down_convs_hier.append(GAL(embedding_channels=hid_channels, num_block=num_block))      
            self.down_convs_local.append(GAL(embedding_channels=hid_channels, num_block=num_block))      
            self.res_convs.append(nn.Sequential(*[ResMLP(hid_channels * 2) for _ in range(num_block)]))
            
            hid_channels *= 2
        
    def forward(self, pos, x, edge_index, batch, ptr):
        x = self.embedding(x)     
        x = self.local_embedding(pos, x, edge_index, batch=batch)
        x = self.res_mlp(x)
        
        pos_down, x_down, \
        batch_down, edge_index_down = [pos], [x], \
                                      [batch], [edge_index]
        
        for i in range(self.num_convs):  
            # Random node pooling
            edge_index_local, perm, ptr_pool = self.node_pool(x, 
                                                              edge_index, 
                                                              self.pool_factors[i],
                                                              ptr, 
                                                              batch)
            x_pool, pos_pool, batch_pool = x[perm], pos[perm], batch[perm] 
            # Hierarchical feature aggregation
            edge_index_hier = knn(x=pos, 
                                  y=pos_pool, 
                                  k=self.num_nbrs + 1,
                                  batch_x=batch, 
                                  batch_y=batch_pool)
            x_hier = self.down_convs_hier[i](pos, 
                                             x, 
                                             edge_index_hier, 
                                             pos_pool, 
                                             x_pool,
                                             batch=batch_pool)
            # Local feature aggregation
            edge_index_local = self.edge_pool(x=x_hier, 
                                              pos=pos_pool, 
                                              edge_index=edge_index_local, 
                                              batch=batch_pool)
            x_local = self.down_convs_local[i](pos_pool, 
                                               x_hier, 
                                               edge_index_local, 
                                               batch=batch_pool)
            x = self.res_convs[i](torch.cat([x_local, x_hier], dim=-1))
            ptr, pos, batch, edge_index = ptr_pool, pos_pool, batch_pool, edge_index_local
            
            pos_down.append(pos)
            x_down.append(x)
            batch_down.append(batch)
            edge_index_down.append(edge_index)
        return pos_down, x_down, batch_down, edge_index_down
    
    
class HGAPEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hid_channels: int, 
                 num_convs: int,  
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int, 
                 alpha: float, 
                 beta: float):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        
        self.num_convs = num_convs
        self.pool_factors = pool_factors
        self.num_nbrs = num_nbrs
            
        self.embedding = MLP([in_channels, hid_channels], plain_last=False)
        
        self.down_convs_hier = nn.ModuleList()
        self.res_convs = nn.ModuleList()
        
        for _ in range(num_convs):                       
            self.down_convs_hier.append(GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block))         
            self.res_convs.append(nn.Sequential(MLP([hid_channels, hid_channels * 2], plain_last=False), 
                                                *[ResMLP(hid_channels * 2) for _ in range(num_block)]))
            
            hid_channels *= 2
        
    def forward(self, pos, x, batch, ptr):
        x = self.embedding(x)      
        pos_down, x_down, batch_down = [pos], [x], [batch]
        
        for i in range(self.num_convs):  
            # Random node pooling
            perm, ptr_pool = decimation_indices(ptr, self.pool_factors[i])
            x_pool, pos_pool, batch_pool = x[perm], pos[perm], batch[perm] 
            # Hierarchical feature aggregation
            edge_index_hier = knn(x=pos, 
                                  y=pos_pool, 
                                  k=self.num_nbrs + 1,
                                  batch_x=batch, 
                                  batch_y=batch_pool)
            x_hier = self.down_convs_hier[i](pos, 
                                             x, 
                                             edge_index_hier, 
                                             pos_pool, 
                                             x_pool,
                                             batch=batch_pool)
            x = self.res_convs[i](x_hier)
            ptr, pos, batch = ptr_pool, pos_pool, batch_pool
            
            pos_down.append(pos)
            x_down.append(x)
            batch_down.append(batch)
        return pos_down, x_down, batch_down
    

class LGAPEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hid_channels: int, 
                 num_convs: int,  
                 pool_factors: list[float], 
                 num_block: int, 
                 alpha: float, 
                 beta: float):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        
        self.num_convs = num_convs
        self.pool_factors = pool_factors
            
        self.embedding = MLP([in_channels, hid_channels], plain_last=False)
        self.local_embedding = GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block)
        self.res_mlp = nn.Sequential(*[ResMLP(hid_channels) for _ in range(num_block)])
         
        self.node_pool = RandomPooling()
        self.edge_pool = EdgeSimilarityPooling()
        
        self.down_convs_local = nn.ModuleList()
        self.res_convs = nn.ModuleList()
        
        for _ in range(num_convs):                         
            self.down_convs_local.append(GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block))    
            self.res_convs.append(nn.Sequential(MLP([hid_channels, hid_channels * 2], plain_last=False), 
                                                *[ResMLP(hid_channels * 2) for _ in range(num_block)]))
            
            hid_channels *= 2
        
    def forward(self, pos, x, edge_index, batch, ptr):
        x = self.embedding(x)     
        x = self.local_embedding(pos, x, edge_index, batch=batch)   
        x = self.res_mlp(x)
        
        pos_down, x_down, \
        batch_down, edge_index_down = [pos], [x], \
                                      [batch], [edge_index]
        
        for i in range(self.num_convs):  
            # Random node pooling
            edge_index_local, perm, ptr_pool = self.node_pool(x, 
                                                              edge_index, 
                                                              self.pool_factors[i],
                                                              ptr, 
                                                              batch)
            x_pool, pos_pool, batch_pool = x[perm], pos[perm], batch[perm] 
            # Local feature aggregation
            edge_index_local = self.edge_pool(x=x_pool, 
                                              pos=pos_pool, 
                                              edge_index=edge_index_local, 
                                              batch=batch_pool)
            x_local = self.down_convs_local[i](pos_pool, 
                                               x_pool, 
                                               edge_index_local, 
                                               batch=batch_pool)
            x = self.res_convs[i](x_local)
            ptr, pos, batch, edge_index = ptr_pool, pos_pool, batch_pool, edge_index_local
            
            pos_down.append(pos)
            x_down.append(x)
            batch_down.append(batch)
            edge_index_down.append(edge_index)
        return pos_down, x_down, batch_down, edge_index_down
    
    
class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hid_channels: int, 
                 num_convs: int,  
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int, 
                 alpha: float, 
                 beta: float):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        
        self.num_convs = num_convs
        self.pool_factors = pool_factors
        self.num_nbrs = num_nbrs
            
        self.embedding = MLP([in_channels, hid_channels], plain_last=False)
        self.local_embedding = GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block)
        self.res_mlp = nn.Sequential(*[ResMLP(hid_channels) for _ in range(num_block)])
         
        self.node_pool = RandomPooling()
        self.edge_pool = EdgeSimilarityPooling()
        
        self.down_convs_hier = nn.ModuleList()
        self.down_convs_local = nn.ModuleList()
        self.res_convs = nn.ModuleList()
        
        for _ in range(num_convs):                       
            self.down_convs_hier.append(GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block))      
            self.down_convs_local.append(GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block))    
            self.res_convs.append(nn.Sequential(*[ResMLP(hid_channels * 2) for _ in range(num_block)]))
            
            hid_channels *= 2
        
    def forward(self, pos, x, edge_index, batch, ptr):
        x = self.embedding(x)     
        x = self.local_embedding(pos, x, edge_index, batch=batch)
        x = self.res_mlp(x)
        
        pos_down, x_down, \
        batch_down, edge_index_down = [pos], [x], \
                                      [batch], [edge_index]
        
        for i in range(self.num_convs):  
            # Random node pooling
            edge_index_local, perm, ptr_pool = self.node_pool(x, 
                                                              edge_index, 
                                                              self.pool_factors[i],
                                                              ptr, 
                                                              batch)
            x_pool, pos_pool, batch_pool = x[perm], pos[perm], batch[perm] 
            # Hierarchical feature aggregation
            edge_index_hier = knn(x=pos, 
                                  y=pos_pool, 
                                  k=self.num_nbrs + 1,
                                  batch_x=batch, 
                                  batch_y=batch_pool)
            x_hier = self.down_convs_hier[i](pos, 
                                             x, 
                                             edge_index_hier, 
                                             pos_pool, 
                                             x_pool,
                                             batch=batch_pool)
            # Local feature aggregation
            edge_index_local = self.edge_pool(x=x_hier, 
                                              pos=pos_pool, 
                                              edge_index=edge_index_local, 
                                              batch=batch_pool)
            x_local = self.down_convs_local[i](pos_pool, 
                                               x_hier, 
                                               edge_index_local, 
                                               batch=batch_pool)
            x = self.res_convs[i](torch.cat([x_local, x_hier], dim=-1))
            ptr, pos, batch, edge_index = ptr_pool, pos_pool, batch_pool, edge_index_local
            
            pos_down.append(pos)
            x_down.append(x)
            batch_down.append(batch)
            edge_index_down.append(edge_index)
        return pos_down, x_down, batch_down, edge_index_down
    
    
class Decoder(nn.Module):
    def __init__(self, 
                 hid_channels: int,
                 num_convs: int, 
                 num_nbrs: int = 3):
        super().__init__()       
        self.up_convs = nn.ModuleList() 
        
        up_channels = []
        for i in range(num_convs + 1):
            up_channels.append(hid_channels)
            hid_channels *= 2

        up_channels.reverse()
        
        channel = up_channels[0]
        for i in range(num_convs):
            self.up_convs.append(nn.Sequential(MLP([channel + up_channels[i+1], up_channels[i+1]], plain_last=False),
                                               ResMLP(up_channels[i+1])))
            channel = up_channels[i+1]
            
        self.num_convs = num_convs
        self.num_nbrs = num_nbrs
        
    def unpool(self, pos, pos_up, x, x_up, batch, batch_up):
        up = knn_interpolate(x, 
                             pos, pos_up, 
                             batch, batch_up, 
                             self.num_nbrs)
        x = torch.cat((x_up, up), dim=-1)
        return x
        
    def forward(self, pos, x, batch):
        x_i = x[0]
        for i in range(self.num_convs):
            x_i = self.unpool(pos[i], pos[i+1], 
                              x_i, x[i+1], 
                              batch[i], batch[i+1])
            x_i = self.up_convs[i](x_i)
        return x_i
    

class GANet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hid_channels: int, 
                 num_convs: int,
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int):
        super().__init__()
        # Parametric Encoder
        self.Enc = GAEncoder(in_channels, hid_channels, 
                             num_convs, pool_factors,
                             num_nbrs, num_block)    
        # Parametric Decoder
        self.Dec = Decoder(hid_channels, num_convs)   
        # MLP Classifier
        self.mlp = MLP([hid_channels, 128, 128, out_channels], dropout=0.5)
        
        self.init_weights()
        
    def init_weights(self):
        for param in self.Enc.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.Dec.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.mlp.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        
    def forward(self, data):
        pos, x, edge_index, batch, ptr = data.pos, data.x, \
                                         data.edge_index, data.batch, data.ptr
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)

        pos_down, x_down, batch_down, _ = self.Enc(pos, x, edge_index, batch, ptr) 
        f = self.Dec(pos_down[::-1], x_down[::-1], batch_down[::-1])
        y = self.mlp(f)
        return {'y':y} 
    

class HGAPNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hid_channels: int, 
                 num_convs: int,
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int,
                 alpha: float, 
                 beta: float):
        super().__init__()
        # Parametric Encoder
        self.Enc = HGAPEncoder(in_channels, hid_channels, 
                               num_convs, pool_factors,
                               num_nbrs, num_block,
                               alpha, beta)    
        # Parametric Decoder
        self.Dec = Decoder(hid_channels, num_convs)   
        # MLP Classifier
        self.mlp = MLP([hid_channels, 128, 128, out_channels], dropout=0.5)
        
        self.init_weights()
        
    def init_weights(self):
        for param in self.Enc.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.Dec.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.mlp.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        
    def forward(self, data):
        pos, x, batch, ptr = data.pos, data.x, data.batch, data.ptr
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)

        pos_down, x_down, batch_down = self.Enc(pos, x, batch, ptr) 
        f = self.Dec(pos_down[::-1], x_down[::-1], batch_down[::-1])
        y = self.mlp(f)
        return {'y':y} 
    
    
class LGAPNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hid_channels: int, 
                 num_convs: int,
                 pool_factors: list[float], 
                 num_block: int,
                 alpha: float, 
                 beta: float):
        super().__init__()
        # Parametric Encoder
        self.Enc = LGAPEncoder(in_channels, hid_channels, 
                               num_convs, pool_factors,
                               num_block,
                               alpha, beta)    
        # Parametric Decoder
        self.Dec = Decoder(hid_channels, num_convs)   
        # MLP Classifier
        self.mlp = MLP([hid_channels, 128, 128, out_channels], dropout=0.5)
        
        self.init_weights()
        
    def init_weights(self):
        for param in self.Enc.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.Dec.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            
        for param in self.mlp.parameters():
            if param.ndim > 1: 
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        
    def forward(self, data):
        pos, x, edge_index, batch, ptr = data.pos, data.x, \
                                         data.edge_index, data.batch, data.ptr
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)

        pos_down, x_down, batch_down, _ = self.Enc(pos, x, edge_index, batch, ptr) 
        f = self.Dec(pos_down[::-1], x_down[::-1], batch_down[::-1])
        y = self.mlp(f)
        return {'y':y} 
        

class LMSegNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hid_channels: int, 
                 num_convs: int,
                 pool_factors: list[float], 
                 num_nbrs: int,
                 num_block: int,
                 alpha: float, 
                 beta: float):
        super().__init__()
        # Parametric Encoder
        self.Enc = Encoder(in_channels, hid_channels, 
                           num_convs, pool_factors,
                           num_nbrs, num_block,
                           alpha, beta)    
        # Parametric Decoder
        self.Dec = Decoder(hid_channels, num_convs)   
        # MLP Classifier
        self.mlp = MLP([hid_channels, 128, 128, out_channels], dropout=0.5)
        
        self.init_weights()
        
    def init_weights(self):
        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.Enc.apply(weights_init)
        self.Dec.apply(weights_init)
        self.mlp.apply(weights_init)

    def forward(self, data):
        pos, x, edge_index, batch, ptr = data.pos, data.x, \
                                         data.edge_index, data.batch, data.ptr
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)

        pos_down, x_down, batch_down, _ = self.Enc(pos, x, edge_index, batch, ptr) 
        f = self.Dec(pos_down[::-1], x_down[::-1], batch_down[::-1])
        y = self.mlp(f)
        return {'y':y} 