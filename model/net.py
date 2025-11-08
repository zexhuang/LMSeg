import torch
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.nn import knn_interpolate, knn
from torch_geometric.nn.pool.decimation import decimation_indices

from .layer import GAL, GAPL, ResMLP, MeshFeatureEncoder
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
            self.res_convs.append(
                nn.Sequential(*[ResMLP(hid_channels * 2) for _ in range(num_block)])
            )
            hid_channels *= 2

    def forward(self, pos, x, edge_index, batch, ptr):
        x = self.embedding(x)
        x = self.local_embedding(pos, x, edge_index, batch=batch)
        x = self.res_mlp(x)

        pos_down, x_down = [pos], [x]
        batch_down, edge_index_down = [batch], [edge_index]

        for i in range(self.num_convs):
            edge_index_local, perm, ptr_pool = self.node_pool(
                x, edge_index, self.pool_factors[i], ptr, batch
            )
            pos_pool, x_pool, batch_pool = pos[perm], x[perm], batch[perm]

            edge_index_hier = knn(
                x=pos, y=pos_pool, k=self.num_nbrs,
                batch_x=batch, batch_y=batch_pool
            )

            x_hier = self.down_convs_hier[i](
                pos, x, edge_index_hier,
                pos_pool, x_pool,
                batch=batch_pool
            )

            edge_index_local = self.edge_pool(
                x=x_hier, pos=pos_pool, edge_index=edge_index_local, batch=batch_pool
            )

            x_local = self.down_convs_local[i](
                pos_pool, x_hier, edge_index_local, batch=batch_pool
            )

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

        self.num_convs = num_convs
        self.pool_factors = pool_factors
        self.num_nbrs = num_nbrs

        self.embedding = MLP([in_channels, hid_channels], plain_last=False)

        self.down_convs_hier = nn.ModuleList()
        self.res_convs = nn.ModuleList()

        for _ in range(num_convs):
            self.down_convs_hier.append(
                GAPL(embedding_channels=hid_channels, alpha=alpha, beta=beta, num_block=num_block)
            )
            self.res_convs.append(
                nn.Sequential(
                    MLP([hid_channels, hid_channels * 2], plain_last=False),
                    *[ResMLP(hid_channels * 2) for _ in range(num_block)]
                )
            )
            hid_channels *= 2

    def forward(self, pos, x, batch, ptr):
        x = self.embedding(x)

        pos_down, x_down, batch_down = [pos], [x], [batch]

        for i in range(self.num_convs):
            perm, ptr_pool = decimation_indices(ptr, self.pool_factors[i])
            pos_pool, x_pool, batch_pool = pos[perm], x[perm], batch[perm]

            edge_index_hier = knn(
                x=pos, y=pos_pool, k=self.num_nbrs,
                batch_x=batch, batch_y=batch_pool
            )

            x_hier = self.down_convs_hier[i](
                pos, x, edge_index_hier,
                pos_pool, x_pool,
                batch=batch_pool
            )

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

        self.num_convs = num_convs
        self.pool_factors = pool_factors

        self.embedding = MLP([in_channels, hid_channels], plain_last=False)

        self.local_embedding = GAPL(
            embedding_channels=hid_channels, 
            alpha=alpha, beta=beta, 
            num_block=num_block
        )
        self.res_mlp = nn.Sequential(*[ResMLP(hid_channels) for _ in range(num_block)])

        self.node_pool = RandomPooling()
        self.edge_pool = EdgeSimilarityPooling()

        self.down_convs_local = nn.ModuleList()
        self.res_convs = nn.ModuleList()

        for _ in range(num_convs):
            self.down_convs_local.append(
                GAPL(hid_channels, alpha=alpha, beta=beta, num_block=num_block)
            )
            self.res_convs.append(
                nn.Sequential(
                    MLP([hid_channels, hid_channels * 2], plain_last=False),
                    *[ResMLP(hid_channels * 2) for _ in range(num_block)]
                )
            )
            hid_channels *= 2

    def forward(self, pos, x, edge_index, batch, ptr):
        x = self.embedding(x)
        x = self.local_embedding(pos, x, edge_index, batch=batch)
        x = self.res_mlp(x)

        pos_down, x_down = [pos], [x]
        batch_down, edge_index_down = [batch], [edge_index]

        for i in range(self.num_convs):
            edge_index_local, perm, ptr_pool = self.node_pool(
                x, edge_index, self.pool_factors[i], ptr, batch
            )
            pos_pool, x_pool, batch_pool = pos[perm], x[perm], batch[perm]

            edge_index_local = self.edge_pool(
                x=x_pool, pos=pos_pool, edge_index=edge_index_local, batch=batch_pool
            )

            x_local = self.down_convs_local[i](
                pos_pool, x_pool, edge_index_local, batch=batch_pool
            )

            x = self.res_convs[i](x_local)

            pos, x, batch, edge_index, ptr = pos_pool, x, batch_pool, edge_index_local, ptr_pool

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

        self.num_convs = num_convs
        self.pool_factors = pool_factors
        self.num_nbrs = num_nbrs

        # Initial MLP to project input features
        self.embedding = MLP([in_channels, hid_channels], plain_last=False)

        # Initial local geometric encoder
        self.local_embedding = GAPL(
            embedding_channels=hid_channels, 
            alpha=alpha, beta=beta, 
            num_block=num_block
        )
        self.res_mlp = nn.Sequential(*[ResMLP(hid_channels) for _ in range(num_block)])

        # Pooling modules
        self.node_pool = RandomPooling()
        self.edge_pool = EdgeSimilarityPooling()

        # Hierarchical and local Message passing modules
        self.down_convs_hier = nn.ModuleList()
        self.down_convs_local = nn.ModuleList()
        self.res_convs = nn.ModuleList()

        for _ in range(num_convs):
            self.down_convs_hier.append(
                GAPL(hid_channels, alpha=alpha, beta=beta, num_block=num_block)
            )
            self.down_convs_local.append(
                GAPL(hid_channels, alpha=alpha, beta=beta, num_block=num_block)
            )
            self.res_convs.append(
                nn.Sequential(*[ResMLP(hid_channels * 2) for _ in range(num_block)])
            )
            hid_channels *= 2  # Double feature channels per level

    def forward(self, pos, x, edge_index, batch, ptr):
        # Initial embedding
        x = self.embedding(x)
        x = self.local_embedding(pos, x, edge_index, batch=batch)
        x = self.res_mlp(x)

        # Multi-scale outputs
        pos_down, x_down = [pos], [x]
        batch_down, edge_index_down = [batch], [edge_index]

        for i in range(self.num_convs):
            # 1. Random Node Pooling (subsample graph)
            edge_index_local, perm, ptr_pool = self.node_pool(
                x, edge_index, self.pool_factors[i], ptr, batch
            )
            pos_pool, x_pool, batch_pool = pos[perm], x[perm], batch[perm]

            # 2. Hierarchical Feature Aggregation
            edge_index_hier = knn(
                x=pos, y=pos_pool, k=self.num_nbrs, 
                batch_x=batch, batch_y=batch_pool
            )
            x_hier = self.down_convs_hier[i](
                pos, x, edge_index_hier, 
                pos_pool, x_pool, batch=batch_pool
            )

            # 3. Local Feature Aggregation
            edge_index_local = self.edge_pool(
                x=x_hier, pos=pos_pool, 
                edge_index=edge_index_local, batch=batch_pool
            )
            x_local = self.down_convs_local[i](
                pos_pool, x_hier, edge_index_local, batch=batch_pool
            )

            # 4. Residual Update from Combined Features
            x = self.res_convs[i](torch.cat([x_local, x_hier], dim=-1))

            # Update graph structures for next layer
            pos, x, batch, edge_index, ptr = pos_pool, x, batch_pool, edge_index_local, ptr_pool

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
        
        self.num_convs = num_convs
        self.num_nbrs = num_nbrs
        self.up_convs = nn.ModuleList()

        # Compute feature dimensions for each decoder level
        up_channels = [hid_channels * (2 ** i) for i in range(num_convs + 1)]
        up_channels.reverse()  # Top-down from deepest to shallowest

        in_ch = up_channels[0]
        for i in range(num_convs):
            out_ch = up_channels[i + 1]
            self.up_convs.append(
                nn.Sequential(
                    MLP([in_ch + out_ch, out_ch], plain_last=False),
                    ResMLP(out_ch)
                )
            )
            in_ch = out_ch

    def unpool(self, pos, pos_up, x, x_up, batch, batch_up):
        up_feat = knn_interpolate(x, pos, pos_up, batch, batch_up, k=self.num_nbrs)
        return torch.cat([x_up, up_feat], dim=-1)

    def forward(self, pos, x, batch):
        x_i = x[0]
        for i in range(self.num_convs):
            x_i = self.unpool(pos[i], pos[i+1], x_i, x[i+1], batch[i], batch[i+1])
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
        # Mesh Feature Encoder
        self.ColorEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        self.NormalEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        # Parametric Encoder
        self.Enc = GAEncoder(hid_channels + 3, hid_channels, 
                             num_convs, pool_factors,
                             num_nbrs, num_block)    
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
        self.ColorEncoder.apply(weights_init)
        self.NormalEncoder.apply(weights_init)
        
    def forward(self, data):
        pos, edge_index, batch, ptr = data.pos, data.edge_index, data.batch, data.ptr
        
        x_rgb = self.ColorEncoder(data.rgb.view(-1, 4, 3))
        x_normals = self.NormalEncoder(data.normals.view(-1, 4, 3))
        x = torch.cat([x_rgb, x_normals, pos], dim=-1)

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
        # Mesh Feature Encoder
        self.ColorEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        self.NormalEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        # Parametric Encoder
        self.Enc = HGAPEncoder(hid_channels + 3, hid_channels, 
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
        self.ColorEncoder.apply(weights_init)
        self.NormalEncoder.apply(weights_init)
        
    def forward(self, data):
        pos, _, batch, ptr = data.pos, data.edge_index, data.batch, data.ptr
        
        x_rgb = self.ColorEncoder(data.rgb.view(-1, 4, 3))
        x_normals = self.NormalEncoder(data.normals.view(-1, 4, 3))
        x = torch.cat([x_rgb, x_normals, pos], dim=-1)

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
        # Mesh Feature Encoder
        self.ColorEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        self.NormalEncoder = MeshFeatureEncoder(in_dim=in_channels, pos_dim=8, out_dim=hid_channels//2, num_tokens=4)
        # Parametric Encoder
        self.Enc = LGAPEncoder(hid_channels + 3, hid_channels, 
                               num_convs, pool_factors,
                               num_block,
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
        self.ColorEncoder.apply(weights_init)
        self.NormalEncoder.apply(weights_init)
        
    def forward(self, data):
        pos, edge_index, batch, ptr = data.pos, data.edge_index, data.batch, data.ptr
        
        x_rgb = self.ColorEncoder(data.rgb.view(-1, 4, 3))
        x_normals = self.NormalEncoder(data.normals.view(-1, 4, 3))
        x = torch.cat([x_rgb, x_normals, pos], dim=-1)

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
                 beta: float,
                 load_feature: str = 'all'):
        super().__init__()
        
        self.load_feature = load_feature

        # Mesh Feature Encoders
        if load_feature in ['all', 'rgb']:
            self.ColorEncoder = MeshFeatureEncoder(
                in_dim=in_channels, pos_dim=8, out_dim=hid_channels // 2, num_tokens=4
            )
        if load_feature in ['all', 'normals']:
            self.NormalEncoder = MeshFeatureEncoder(
                in_dim=in_channels, pos_dim=8, out_dim=hid_channels // 2, num_tokens=4
            )

        # Parametric Encoder
        if load_feature == 'all':
            encoder_in_dim = hid_channels + 3
        elif load_feature in ['rgb', 'normals']:
            encoder_in_dim = hid_channels // 2 + 3
        else:
            encoder_in_dim = 3
        
        self.Enc = Encoder(
            encoder_in_dim, hid_channels, num_convs, 
            pool_factors, num_nbrs, num_block, alpha, beta
        )

        # Parametric Decoder
        self.Dec = Decoder(hid_channels, num_convs)

        # Classifier
        self.mlp = MLP([hid_channels, 128, 128, out_channels], dropout=0.5)

        # Initialize weights
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

        if self.load_feature in ['all', 'rgb']:
            self.ColorEncoder.apply(weights_init)
        if self.load_feature in ['all', 'normals']:
            self.NormalEncoder.apply(weights_init)

    def forward(self, data):
        pos, edge_index, batch, ptr = data.pos, data.edge_index, data.batch, data.ptr

        if self.load_feature == 'all':
            x_rgb = self.ColorEncoder(data.rgb.view(-1, 4, 3))
            x_normals = self.NormalEncoder(data.normals.view(-1, 4, 3))
            x = torch.cat([x_rgb, x_normals, pos], dim=-1)

        elif self.load_feature == 'rgb':
            x_rgb = self.ColorEncoder(data.rgb.view(-1, 4, 3))
            x = torch.cat([x_rgb, pos], dim=-1)

        elif self.load_feature == 'normals':
            x_normals = self.NormalEncoder(data.normals.view(-1, 4, 3))
            x = torch.cat([x_normals, pos], dim=-1)

        else:  # load_feature is None
            x = pos.detach().clone()

        # Encoder forward
        pos_down, x_down, batch_down, _ = self.Enc(pos, x, edge_index, batch, ptr)

        # Decoder forward
        f = self.Dec(pos_down[::-1], x_down[::-1], batch_down[::-1])

        y = self.mlp(f)
        return {'y': y}