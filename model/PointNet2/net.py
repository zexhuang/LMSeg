import torch
import torch.nn as nn

from torch_geometric.nn import PointNetConv, MLP
from torch_geometric.nn import fps, radius, global_max_pool, knn_interpolate
    

class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=32, num_workers=5)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):            
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
    

class FPModule(nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 pool_ratio: float = 0.25,
                 num_nbrs: int = 3):
        super().__init__()
        self.sa1_module = SAModule(pool_ratio, 0.1, MLP([in_channels, 32, 32, 64], plain_last=False))
        self.sa2_module = SAModule(pool_ratio, 0.2, MLP([64 + 3, 64, 64, 128], plain_last=False))
        self.sa3_module = SAModule(pool_ratio, 0.4, MLP([128 + 3, 128, 128, 256], plain_last=False))
        self.sa4_module = SAModule(pool_ratio, 0.8, MLP([256 + 3, 256, 256, 512], plain_last=False))

        self.fp4_module = FPModule(num_nbrs, MLP([512 + 256, 256, 256], plain_last=False))
        self.fp3_module = FPModule(num_nbrs, MLP([256 + 128, 256, 256], plain_last=False))
        self.fp2_module = FPModule(num_nbrs, MLP([256 + 64, 256, 128], plain_last=False))
        self.fp1_module = FPModule(num_nbrs, MLP([128 + in_channels - 3, 128, 128, 128], plain_last=False))
        
        self.mlp = MLP([128, 128, 128, out_channels], dropout=0.5)
        
    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)
        
        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        z, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        
        y = self.mlp(z)
        return {'y':y} 