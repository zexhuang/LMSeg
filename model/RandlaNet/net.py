"""An PyG implementation of RandLA-Net based on the `"RandLA-Net: Efficient
Semantic Segmentation of Large-Scale Point Clouds"
<https://arxiv.org/abs/1911.11236>`_ paper.

Code modified based on: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/randlanet_segmentation.py
"""
import torch
from torch import Tensor

from torch_geometric.nn import MLP, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.utils import softmax


# Default activation and batch norm parameters used by RandLA-Net:
lrelu02_kwargs = {'negative_slope': 0.2}
bn099_kwargs = {'momentum': 0.01, 'eps': 1e-6}


class SharedMLP(MLP):
    """SharedMLP following RandLA-Net paper."""
    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs['plain_last'] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs['act'] = kwargs.get('act', 'LeakyReLU')
        kwargs['act_kwargs'] = kwargs.get('act_kwargs', lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs['norm_kwargs'] = kwargs.get('norm_kwargs', bn099_kwargs)
        super().__init__(*args, **kwargs)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.mlp_encoder = SharedMLP([10, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False,
                                       act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                index: Tensor) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])

        Returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance],
                                   dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding],
                                   dim=1)  # N * K, 2d

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out
    
    
class DilatedResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        return x, pos, batch
    

class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
    

def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """Decimates each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim


class RandlaNet(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        decimation: int = 4,
        num_neighbors: int = 16
    ):
        super().__init__()

        self.decimation = decimation
        
        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        d_bottleneck = max(32, num_classes, num_features)

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 512)
        self.mlp_summit = SharedMLP([512, 512])
        self.fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))
        self.mlp_classif = SharedMLP([d_bottleneck, 64, 32],
                                     dropout=[0.0, 0.5])
        self.fc_classif = Linear(32, num_classes)

    def forward(self, data):
        pos, batch, ptr = data.pos, data.batch, data.ptr
        x = torch.cat([data.rgb, data.normals, pos], dim=-1)

        b1_out = self.block1(self.fc0(x), pos, batch)
        b1_out_decimated, ptr1 = decimate(b1_out, ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, ptr3 = decimate(b3_out, ptr2, self.decimation)

        b4_out = self.block4(*b3_out_decimated)
        b4_out_decimated, _ = decimate(b4_out, ptr3, self.decimation)

        mlp_out = (
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )

        fp4_out = self.fp4(*mlp_out, *b3_out_decimated)
        fp3_out = self.fp3(*fp4_out, *b2_out_decimated)
        fp2_out = self.fp2(*fp3_out, *b1_out_decimated)
        fp1_out = self.fp1(*fp2_out, *b1_out)

        x = self.mlp_classif(fp1_out[0])
        y = self.fc_classif(x)
        return {'y': y}
