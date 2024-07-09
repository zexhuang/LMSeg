import torch
import torch.nn.functional as F

from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv


class DeeperGCN(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 hidden_channels: int = 64, 
                 num_layers: int = 28):
        super().__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, 
                           hidden_channels, 
                           aggr='softmax', 
                           t=1.0, 
                           learn_t=True, 
                           msg_norm=True, 
                           learn_msg_scale=True,
                           num_layers=2, 
                           norm='layer', 
                           flow='target_to_source')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        pos, x = data.pos, data.x
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)
        
        edge_index = data.edge_index
        edge_attr = x[edge_index[1]] - x[edge_index[0]]
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        y = self.lin(x)
        return {'y': y}