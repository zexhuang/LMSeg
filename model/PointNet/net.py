import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch
from .pointnet_utils import PointNetEncoder


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, get_trans_feat: bool = True):
        super(PointNetSeg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.get_trans_feat = get_trans_feat
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=in_channels)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, data):
        pos, x, batch, batch_size = data.pos, data.x, data.batch, data.batch_size
        x = pos.detach().clone() if x is None else torch.cat([x, pos.detach().clone()], dim=-1)
        
        x = to_dense_batch(x, batch, batch_size=batch_size)[0]
        x = x.permute(0,2,1)
        batch_size, num_points = x.size()[0], x.size()[-1]
        
        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        y = x.view(batch_size * num_points, self.out_channels)
        return {'y': (y, trans_feat)} if self.get_trans_feat else {'y': y}

