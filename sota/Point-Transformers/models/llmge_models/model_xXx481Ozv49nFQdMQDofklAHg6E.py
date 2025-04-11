# --PROMPT LOG--
import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance


# --OPTION--
import torch

def square_distance(src, dst):
    """
    Calculate square distance between each source point and each destination point.

    Args:
        src (Tensor): Source points with shape (B, N, C).
        dst (Tensor): Destination points with shape (B, M, C).

    Returns:
        Tensor: Square distance matrix with shape (B, N, M).
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    src_expand = src.unsqueeze(2).expand(-1, -1, M, -1)
    dst_expand = dst.unsqueeze(0).expand(B, N, -1, -1)
    return torch.sum((src_expand - dst_expand) ** 2, dim=-1)

def farthest_point_sample(xyz, npoint):
    """
    Select farthest point samples from a given point cloud.

    Args:
        xyz (Tensor): Point cloud with shape (B, N, C).
        npoint: int: Number of samples to select.
        super().__init__()

    Returns:
        Tensor: Sampled indices with shape (B, npoint).
    """
    self.npoint = npoint
    self.xyz = xyz

    def _get_farthest_point(xyz_batch, idx):
        """
        Helper function to select the farthest point in a batch.
        """
        dists = square_distance(xyz_batch, xyz_batch[idx])
        farthest_idx = dists.argsort(dim=-1, descending=True)[..., :1]
        return farthest_idx

    self.farthest_idx = torch.cat([_get_farthest_point(xyz[i], torch.zeros(1, dtype=torch.long, device=xyz.device)) for i in range(xyz.shape[0])])

def index_points(points, idx):
    """
    Index points along a specific dimension.

    Args:
        points (Tensor): Points to index with shape (B, N, ...).
        idx (Tensor): Indices to index with shape (B, M).

    Returns:
        Tensor: Indexed points with shape (B, M, ...).
    """
    device = points.device
    B, N, _ = points.shape
    idx_expand = idx.unsqueeze(2).expand(B, -1, points.size(-1))
    return points.gather(1, idx_expand)

def sample_and_group(npoint, nsample, xyz, points):
    """
    Sample and group points in the point cloud.

    Args:
        npoint (int): Number of points to sample.
        nsample (int): Number of samples per point.
        xyz (Tensor): Point cloud with shape (B, N, 3).
        points (Tensor): Additional features with shape (B, N, C).

    Returns:
        Tuple[Tensor, Tensor]:
            New point cloud with shape (B, S, 3).
            New features with shape (B, S, C * (1 + nsample)).
        """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xy
# --OPTION--
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

# --OPTION--
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
# --OPTION--
class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

# --OPTION--
class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x