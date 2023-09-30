import time

import torch
import torch.nn as nn

from Config.config import Config
from Util.Universal_Util.Utils import Transform2R, Transform2H


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # 将距离中目标点集的坐标为全0的部分替换为无穷大
    dist = torch.where(torch.all(dst == 0, dim=-1).unsqueeze(1).repeat(1, N, 1),
                       torch.tensor(float('inf')).to(src.device), dist)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, nsample]
    Return:
        new_points:, indexed points data, [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def point_ball_set(nsample, xyz, new_xyz):
    """
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3]
        new_xyz: anchor points [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    _, sort_idx = torch.sort(sqrdists)
    sort_idx = sort_idx[:, :, :nsample]
    batch_idx = torch.arange(B, dtype=torch.long).to(device).view((B, 1, 1)).repeat((1, S, nsample))
    centroids_idx = torch.arange(S, dtype=torch.long).to(device).view((1, S, 1)).repeat((B, 1, nsample))
    return group_idx[batch_idx, centroids_idx, sort_idx]


def AnchorInit(x_min=0, x_max=0.6, x_interval=0.3, y_min=-0.3, y_max=0.3, y_interval=0.3, z_min=-0.3, z_max=0.3,
               z_interval=0.3):
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, xyz] => [3,3,3,3]
    """
    x_size = round((x_max - x_min) / x_interval) + 1
    y_size = round((y_max - y_min) / y_interval) + 1
    z_size = round((z_max - z_min) / z_interval) + 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32).to(device)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx = x_min + x_no * x_interval
                ly = y_min + y_no * y_interval
                lz = z_min + z_no * z_interval
                centroids[z_no, y_no, x_no, 0] = lx
                centroids[z_no, y_no, x_no, 1] = ly
                centroids[z_no, y_no, x_no, 2] = lz
    return centroids


def AnchorGrouping(anchors, nsample, xyz, points):
    """
    Input:
        anchors: [B, 9*3*3, 3], npoint=9*3*3
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    _, S, _ = anchors.shape
    idx = point_ball_set(nsample, xyz, anchors)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_anchors = anchors.view(B, S, 1, C).repeat(1, 1, nsample, 1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors

    grouped_points = index_points(points, idx)
    new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+C+D]
    return new_points


def ForKinematics(q, initial_body, head):
    """
    Input:
        q: [batch_size, length_size, joint_num:14, 3, 3]
        t: [batch_size, length_size, 3]
        initial_body: [batch_size, M:length of connection, 3]
    Return:
        l: location of joints, [batch_size, length_size, joint_num:19/9, 3]
    """
    B, L, N, _, _ = q.shape
    q = q.view(B * L, N, 3, 3)
    _, M, _ = initial_body.shape
    body = initial_body.view(B, M, 3, 1).repeat(L, 1, 1, 1)
    skeleton = Config.skeleton_upper_body
    upper_joint_map = Config.upper_joint_map
    head = head.view(B * L, 3, 1)
    l = torch.zeros((B * L, N + 1, 3, 1), dtype=torch.float32).to(q.device)
    l[:, -1, :, :] = head
    for i, (parent, child) in enumerate(skeleton):
        l[:, upper_joint_map.index(child), :, :] = l[:, upper_joint_map.index(parent), :, :] + torch.matmul(
            q[:, upper_joint_map.index(child), :, :], body[:, i, :, :])
    l = l.view(B, L, N + 1, 3)
    return l


class LocalPointNet(nn.Module):
    def __init__(self):
        super(LocalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24 + 4 + 3, out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))  # (Batch, feature, frame_point_number)

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class LocalVoxelNet(nn.Module):
    def __init__(self):
        super(LocalVoxelNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(1, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1))
        self.cb3 = nn.BatchNorm3d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.permute(0, 4, 1, 2, 3)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.view(batch_size, 64)
        return x


class LocalRNN(nn.Module):
    def __init__(self):
        super(LocalRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1,
                           bidirectional=True)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn) = self.rnn(x, (h0, c0))
        return a_vec, hn, cn


class LocalModule(nn.Module):
    def __init__(self):
        super(LocalModule, self).__init__()
        self.template_point = AnchorInit()
        self.z_size, self.y_size, self.x_size, _ = self.template_point.shape
        self.anchor_size = self.z_size * self.y_size * self.x_size
        self.apointnet = LocalPointNet()
        self.avoxel = LocalVoxelNet()
        self.arnn = LocalRNN()

    def forward(self, x, h0, c0, batch_size, length_size, feature_size):
        # x: batch*length, pt_size, 4+24
        anchors = self.template_point.view(1, self.anchor_size, 3).repeat(batch_size * length_size, 1, 1)
        grouped_points = AnchorGrouping(anchors, nsample=8, xyz=x[..., :3], points=x[..., 3:])
        grouped_points = grouped_points.view(batch_size * length_size * self.anchor_size, 8, 3 + feature_size)
        voxel_points, attn_weights = self.apointnet(grouped_points)
        voxel_points = voxel_points.view(batch_size * length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec = self.avoxel(voxel_points)
        voxel_vec = voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn = self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1, 2).contiguous()

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)
        x = torch.cat((in_mat[:, :, :4], x), -1)

        return x


class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24 + 4, out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class CombineModule(nn.Module):
    def __init__(self):
        super(CombineModule, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 14 * 6 + 3)  # 上半身15个关键点, 14个旋转角+1个头部点

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)

        q = x[:, :, :14 * 6].reshape(batch_size * length_size * 14, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)
        tmp_x = tmp_x.view(batch_size, length_size, 14, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 14, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 14, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)
        head = x[:, :, -3:]
        return q, head


class GlobalModule(nn.Module):
    def __init__(self):
        super(GlobalModule, self).__init__()
        self.gpointnet = GlobalPointNet()
        self.grnn = nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1,
                            bidirectional=True)

    def forward(self, x, h0, c0, batch_size, length_size):
        x, attn_weights = self.gpointnet(x)
        x = x.view(batch_size, length_size, -1)
        g_vec, (hn, cn) = self.grnn(x, (h0, c0))
        return g_vec, attn_weights, hn, cn


class MLPHead(nn.Module):
    def __init__(self, joint_num=Config.joint_num_upper):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, (joint_num - 1) * 6 + 3)  # 上半身15个关键点, 14个旋转角+1个头部点

    def forward(self, g_vec, batch_size, length_size):
        x = self.fc1(g_vec)
        x = self.faf1(x)
        x = self.fc2(x)

        q = x[:, :, :14 * 6].reshape(batch_size * length_size * 14, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)
        tmp_x = tmp_x.view(batch_size, length_size, 14, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 14, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 14, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)
        head = x[:, :, -3:]
        return q, head


class UpperNet(nn.Module):
    def __init__(self):
        super(UpperNet, self).__init__()
        self.module0 = PointNet()
        self.module1 = GlobalModule()
        self.mlpHead = MLPHead()

    def forward(self, x, h0_g, c0_g, initial_body, R, t):

        batch_size = x.size()[0]
        length_size = x.size()[1]
        pt_size = x.size()[2]
        xh = Transform2H(x, batch_size, length_size, pt_size, R, t)
        xh = xh.view(batch_size * length_size, pt_size, -1)
        xf = self.module0(xh)
        g_vec, global_weights, hn_g, cn_g = self.module1(xf, h0_g, c0_g, batch_size, length_size)
        q, head = self.mlpHead(g_vec, batch_size, length_size)

        l = ForKinematics(q, initial_body, head)
        l = Transform2R(l, batch_size, length_size, Config.joint_num_upper, R, t)
        l = l.view(batch_size, length_size, -1, 3)
        return l, q, global_weights, hn_g, cn_g

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class UpperNetwlocal(nn.Module):
    def __init__(self):
        super(UpperNetwlocal, self).__init__()
        self.module0 = PointNet()
        self.module1 = GlobalModule()
        self.module2 = LocalModule()
        self.module3 = CombineModule()

    def forward(self, x, h0_g, c0_g, h0_a, c0_a, initial_body, R, t):

        batch_size = x.size()[0]
        length_size = x.size()[1]
        pt_size = x.size()[2]
        x = Transform2H(x, batch_size, length_size, pt_size, R, t)
        x = x.view(batch_size * length_size, pt_size, -1)
        out_feature_size = 24 + 4  # base module输出的维度
        xf = self.module0(x)
        g_vec, global_weights, hn_g, cn_g = self.module1(xf, h0_g, c0_g, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(xf, h0_a, c0_a, batch_size, length_size,
                                                         out_feature_size)
        q, head = self.module3(g_vec, a_vec, batch_size, length_size)
        # q, head = self.mlpHead(g_vec, batch_size, length_size)

        l = ForKinematics(q, initial_body, head)
        l = Transform2R(l, batch_size, length_size, Config.joint_num_upper, R, t)
        l = l.view(batch_size, length_size, -1, 3)
        return l, q, global_weights, anchor_weights, hn_g, cn_g, hn_a, cn_a

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(pathname))
        else:
            self.load_state_dict(torch.load(pathname), map_location=torch.device('cpu'))