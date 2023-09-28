import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from Config.config import Config
from Net.GCN import Model as GCN
from Util.Universal_Util.Utils import Transform2H, Transform2R


def ForKinematics(q, hip_left, hip_right, initial_body):
    """
    Input:
        q: [batch_size, length_size, joint_num:6, 3, 3]  世界坐标系下相对于父关节的旋转
        t: [batch_size, length_size, 3]
        initial_body: [batch_size, M:length of connection, 3]
    Return:
        l: location of joints, [batch_size, length_size, joint_num:19/9, 3]
    """
    B, L, N, _, _ = q.shape
    q = q.view(B * L, N, 3, 3)
    hip_left = hip_left.view(B * L, 3, 1)
    hip_right = hip_right.view(B * L, 3, 1)
    _, M, _ = initial_body.shape
    body = initial_body.view(B, M, 3, 1).repeat(L, 1, 1, 1)
    skeleton = Config.skeleton_lower_body.tolist()
    lower_joint_map = Config.lower_joint_map
    lower_joint_map_1 = [13, 14, 15, 17, 18, 19]
    l = torch.zeros((B * L, N + 2, 3, 1), dtype=torch.float32).to(q.device)
    l[:, 0, :, :] = hip_left
    l[:, 4, :, :] = hip_right
    for i, (parent, child) in enumerate(skeleton):
        l[:, lower_joint_map.index(child), :, :] = l[:, lower_joint_map.index(parent), :, :] + torch.matmul(
            q[:, lower_joint_map_1.index(child), :, :], body[:, i + 14, :, :])
    l = l.view(B, L, N + 2, 3)
    return l


class BasePointNet(nn.Module):
    def __init__(self, hidden_dim):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(16)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(32)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=hidden_dim - 3, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(61)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        """
        Input:
            in_mat: [batch_size*length_size, lower_pc_no, xyzrvi:6]
        Return:
            x: [batch_size, length_size, lower_pc_no, out_channels+3]
        """
        x = in_mat.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)
        x = torch.cat((in_mat[:, :, :3], x), -1)

        return x


class FusionModule(nn.Module):
    def __init__(self, hidden_dim=64):
        super(FusionModule, self).__init__()
        # self.fc0 = nn.Linear(hidden_dim * 4 + Config.joint_num_upper * 3, 256)
        self.fc0 = nn.Linear(hidden_dim * 2 + Config.joint_num_upper * 3, 256)

        self.faf0 = nn.ReLU()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.scale = hidden_dim ** -0.5
        self.proj = nn.Linear(128, 256)
        self.fc2 = nn.Linear(128, 6 * 6 + 2 * 3)  # 下半身8个关键点, 6个旋转角
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=-1)
        # self.rnn = nn.LSTM(input_size=128+3, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1,
        #                    bidirectional=True)
        self.rnn_p = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim, num_layers=3, batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.rnn_k = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True, dropout=0.1,
                             bidirectional=True)
        self.rnn_pk = nn.LSTM(input_size=hidden_dim*3, hidden_size=hidden_dim, num_layers=3, batch_first=True, dropout=0.1,
                             bidirectional=True)

    def forward(self, p_vec, k_vec, upper_l, batch_size, length_size):
        """
        Input:
            p_vec: [batch_size, length_size, 128]
            k_vec: [batch_size, length_size, 128]
        Return:
            l: location of joints, [batch_size, length_size, joint_num:19/9, 3]
        """
        upper = upper_l.contiguous().view(batch_size, length_size, -1)
        # p_vec = p_vec.view(batch_size, length_size, -1)
        p_vec = p_vec.view(batch_size * length_size, Config.lower_pc_no, -1)
        k_vec = k_vec.view(batch_size * length_size, Config.joint_num_upper, -1)
        t_q = self.to_q(p_vec)
        t_k = self.to_k(k_vec)
        t_v = self.to_v(k_vec)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_x = t_attn @ t_v

        new_p_vec = torch.cat((p_vec, t_x), -1)
        attn_weights = self.softmax(self.attn(new_p_vec))
        a_vec = torch.sum(new_p_vec * attn_weights, dim=1).view(batch_size, length_size, -1)
        # a_vec, _ = self.rnn_p(a_vec)
        k_vec = k_vec.transpose(-2, -1)
        k_vec = F.avg_pool1d(k_vec, k_vec.size(2)).view(batch_size, length_size, -1)
        # k_vec, _ = self.rnn_k(k_vec)
        ak_vec = torch.cat((a_vec, k_vec), -1)
        ak_vec, _ = self.rnn_pk(ak_vec)
        # x = torch.cat((g_vec, a_vec), -1)
        # x = torch.cat((a_vec, k_vec, upper), -1)
        x = torch.cat((ak_vec, upper), -1)
        x = self.fc0(x)
        x = self.faf0(x)
        x = self.fc1(x)
        x1 = self.faf1(x)
        x = self.fc2(x1)

        q = x[:, :, :6 * 6].reshape(batch_size * length_size * 6, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)
        tmp_x = tmp_x.view(batch_size, length_size, 6, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 6, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 6, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)
        hip_l = x[:, :, -6:-3]
        hip_r = x[:, :, -3:]
        return q, hip_l, hip_r


class PointEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(PointEncoder, self).__init__()
        self.module0 = BasePointNet(hidden_dim=hidden_dim)

    def forward(self, lower_x, h0_g, c0_g, batch_size, length_size):
        lower_x = self.module0(lower_x)
        return lower_x


class KeyEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(KeyEncoder, self).__init__()

        self.gcn = GCN(in_channels=3, hidden_dim=hidden_dim, graph_args={'strategy': 'distance'})

    def forward(self, upper_l, h0, c0, batch_size, length_size):
        """
        Input:
            upper_l: [batch_size, length_size, 15, 3]
        Return:
            g_vec: [batch_size, length_size, 128]
        """
        x = upper_l.view(batch_size, length_size, 15, 3).permute(0, 3, 1, 2).contiguous()
        x_flipped = torch.unsqueeze(x, 4)
        k_vec = self.gcn.extract_feature(x_flipped)
        g_vec = k_vec.view(batch_size * length_size, 15, -1)

        return g_vec


class LowerNet(nn.Module):
    def __init__(self, hidden_dim):
        super(LowerNet, self).__init__()
        self.pointEncoder = PointEncoder(hidden_dim=hidden_dim)
        self.keyEncoder = KeyEncoder(hidden_dim=hidden_dim)
        self.fusion = FusionModule(hidden_dim=hidden_dim)

    def forward(self, upper_l, x, h0_p, c0_p, h0_k, c0_k, initial_body, R, t):
        """
        Input:
            upper_l: [batch_size, length_size, 15, 3]
        Return:
            l: location of lower body joints, [batch_size, length_size, 6, 3]
        """
        if torch.cuda.is_available():
            device = 'cuda:%d' % 0
        else:
            device = 'cpu'
        lower_pc_no = Config.lower_pc_no
        upper_joint_no = Config.joint_num_upper
        batch_size = x.size()[0]
        length_size = x.size()[1]
        pt_size = x.size()[2]
        in_feature_size = x.size()[3]
        x = x.view(batch_size * length_size, pt_size, in_feature_size)
        x = Transform2H(x, batch_size, length_size, pt_size, R, t)

        upper = upper_l.view(batch_size * length_size, -1, 3).clone()

        # plevis_x = upper[:, 0, 0].view(batch_size * length_size, 1).repeat([1, pt_size])
        # plevis_d = torch.norm(upper[:, 0, :], dim=-1).view(batch_size * length_size, 1)
        #
        # x_d = x[..., 3]
        # mask = x_d > plevis_d
        #
        # selected_points = []
        # for b in range(batch_size * length_size):
        #     batch_indices = torch.nonzero(mask[b]).squeeze()  # 找到满足条件的点的索引
        #     if batch_indices.numel() >= lower_pc_no:
        #         batch_indices = batch_indices[:lower_pc_no]  # 如果超过 P 个点，只选择前 P 个
        #     else:
        #         num_padding = lower_pc_no - batch_indices.numel()
        #         padding = torch.zeros(num_padding, dtype=torch.bool, device=x.device)
        #         batch_indices = torch.cat([batch_indices, padding], dim=0)
        #     # print(batch_indices)
        #     selected_points.append(x[b, batch_indices])
        #
        # lower_x = torch.stack(selected_points)

        lower_x = x[..., 0]
        lower_x = lower_x.view(batch_size * length_size, pt_size)
        _, indices = torch.sort(lower_x, dim=1, descending=True)
        indices = indices[:, :lower_pc_no]
        lower_idx = torch.arange(pt_size, dtype=torch.long).to(device).view(1, pt_size) \
            .repeat([batch_size * length_size, 1])
        batch_idx = torch.arange(batch_size * length_size, dtype=torch.long).to(device) \
            .view((batch_size * length_size, 1)) \
            .repeat((1, lower_pc_no))
        lower_idx = lower_idx[batch_idx, indices]
        lower_x = x[batch_idx, lower_idx, :]
        lower_x = lower_x.view(batch_size * length_size, lower_pc_no, -1)

        upper = Transform2H(upper, batch_size, length_size, upper_joint_no, R, t)

        p_vec = self.pointEncoder(lower_x, h0_p, c0_p, batch_size, length_size)
        upper = upper.view(batch_size, length_size, -1, 3)
        k_vec = self.keyEncoder(upper, h0_k, c0_k, batch_size, length_size)
        q, hip_l, hip_r = self.fusion(p_vec, k_vec, upper, batch_size, length_size)
        l = ForKinematics(q, hip_l, hip_r, initial_body)

        l = Transform2R(l, batch_size, length_size, 8, R, t)
        l = l.view(batch_size, length_size, 8, 3)
        return l, q

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
