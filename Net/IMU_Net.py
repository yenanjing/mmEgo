import time
import torch
import torch.nn as nn


# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# batch*n
def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# batch*6
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


class IMUNet(nn.Module):
    def __init__(self, input_n, output_n, hidden_n, n_rnn_layer, bidirectional=True, dropout=0):
        super(IMUNet, self).__init__()
        self.fc1 = nn.Linear(input_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n * (2 if bidirectional else 1), output_n)  # * (2 if bidirectional else 1)
        self.fc3 = nn.Linear(output_n, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.rnn_fast = nn.LSTM(hidden_n, hidden_n, n_rnn_layer, bidirectional=bidirectional, batch_first=True,
                                dropout=dropout)

        self.rnn_slow = nn.LSTM(2 * hidden_n, hidden_n, n_rnn_layer, bidirectional=bidirectional, batch_first=True,
                                dropout=dropout)

        self.attn = nn.Linear(hidden_n * (2 if bidirectional else 1), 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, imu, h0_i=None):
        """
        Input:
            imu: [batch_size, length_size, 20, f]
        Return:
            R: R_R0R [batch_size, length_size,  3, 3]
            t: H_pos [batch_size, length_size,  3]
        """
        # imu = imu[:, :, :, 9:]
        batch_size, length_size, N, feature = imu.shape

        imu = imu.view(batch_size * length_size, N, -1)
        imu_h = self.relu(self.fc1(imu))
        imu_f, hn_fast = self.rnn_fast(imu_h, h0_i)

        attn_weights = self.softmax(self.attn(imu_f))
        imu_f = torch.sum(imu_f * attn_weights, dim=1)
        imu_f = imu_f.view(batch_size, length_size, -1)
        imu_f, hn_slow = self.rnn_slow(imu_f, h0_i)

        T = self.fc2(imu_f)
        T = T.view(batch_size * length_size, -1)
        R_T = T[:, :6].contiguous()
        R = compute_rotation_matrix_from_ortho6d(R_T)
        t = T[:, 6:]
        R = R.view(batch_size, length_size, 3, 3)
        t = t.view(batch_size, length_size, 3)
        return R, t

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


if __name__ == '__main__':
    model_IMU = IMUNet(15, 6 + 3, 512, 2, True, 0)
