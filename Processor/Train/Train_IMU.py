import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import Util.Universal_Util.Utils as utils
from Config.config import Config
from Net.IMU_Net import IMUNet
from Util.Universal_Util.Dataset_sample import PosePC

# from Util.Universal_Util.Dataset import PosePC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_current_path = os.path.dirname(__file__)


# batch*3*3
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m1 = m1.view(-1, 3, 3)
        m2 = m2.view(-1, 3, 3)
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        return torch.sum(theta)


class MMEgo:
    def __init__(self):
        self.vis_loader = None
        self.vis_data = None
        torch.set_printoptions(profile="full")
        self.device = Config.device
        self.num_epochs = Config.epochs
        self.save_slot = 50
        self.frame_no = Config.frame_no
        self.learning_rate = Config.lr
        self.batchsize = Config.batch_size
        self.joint_num = Config.joint_num_all
        self.upper_joint_num = Config.joint_num_upper
        self.model_IMU = IMUNet(15, 6 + 3, 512, 2, True, 0).to(
            self.device)  # input_n, output_n, hidden_n, n_rnn_layer, bidirectional=True, dropout=0
        if Config.IMU_pretrained:
            self.model_IMU.load(Config.model_IMU_path)
        self.pb = Config.pb
        self.Idx = Config.Idx
        self.targetPath = os.path.join(_current_path, './report/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        print('report saved in ' + self.targetPath)
        self.targetPath = os.path.join(_current_path, './model/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        print('model saved in ' + self.targetPath)

        self.targetPath = os.path.join(_current_path, './lossAndacc/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        print('Loss and accuracy saved in ' + self.targetPath)

        self.loss_geodesic = GeodesicLoss()
        self.optimizer_IMU = torch.optim.Adam(self.model_IMU.parameters(), lr=self.learning_rate,
                                              weight_decay=0.001)
        self.train_data = PosePC(batch_length=self.frame_no)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batchsize, shuffle=True, drop_last=False)
        self.test_data = PosePC(train=False, batch_length=self.frame_no)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batchsize, shuffle=True, drop_last=False)
        self.lossfile = open(os.path.join(_current_path, './report/%d/log-loss.txt' % (self.Idx)), 'w')
        self.evalfile = open(os.path.join(_current_path, './report/%d/log-eval.txt' % (self.Idx)), 'w')
        self.loss_total = 0
        self.R_RI = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        self.R_RI = torch.tensor(self.R_RI, dtype=torch.float32, device=self.device)

    def save_models(self, epoch, model):
        torch.save(model.state_dict(), os.path.join(_current_path,
                                                    "./model/{}/epoch{}_batch{}frame{}lr{}.pth".format(self.Idx, epoch,
                                                                                                       self.batchsize,
                                                                                                       self.frame_no,
                                                                                                       self.learning_rate)))

    def train_imu(self):
        early_stopping = utils.EarlyStopping(patience=30)
        loss_l = []
        epochs = self.num_epochs
        for epoch in range(self.num_epochs):
            print("epoch: {}".format(epoch + 1))
            train_loss = self.train_imu_once()
            if (epoch + 1) % self.save_slot == 0:
                self.save_models(epoch, self.model_IMU)
            eval_loss, eval_loss_l = self.eval_imu()
            loss_l.append(eval_loss)
            self.lossfile.write('%d %f\n' % (epoch + 1, eval_loss))
            self.lossfile.write(str(eval_loss_l))
            self.lossfile.write('\n')
            self.lossfile.flush()
            print('Train_loss: {}'.format(train_loss))
            print('Eval_loss: {}  Eval_loss_l (angle, H_pos): {}'.format(eval_loss, eval_loss_l))
            if early_stopping(eval_loss):
                print("Early stopping")
                self.save_models(epoch, self.model_IMU)
                epochs = epoch + 1
                break
        utils.draw_fig(loss_l, "loss", epochs, self.pb, self.Idx)

    def train_imu_once(self):
        self.model_IMU.train()
        training_loss = []
        # train_accu_epoch = []
        for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R) in tqdm(
                enumerate(self.train_loader)):
            target = np.asarray(target)
            imu = np.asarray(imu)
            batch_size, length_size, _, _ = imu.shape
            R_R0R = np.asarray(R_R0R)
            # t_R0R = np.asarray(t_R0R)
            imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device).squeeze()
            target = torch.tensor(target, dtype=torch.float32, device=self.device)
            target_h = target[:, :, 20, :]
            # target_N = target[:, :, 3, :]
            self.optimizer_IMU.zero_grad()
            R_R0R = torch.tensor(R_R0R, dtype=torch.float32, device=self.device)
            # t_R0R = torch.tensor(t_R0R, dtype=torch.float32, device=self.device).squeeze()
            R, t = self.model_IMU(imu_i)

            R = R.view(batch_size, length_size, 3, 3)
            t = t.view(batch_size, length_size, 3)

            # loss = self.loss_geodesic(R, R_R0R)
            loss_1 = self.loss_geodesic(R, R_R0R) / 3.14159265358 * 180
            loss_2 = torch.sum(torch.sqrt(torch.sum(torch.square(t - target_h), dim=-1)))

            loss = loss_1 + 100 * loss_2

            loss.backward()
            self.optimizer_IMU.step()

            training_loss.append((loss / batch_size / length_size).item())

        training_loss = np.mean(training_loss)
        return training_loss

    def eval_imu(self):
        self.model_IMU.eval()
        eval_loss = []
        eval_loss_l = []
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R) in tqdm(
                    enumerate(self.test_loader)):
                # print(type(data))    ##tensor
                target = np.asarray(target)
                imu = np.asarray(imu)
                batch_size, length_size, _, _ = imu.shape
                R_R0R = np.asarray(R_R0R)
                # t_R0R = np.asarray(t_R0R)
                imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device).squeeze()
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                target_h = target[:, :, 20]
                target_N = target[:, :, 3, :]
                self.optimizer_IMU.zero_grad()
                R_R0R = torch.tensor(R_R0R, dtype=torch.float32, device=self.device)
                # t_R0R = torch.tensor(t_R0R, dtype=torch.float32, device=self.device).squeeze()
                R, t = self.model_IMU(imu_i)
                R = R.view(batch_size, length_size, 3, 3)
                t = t.view(batch_size, length_size, 3)

                loss_1 = self.loss_geodesic(R, R_R0R) / 3.14159265358 * 180
                loss_2 = torch.sum(torch.sqrt(torch.sum(torch.square(t - target_h), dim=-1)))
                loss = loss_1 + 100 * loss_2
                # loss = self.loss_L1(t, target_h)
                # loss = 100*self.loss_L1(R, R_R0R)
                # 100*self.loss_fn2(t, t_R0R)
                eval_loss.append(loss.item() / batch_size / length_size)
                eval_loss_l.append([loss_1.item() / batch_size / length_size, loss_2.item() / batch_size / length_size])
        eval_loss = np.mean(eval_loss)
        eval_loss_l = np.mean(eval_loss_l, axis=0)
        return eval_loss, eval_loss_l

    def eval_all_imu(self):
        """画出每一帧的误差"""
        self.model_IMU.load(Config.model_IMU_path)
        self.vis_data = PosePC(train=False, vis=True, batch_length=self.frame_no)
        self.vis_loader = DataLoader(self.vis_data, batch_size=1, shuffle=False, drop_last=False)
        self.model_IMU.eval()
        eval_loss_R_l = []
        eval_loss_H_pos_l = []
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, init_head, ground, foot_contact, R_R0R, t_R0R) in tqdm(
                    enumerate(self.vis_loader)):
                target = np.asarray(target)
                imu = np.asarray(imu)
                batch_size, length_size, _, _ = imu.shape
                R_R0R = np.asarray(R_R0R)
                # t_R0R = np.asarray(t_R0R)
                imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device)
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                target_h = target[:, :, 20]
                target_N = target[:, :, 3, :]
                self.optimizer_IMU.zero_grad()
                R_R0R = torch.tensor(R_R0R, dtype=torch.float32, device=self.device)
                # t_R0R = torch.tensor(t_R0R, dtype=torch.float32, device=self.device).squeeze()
                R, t = self.model_IMU(imu_i)
                R = R.view(batch_size, length_size, 3, 3)
                t = t.view(batch_size, length_size, 3)

                loss_1_l = utils.angle_minus(R, R_R0R).cpu().numpy().tolist()
                loss_2_l = (torch.sqrt(torch.sum(torch.square(t - target_h), dim=-1)) * 100).cpu().numpy().tolist()[0]
                eval_loss_H_pos_l.extend(loss_2_l)
                eval_loss_R_l.extend(loss_1_l)
        utils.draw_loss_frame(eval_loss_R_l, eval_loss_H_pos_l, "eval_loss_per_frame")


if __name__ == '__main__':
    mmEgo = MMEgo()
    mmEgo.train_imu()
    # mmEgo.eval_all_imu()
