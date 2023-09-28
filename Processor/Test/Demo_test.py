import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import Util.Universal_Util.Utils as utils
from Config.config_demo import Config
from Net.IMU_Net import IMUNet
from Net.Lower_Net import LowerNet
from Net.Upper_Net import UpperNet
from Util.Universal_Util.Dataset_sample import PosePC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.autograd.set_detect_anomaly(True)

class MMEgo():
    def __init__(self):
        torch.set_printoptions(profile="full")
        if torch.cuda.is_available():
            self.device = 'cuda:%d' % 0
        else:
            self.device = 'cpu'
        self.num_epochs = 500
        self.save_slot = 50
        self.frame_no = Config.frame_no
        self.learning_rate = 3e-4
        self.batchsize = 20
        self.joint_num = Config.joint_num_all
        self.upper_joint_num = Config.joint_num_upper
        self.lower_joint_num = Config.joint_num_lower
        self.pb = 10
        self.Idx = Config.Idx
        self.skeleton_lower = Config.skeleton_lower_body
        self.skeleton = Config.skeleton_all
        self.hand_joint_map = Config.hand_joint_map
        self.upper_joint_map = Config.upper_joint_map
        self.lower_joint_map = Config.lower_joint_map
        self.root_kp = self.skeleton_lower[:, 0]
        self.leaf_kp = self.skeleton_lower[:, 1]
        self.root_kp = torch.tensor(self.root_kp, dtype=torch.long, device=self.device)
        self.leaf_kp = torch.tensor(self.leaf_kp, dtype=torch.long, device=self.device)
        self.root_kp_all = self.skeleton[:, 0]
        self.leaf_kp_all = self.skeleton[:, 1]
        self.root_kp_all = torch.tensor(self.root_kp_all, dtype=torch.long, device=self.device)
        self.leaf_kp_all = torch.tensor(self.leaf_kp_all, dtype=torch.long, device=self.device)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
        self.loss_fn2 = torch.nn.L1Loss(reduction='sum')
        self.model = LowerNet(hidden_dim=64).to(self.device)
        if Config.Lower_pretrained:
            self.model.load(Config.model_lower_path)
        self.model_IMU = IMUNet(15, 6 + 3, 512, 2, True, 0.1).to(
            self.device)  # input_n, output_n, hidden_n, n_rnn_layer, bidirectional=True, dropout=0
        self.model_IMU.load(Config.model_IMU_path)
        self.Upper_net = UpperNet().to(self.device)
        self.Upper_net.load(Config.model_upper_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_data = PosePC(batch_length=self.frame_no)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batchsize, shuffle=True, drop_last=False)
        self.test_data = PosePC(train=False, batch_length=self.frame_no)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batchsize, shuffle=True, drop_last=False)
        self.loss_total = 0

    def save_models(self, epoch, model):
        torch.save(model.state_dict(),
                   "./model/{}/epoch{}_batch{}frame{}lr{}.pth".format(self.Idx, epoch, self.batchsize, self.frame_no,
                                                                      self.learning_rate))

    def eval_all_skeleton(self):
        """画出每一帧的骨架"""
        self.model.load(Config.model_lower_path)
        self.vis_data = PosePC(train=False, vis=True, batch_length=self.frame_no)
        self.vis_loader = DataLoader(self.vis_data, batch_size=1, shuffle=False, drop_last=False)
        self.model_IMU.eval()
        self.Upper_net.eval()
        self.model.eval()
        index = 0
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R, R_RtW) in tqdm(
                    enumerate(self.vis_loader)):
                # print(type(data))    ##tensor
                data = np.asarray(data)
                target = np.asarray(target)
                imu = np.asarray(imu)
                skl = np.asarray(skl)
                R_RtW = np.asarray(R_RtW)
                ground = np.asarray(ground)
                # init_head = np.asarray(init_head)
                # foot_contact = np.asarray(foot_contact)
                batch_size, seq_len, point_num, dim = data.shape
                imu = np.asarray(imu)
                imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device)
                data_ti = torch.tensor(data, dtype=torch.float32, device=self.device)
                target_upper = target[:, :, self.upper_joint_map, :]
                target_upper = torch.tensor(target_upper, dtype=torch.float32, device=self.device)
                target_lower = target[:, :, self.lower_joint_map, :]
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                # foot_contact = torch.tensor(foot_contact, dtype=torch.float32, device=self.device).squeeze()
                target_lower = torch.tensor(target_lower, dtype=torch.float32, device=self.device)
                self.optimizer.zero_grad()

                initial_body = torch.tensor(skl, dtype=torch.float32, device=self.device)
                h0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                h0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)

                R_p, t_p = self.model_IMU(imu_i)
                R, t = R_p.clone().detach(), t_p.clone().detach()
                # pred_l, pred_q, pred_t, pred_g, _, _, _, _, _, _ = model(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body)
                upper, _, _, _, _ = self.Upper_net(data_ti, h0_g, c0_g, initial_body, R, t)
                upper_l = upper.clone().detach()
                lower_l, lower_q = self.model(upper_l, data_ti, h0_g, c0_g, h0_a, c0_a,
                                              initial_body, R, t)

                pred_l = torch.zeros((batch_size, seq_len, self.joint_num, 3), dtype=torch.float32, device=self.device)
                pred_l[:, :, self.upper_joint_map, :] = upper_l
                pred_l[:, :, self.lower_joint_map, :] = lower_l
                # data = data_ti.view(batch_size * seq_len, point_num, dim).cpu().numpy()
                real = target.view(batch_size * seq_len, self.joint_num, 3).cpu().numpy()
                pred = pred_l.view(batch_size * seq_len, self.joint_num, 3).cpu().numpy()
                real = real @ R_RtW.squeeze(0)
                pred = pred @ R_RtW.squeeze(0)
                ground = ground.reshape(batch_size * seq_len, 4)
                floor_level = ground[:, -1]
                # show_s_2=torso_l.view(batch_size * seq_len, joint_num, 3).cpu().numpy()
                utils.draw3Dpose_frames(pred, real, index, floor_level)
                index += batch_size * seq_len


if __name__ == '__main__':
    mmEgo = MMEgo()
    mmEgo.eval_all_skeleton()
