import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import Util.Universal_Util.Utils_demo as utils
from Config.config_demo import Config
from Net.IMU_Net import IMUNet
from Net.Lower_Net import LowerNet
from Net.Upper_Net import UpperNet
from Util.Universal_Util.Dataset_action import PoseByAction
from Util.Universal_Util.Dataset_sample import PosePC

# from Util.Universal_Util.Dataset_test import PosePC


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MMEgo():
    def __init__(self):
        torch.set_printoptions(profile="full")
        self.device = Config.device
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
        self.vis_data = PosePC(train=False, vis=True, batch_length=self.frame_no)
        self.vis_loader = DataLoader(self.vis_data, batch_size=1, shuffle=False, drop_last=False)
        self.loss_total = 0

    def angle_loss(self, pred_ske, true_ske):
        pred_vec = pred_ske[:, :, [l for l in self.leaf_kp_all], :] - pred_ske[:, :, [l for l in self.root_kp_all], :]
        true_vec = true_ske[:, :, [l for l in self.leaf_kp_all], :] - true_ske[:, :, [l for l in self.root_kp_all], :]
        cos_sim = torch.nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
        angle_l = torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0)
        return angle_l

    def eval_model(self):
        ## evaluation
        self.model.load(Config.model_lower_path)
        self.model.eval()
        self.Upper_net.eval()
        self.model_IMU.eval()
        eval_loss = []
        eval_accu = []
        accu_ll = []
        accu_upper_l = []
        accu_lower_l = []
        loss_l = []
        angle_ll = []
        eval_loss_l = []
        iteration = 0
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R, R_RtW) in tqdm(
                    enumerate(self.vis_loader)):
                data = np.asarray(data)
                target = np.asarray(target)
                imu = np.asarray(imu)
                skl = np.asarray(skl)
                R_R0R = np.asarray(R_R0R)
                batch_size, seq_len, point_num, dim = data.shape
                imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device)
                data_ti = torch.tensor(data, dtype=torch.float32, device=self.device)
                R_R0R = torch.tensor(R_R0R, dtype=torch.float32, device=self.device)
                target_upper = target[:, :, self.upper_joint_map, :]
                target_upper = torch.tensor(target_upper, dtype=torch.float32, device=self.device)
                target_lower = target[:, :, self.lower_joint_map, :]
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                target_lower = torch.tensor(target_lower, dtype=torch.float32, device=self.device)
                self.optimizer.zero_grad()

                initial_body = torch.tensor(skl, dtype=torch.float32, device=self.device)
                h0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                h0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)

                R_p, t_p = self.model_IMU(imu_i)
                R, t = R_p.clone().detach(), t_p.clone().detach()

                upper, _, _, _, _ = self.Upper_net(data_ti, h0_g, c0_g, initial_body,
                                                   R, t)
                upper_l = upper.clone().detach()
                # upper_l = target_upper
                lower_l, lower_q = self.model(upper_l, data_ti, h0_g, c0_g, h0_a, c0_a,
                                              initial_body, R, t)

                pred_l = torch.zeros((batch_size, seq_len, self.joint_num, 3), dtype=torch.float32, device=self.device)
                pred_l[:, :, self.upper_joint_map, :] = upper_l
                pred_l[:, :, self.lower_joint_map, :] = lower_l
                pred_vec = lower_l[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - lower_l[:, :,
                                                                                                     [
                                                                                                         self.lower_joint_map.index(
                                                                                                             l) for
                                                                                                         l
                                                                                                         in
                                                                                                         self.root_kp],
                                                                                                     :]
                true_vec = target_lower[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - target_lower[
                                                                                                          :, :,
                                                                                                          [
                                                                                                              self.lower_joint_map.index(
                                                                                                                  l)
                                                                                                              for l
                                                                                                              in
                                                                                                              self.root_kp],
                                                                                                          :]
                loss_1 = self.loss_fn2(lower_l, target_lower)
                lower_l = lower_l.view(batch_size * seq_len, 8, 3)

                loss_3 = self.loss_fn2(pred_vec, true_vec)
                loss = loss_1
                eval_loss.append(loss.item() / batch_size / (seq_len))
                eval_loss_l.append([loss_1.item() / batch_size / (seq_len), loss_3.item() / batch_size / (seq_len)])

                lower_l = lower_l.view(batch_size, seq_len, 8, 3)
                accu_upper = torch.sqrt(torch.sum(torch.square(upper_l - target_upper), dim=-1))
                accu_upper = torch.mean(accu_upper).item()

                accu_lower = torch.sqrt(torch.sum(torch.square(lower_l - target_lower), dim=-1))
                accu_lower = torch.mean(accu_lower).item()
                angle_l = torch.mean(self.angle_loss(pred_l, target), dim=0).mean(dim=0).cpu().numpy().tolist()
                accu_a = torch.sqrt(torch.sum(torch.square(pred_l - target), dim=-1))
                accu = torch.mean(accu_a).item()
                accu_l = torch.mean(accu_a, dim=0).mean(dim=0).cpu().numpy().tolist()
                accu_ll.append(accu_l)
                eval_accu.append(accu)
                accu_lower_l.append(accu_lower)
                accu_upper_l.append(accu_upper)
                angle_ll.append(angle_l)
                iteration += 1
            eval_loss = np.mean(eval_loss)
            eval_loss_l = np.mean(eval_loss_l, axis=0)
            # eval_loss_l = np.mean(eval_loss_l, axis=0)
            eval_loss_l[0] /= self.lower_joint_num
            angle_ll = np.mean(angle_ll, axis=0)
            eval_accu = np.mean(eval_accu)
            accu_ll = np.mean(accu_ll, axis=0)
            loss_l.append(eval_loss)
            accu_lower_l = np.mean(accu_lower_l)
            accu_upper_l = np.mean(accu_upper_l)

        print('Average Joint Localization Error(cm): {}'.format(eval_accu * 100))
        print('Average UpperBody Joint Localization Error(cm): {}'.format(accu_upper_l * 100))
        print('Average LowerBody Joint Localization Error(cm): {}'.format(accu_lower_l * 100))
        print('Average Joint Rotation Error(°): {}'.format(sum(angle_ll) / len(angle_ll)))
        print('Per Joint Localization Error(cm): {}'.format(accu_ll * 100))
        results = (accu_ll * 100).tolist()
        # results.append(eval_accu.item())
        utils.draw_bar(results, self.Idx, 'pos')
        return eval_loss, eval_loss_l, eval_accu, accu_lower_l, accu_ll, angle_ll

    def eval_all_skeleton(self):
        self.model.load(Config.model_lower_path)
        self.model_IMU.eval()
        self.Upper_net.eval()
        self.model.eval()
        index = 0
        action_data = PoseByAction(train=False, vis=True, batch_length=self.frame_no)
        action_loader = DataLoader(action_data, batch_size=Config.batch_per_action, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R, R_RtW) in tqdm(
                    enumerate(action_loader)):
                data = np.asarray(data)
                target = np.asarray(target)
                imu = np.asarray(imu)
                skl = np.asarray(skl)
                R_RtW = np.asarray(R_RtW)
                ground = np.asarray(ground)
                batch_size, seq_len, point_num, dim = data.shape
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
                R_RtW = R_RtW.reshape((batch_size * seq_len, 3, 3))
                real = real @ R_RtW
                pred = pred @ R_RtW
                # real = real @ R_RtW.squeeze(0)
                # pred = pred @ R_RtW.squeeze(0)
                ground = ground.reshape(batch_size * seq_len, 4)
                floor_level = ground[:, -1]
                # show_s_2=torso_l.view(batch_size * seq_len, joint_num, 3).cpu().numpy()
                # utils.draw3Dpose_frames(pred, real, index, floor_level)
                if Config.colab:
                    utils.draw3Dpose_action_gif_colab(pred, real, batch_idx, floor_level)
                else:
                    utils.draw3Dpose_action_gif(pred, real, batch_idx, floor_level)

                index += batch_size * seq_len


if __name__ == '__main__':
    mmEgo = MMEgo()
    mmEgo.eval_all_skeleton()
    # mmEgo.eval_model()
