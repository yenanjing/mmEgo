import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import Util.Universal_Util.Utils as utils
from Config.config import Config
from Net.IMU_Net import IMUNet
from Net.Lower_Net import LowerNet
from Net.Upper_Net import UpperNet
# from Util.Universal_Util.Dataset import PosePC
from Util.Universal_Util.Dataset_sample import PosePC


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_current_path = os.path.dirname(__file__)

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
        self.targetPath = os.path.join(_current_path, './report/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        else:
            print('路径已经存在！')
        self.targetPath = os.path.join(_current_path, './model/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        else:
            print('路径已经存在！')
        self.targetPath = os.path.join(_current_path, './lossAndacc/%d' % (self.Idx))
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        else:
            print('路径已经存在！')
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
        self.lossfile = open(os.path.join(_current_path, './report/%d/log-loss.txt' % (self.Idx)), 'w')
        self.evalfile = open(os.path.join(_current_path, './report/%d/log-eval.txt' % (self.Idx)), 'w')
        self.loss_total = 0

    def save_models(self, epoch, model):
        torch.save(model.state_dict(), os.path.join(_current_path,
                                                    "./model/{}/epoch{}_batch{}frame{}lr{}.pth".format(self.Idx, epoch,
                                                                                                       self.batchsize,
                                                                                                       self.frame_no,
                                                                                                       self.learning_rate)))

    def angle_loss(self, pred_ske, true_ske):
        pred_vec = pred_ske[:, :, [l for l in self.leaf_kp_all], :] - pred_ske[:, :, [l for l in self.root_kp_all], :]
        true_vec = true_ske[:, :, [l for l in self.leaf_kp_all], :] - true_ske[:, :, [l for l in self.root_kp_all], :]
        cos_sim = torch.nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
        angle_l = torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0)
        return angle_l

    def angle_loss_lower(self, pred_ske, true_ske):
        pred_vec = pred_ske[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - pred_ske[:, :,
                                                                                              [
                                                                                                  self.lower_joint_map.index(
                                                                                                      l) for l in
                                                                                                  self.root_kp], :]
        true_vec = true_ske[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - true_ske[:, :,
                                                                                              [
                                                                                                  self.lower_joint_map.index(
                                                                                                      l) for l in
                                                                                                  self.root_kp], :]
        cos_sim = torch.nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
        angle_l = torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0)
        return angle_l

    def train_lower(self):
        early_stopping = utils.EarlyStopping(patience=30)
        loss_l = []
        acc_l = []
        epochs = self.num_epochs
        for epoch in range(self.num_epochs):
            print("epoch: {}".format(epoch + 1))
            train_accu, train_loss = self.train_once()
            if (epoch + 1) % self.save_slot == 0:
                self.save_models(epoch, self.model)
            eval_loss, eval_loss_l, eval_accu, accu_lower_l, accu_ll, angle_ll = self.eval_model()
            loss_l.append(eval_loss)
            acc_l.append(eval_accu)
            self.lossfile.write('%d %f\n' % (epoch + 1, eval_loss))
            self.lossfile.write(str(eval_loss_l))
            self.lossfile.write('\n')
            self.lossfile.flush()

            self.evalfile.write('%d %f %f %f\n' % (epoch + 1, eval_accu, accu_lower_l, sum(angle_ll) / len(angle_ll)))
            self.evalfile.write(str(accu_ll))
            self.evalfile.write('\n')
            self.evalfile.write(str(angle_ll))
            self.evalfile.write('\n')
            self.evalfile.write(str(accu_ll[self.hand_joint_map]))
            self.evalfile.write('\n')
            self.evalfile.flush()
            print('\n wrist elbow(l, r): {}'.format(accu_ll[self.hand_joint_map]))
            print('Average Joint Localization Error: {}'.format(eval_accu))
            print('Average LowerBody Joint Localization Error: {}'.format(accu_lower_l))
            print('Average Joint Rotation Error: {}'.format(sum(angle_ll) / len(angle_ll)))
            print(accu_ll)
            print('Eval_loss: {} Eval_loss_l (l, g): {} \n Angle_loss: {}'.format(eval_loss,
                                                                                  eval_loss_l,
                                                                                  angle_ll))
            if early_stopping(eval_loss):
                print("Early stopping")
                self.save_models(epoch, self.model)
                epochs = epoch + 1
                break
        results = accu_ll.tolist()
        results.append(eval_accu.item())
        utils.draw_fig(loss_l, "loss", epochs, self.pb, self.Idx)
        utils.draw_fig(acc_l, "acc", epochs, self.pb, self.Idx)
        utils.draw_bar(results, self.Idx, 'pos')

    def train_once(self):
        self.model.train()
        self.model_IMU.eval()
        self.Upper_net.eval()
        # self.model_IMU.train()
        # self.Upper_net.train()
        training_loss = []
        train_accu_epoch = []
        for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R) in tqdm(
                enumerate(self.train_loader)):
            # print(type(data))    ##tensor
            data = np.asarray(data)
            target = np.asarray(target)
            R_R0R = np.asarray(R_R0R)
            imu = np.asarray(imu)
            skl = np.asarray(skl)
            ground = np.asarray(ground)
            batch_size, seq_len, point_num, dim = data.shape

            imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device).squeeze()
            R_R0R = torch.tensor(R_R0R, dtype=torch.float32, device=self.device)
            data_ti = torch.tensor(data, dtype=torch.float32, device=self.device)
            target_lower = target[:, :, self.lower_joint_map, :]
            target_upper = target[:, :, self.upper_joint_map, :]
            target_upper = torch.tensor(target_upper, dtype=torch.float32, device=self.device)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)
            target_lower = torch.tensor(target_lower, dtype=torch.float32, device=self.device)
            ground = torch.tensor(ground, dtype=torch.float32, device=self.device).squeeze()
            self.optimizer.zero_grad()
            initial_body = torch.tensor(skl, dtype=torch.float32, device=self.device)
            h0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
            c0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
            h0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
            c0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)

            R_p, t_p = self.model_IMU(imu_i)
            R, t = R_p.clone().detach(), t_p.clone().detach()
            # R, t = R_p.clone(), t_p.clone()
            # R, t = R_R0R, torch.zeros((batch_size, seq_len, 3), dtype=torch.float32, device=self.device)

            upper, _, _, _, _ = self.Upper_net(data_ti, h0_g, c0_g, initial_body, R, t)
            upper_l = upper.clone().detach()
            # upper_l = upper.clone()
            # upper_l = target_upper
            lower_l, lower_q = self.model(upper_l, data_ti, h0_g, c0_g, h0_a, c0_a,
                                          initial_body, R, t)
            pred_vec = lower_l[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - lower_l[:, :,
                                                                                                 [
                                                                                                     self.lower_joint_map.index(
                                                                                                         l) for l
                                                                                                     in
                                                                                                     self.root_kp], :]
            true_vec = target_lower[:, :, [self.lower_joint_map.index(l) for l in self.leaf_kp], :] - target_lower[:, :,
                                                                                                      [
                                                                                                          self.lower_joint_map.index(
                                                                                                              l)
                                                                                                          for l
                                                                                                          in
                                                                                                          self.root_kp],
                                                                                                      :]

            loss_0 = self.loss_fn2(lower_l, target_lower)

            # lower_l = lower_l.view(batch_size * seq_len, 8, 3)

            # loss_4 = self.loss_fn2(pred_vec, true_vec)

            loss = loss_0
            loss.backward()
            self.optimizer.step()
            training_loss.append(loss.item())

            lower_l = lower_l.view(batch_size, seq_len, 8, 3)
            train_accu = torch.mean(torch.sqrt(torch.sum(torch.square(lower_l - target_lower), dim=-1))).item()
            train_accu_epoch.append(train_accu)
        return train_accu_epoch, training_loss

    def eval_model(self):
        ## evaluation
        self.model.eval()
        self.Upper_net.eval()
        self.model_IMU.eval()
        eval_loss = []
        eval_accu = []
        accu_ll = []
        accu_lower_l = []
        loss_l = []
        angle_ll = []
        eval_loss_l = []
        iteration = 0
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, ground, foot_contact, R_R0R, t_R0R) in tqdm(
                    enumerate(self.test_loader)):
                data = np.asarray(data)
                target = np.asarray(target)
                imu = np.asarray(imu)
                skl = np.asarray(skl)
                R_R0R = np.asarray(R_R0R)
                batch_size, seq_len, point_num, dim = data.shape
                imu = np.asarray(imu)
                imu_i = torch.tensor(imu, dtype=torch.float32, device=self.device).squeeze()
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
                # R, t = R_R0R, torch.zeros((batch_size, seq_len, 3), dtype=torch.float32, device=self.device)

                # pred_l, pred_q, pred_t, pred_g, _, _, _, _, _, _ = model(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body)
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
                accu_lower = torch.sqrt(torch.sum(torch.square(lower_l - target_lower), dim=-1))
                accu_lower = torch.mean(accu_lower).item()
                angle_l = torch.mean(self.angle_loss(pred_l, target), dim=0).mean(dim=0).cpu().numpy().tolist()
                accu_a = torch.sqrt(torch.sum(torch.square(pred_l - target), dim=-1))
                accu = torch.mean(accu_a).item()
                accu_l = torch.mean(accu_a, dim=0).mean(dim=0).cpu().numpy().tolist()
                accu_ll.append(accu_l)
                eval_accu.append(accu)
                accu_lower_l.append(accu_lower)
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
        return eval_loss, eval_loss_l, eval_accu, accu_lower_l, accu_ll, angle_ll

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
            for batch_idx, (data, target, skl, imu, init_head, ground, foot_contact, R_R0R, t_R0R, R_RtW) in tqdm(
                    enumerate(self.vis_loader)):
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
                target_lower = torch.tensor(target_lower, dtype=torch.float32, device=self.device)
                self.optimizer.zero_grad()

                initial_body = torch.tensor(skl, dtype=torch.float32, device=self.device)
                h0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                h0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)
                c0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=self.device)

                R_p, t_p = self.model_IMU(imu_i)
                R, t = R_p.clone().detach(), t_p.clone().detach()
                upper, _, _, _, _, _, _, _ = self.Upper_net(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body,
                                                            R, t)
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

    def eval_hand(self):
        """画出多帧的手部姿态"""
        self.model.load(Config.model_lower_path)
        self.vis_data = PosePC(train=False, vis=True, batch_length=self.frame_no)
        self.vis_loader = DataLoader(self.vis_data, batch_size=1, shuffle=False, drop_last=False)
        self.model_IMU.eval()
        self.Upper_net.eval()
        self.model.eval()
        index = 0
        with torch.no_grad():
            for batch_idx, (data, target, skl, imu, init_head, ground, foot_contact, R_R0R, t_R0R, R_RtW) in tqdm(
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

                time_st = time.time()
                R_p, t_p = self.model_IMU(imu_i)
                time_end_1 = time.time()
                time_sum_1 = time_end_1 - time_st
                print(time_sum_1)
                R, t = R_p.clone().detach(), t_p.clone().detach()
                # pred_l, pred_q, pred_t, pred_g, _, _, _, _, _, _ = model(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body)
                upper, _, _, _, _, _, _, _ = self.Upper_net(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body,
                                                            R, t)
                time_end_2 = time.time()
                time_sum_2 = time_end_2 - time_end_1
                print(time_sum_2)
                upper_l = upper.clone().detach()
                lower_l, lower_q = self.model(upper_l, data_ti, h0_g, c0_g, h0_a, c0_a,
                                              initial_body, R, t)
                time_end_3 = time.time()
                time_sum_3 = time_end_3 - time_end_2
                print(time_sum_3)
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
                # utils.draw_hand(pred[:, 11, :].squeeze(), real[:, 11, :].squeeze(), index)
                index += batch_size * seq_len


if __name__ == '__main__':
    mmEgo = MMEgo()
    mmEgo.train_lower()
    # mmEgo.eval_all_skeleton()
