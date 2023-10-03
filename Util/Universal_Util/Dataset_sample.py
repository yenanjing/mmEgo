import glob
import os
import re

import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset

from Config.config import Config


class PosePC(Dataset):
    def __init__(self, train=True, vis=False, batch_length=None):
        self.vis = vis
        self.train = train
        self.pc_no = Config.pc_no
        self.frame_no = Config.frame_no
        self.R_RI = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        self.R_ttb = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.R_ctw = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 彩色相机转世界的旋转矩阵

        if batch_length is not None:
            self.frame_no = batch_length

        self.body_length_all = []
        self.initial_body = None
        self.initial_body_unit = None
        # 选择21个关节点
        self.joint_selection = Config.kinect_joint_selection
        self.skeleton = Config.skeleton_all.tolist()

        self.data_ti_, self.data_key_, self.imu_, self.skl_, self.ground_, \
        self.foot_contact_, self.R_R0R_, self.t_R0R_, self.R_RtW_ \
            = self.dataRead()

        if self.vis is False:
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.data_ti_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.data_key_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.skl_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.ground_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.foot_contact_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.imu_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.R_R0R_)
            random_state = np.random.RandomState(Config.dataset_random_seed)
            random_state.shuffle(self.t_R0R_)
        if self.train:  # train
            self.data_ti = self.data_ti_[0:int(len(self.data_ti_) * 0.8)]
            self.data_key = self.data_key_[0:int(len(self.data_key_) * 0.8)]
            self.skl = self.skl_[0:int(len(self.skl_) * 0.8)]
            self.imu = self.imu_[0:int(len(self.imu_) * 0.8)]
            self.ground = self.ground_[0:int(len(self.ground_) * 0.8)]
            self.foot_contact = self.foot_contact_[0:int(len(self.foot_contact_) * 0.8)]
            self.R_R0R = self.R_R0R_[0:int(len(self.R_R0R_) * 0.8)]
            self.t_R0R = self.t_R0R_[0:int(len(self.t_R0R_) * 0.8)]
        elif not self.vis:  # test
            self.data_test_ti = self.data_ti_[int(len(self.data_ti_) * 0.8):]
            self.data_test_key = self.data_key_[int(len(self.data_key_) * 0.8):]
            self.skl_test = self.skl_[int(len(self.skl_) * 0.8):]
            self.imu_test = self.imu_[int(len(self.imu_) * 0.8):]
            self.ground_test = self.ground_[int(len(self.ground_) * 0.8):]
            self.foot_contact_test = self.foot_contact_[int(len(self.foot_contact_) * 0.8):]
            self.R_R0R_test = self.R_R0R_[int(len(self.R_R0R_) * 0.8):]
            self.t_R0R_test = self.t_R0R_[int(len(self.t_R0R_) * 0.8):]

    def __getitem__(self, index):
        if self.vis:
            ti, label, skl, imu, ground, foot_contact, R_R0R, t_R0R, R_RtW = \
                self.data_ti_[index], self.data_key_[index], self.skl_[index], self.imu_[
                    index], self.ground_[index], self.foot_contact_[index], self.R_R0R_[index], \
                self.t_R0R_[index], self.R_RtW_[index]
            return ti, label, skl, imu, ground, foot_contact, R_R0R, t_R0R, R_RtW
        else:
            if self.train:
                ti, label, skl, imu, ground, foot_contact, R_R0R, t_R0R \
                    = self.data_ti[index], self.data_key[index], self.skl[index], self.imu[index], \
                      self.ground[index], self.foot_contact[index], self.R_R0R[index], \
                      self.t_R0R[index]
            else:  # test 固定的ti kinect 数据
                ti, label, skl, imu, ground, foot_contact, R_R0R, t_R0R \
                    = self.data_test_ti[index], self.data_test_key[index], self.skl_test[index], \
                      self.imu_test[index], \
                      self.ground_test[index], self.foot_contact_test[index], \
                      self.R_R0R_test[index], self.t_R0R_test[index]
            return ti, label, skl, imu, ground, foot_contact, R_R0R, t_R0R

    def __len__(self):
        if self.vis:
            return len(self.data_ti_)
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        list_all_ti = []
        list_all_imu = []
        list_all_R_R0R = []
        list_all_t_R0R = []
        list_all_R_RtW = []
        list_all_key_xyz = []
        list_all_skl = []
        list_all_ground = []
        list_all_foot_contact = []

        _current_path = os.path.dirname(__file__)
        rootDir = os.path.join(_current_path, "../../Resource/Sample_data")
        subDirlist = sorted(os.listdir(rootDir), key=lambda x: int(x))
        orientation_ref = []
        R_ref = []
        body_length = []
        st = 0
        for act in range(0, len(subDirlist)):
            subsubDir = os.path.join(rootDir, subDirlist[act])
            subsubDirlist = sorted(os.listdir(subsubDir))
            # no_per_action = 1 if self.vis else len(subsubDirlist)
            no_per_action = len(subsubDirlist)
            for j in range(0, no_per_action):  # 遍历每一次采集的数据
                path = os.path.join(subsubDir, subsubDirlist[j])
                if not os.path.isdir(path):
                    continue
                # print(path)
                regex = re.compile(r'\d+')
                matFilelist = sorted(glob.glob(os.path.join(path, "*.mat")),
                                     key=lambda x: [int(y) for y in (regex.findall(os.path.basename(x)))])
                if len(matFilelist) == 0:
                    continue
                if act == 0 and j == 0:
                    continue
                list_person_once_ti = []
                list_person_once_imu = []
                list_person_once_R_R0R = []
                list_person_once_t_R0R = []
                list_person_once_R_RtW = []
                list_person_once_key_xyz = []
                list_person_once_ground = []
                list_person_once_foot_contact = []
                # frames = self.frame_no if self.vis else len(matFilelist)
                frames = len(matFilelist)
                for frame in range(0, frames):  # 遍历每一帧
                    data = scio.loadmat(matFilelist[frame])

                    pc_xyz_key3 = data['pc_xyz_key_2'][:, 0:3]
                    pc_xyziv_ti3 = data['pc_xyziv_ti2'][:, 0:5].tolist()

                    pc_xyziv_ti3 = np.asarray(pc_xyziv_ti3)
                    if len(pc_xyziv_ti3) == 0:
                        continue
                    pc_xyz_key = np.asarray([pc_xyz_key3[i] for i in self.joint_selection])
                    imu_save_l = data['imu_save_l']
                    orientation_imu_img = data['orientation_imu_img']
                    orientation_imu_img = np.asarray(orientation_imu_img)

                    t_R0R = data['t_R0R']
                    # t_RR0 = -1* (R_R0R.T @ t_R0R.T)
                    # H_pos = pc_xyz_key[20]
                    if st == 0:
                        R_ref = data['R_btc']
                        orientation_ref = orientation_imu_img

                        for (parent, child) in self.skeleton:
                            body = pc_xyz_key[parent] - pc_xyz_key[child]
                            body_length.append(body)
                        if self.initial_body is None:
                            self.initial_body = body_length
                            body_np = np.asarray(self.initial_body)
                            self.initial_body_unit = body_np / np.linalg.norm(body_np, axis=-1)[:, None].tolist()
                        else:
                            body_np = np.asarray(body_length)
                            body_norm = np.linalg.norm(body_np, axis=-1)
                            body_length = (body_norm[:, None] * np.asarray(self.initial_body_unit)).tolist()
                        st = 1

                    R_btc = data['R_btc']
                    R_R0R = self.R_ttb @ R_ref @ R_btc.T @ self.R_ttb.T
                    R_RtW = self.R_ttb @ R_btc @ self.R_ctw

                    R_NI = np.stack([imu_save_l[:, :3], imu_save_l[:, 3:6], imu_save_l[:, 6:9]], axis=2)
                    R_I0I = orientation_ref.T @ R_NI
                    imu_RR0 = self.R_RI @ R_I0I @ self.R_RI.T
                    imu_save_l[:, :3] = imu_RR0[:, 0, :]
                    imu_save_l[:, 3:6] = imu_RR0[:, 1, :]
                    imu_save_l[:, 6:9] = imu_RR0[:, 2, :]
                    imu_save_l[:, 11] = imu_save_l[:, 11] + 9.8

                    imu_save_l[:, 10:12] = -1 * imu_save_l[:, 10:12]
                    imu_save_l[:, 13:] = -1 * imu_save_l[:, 13:]

                    foot_contact_ = data['foot_contact']
                    foot_contact = [[0] * 2] * 2
                    foot_contact[0] = [0, 1] if foot_contact_[0, 0] else [1, 0]
                    foot_contact[1] = [0, 1] if foot_contact_[0, 1] else [1, 0]
                    abcd_ground_3 = data['abcd_ground_2']
                    if abcd_ground_3[0, 0] > 0:
                        abcd_ground_3 = -1 * abcd_ground_3

                    pc_r_ti = np.linalg.norm(pc_xyziv_ti3[:, 0:3], axis=1)
                    pc_xyzrvi = np.zeros((len(pc_xyziv_ti3), 6), dtype=np.float32)

                    pc_xyzrvi[:, 0:3] = pc_xyziv_ti3[:, :3]
                    pc_xyzrvi[:, 3] = pc_r_ti
                    pc_xyzrvi[:, 4:6] = pc_xyziv_ti3[:, 4:2:-1]

                    pc_frame_ti = np.zeros((self.pc_no, 6), dtype=np.float32)
                    pc_no_ti = pc_xyziv_ti3.shape[0]
                    if pc_no_ti == 0:
                        continue
                    if pc_no_ti < self.pc_no:
                        fill_list = np.random.choice(self.pc_no, size=pc_no_ti, replace=False)
                        fill_set = set(fill_list)
                        pc_frame_ti[fill_list] = pc_xyzrvi
                        dupl_list = [x for x in range(self.pc_no) if x not in fill_set]
                        dupl_pc = np.random.choice(pc_no_ti, size=len(dupl_list), replace=True)
                        # pc_frame_ti[dupl_list] = pc_xyzrvi[dupl_pc]
                        pc_frame_ti[dupl_list] = 0
                    else:
                        pc_list = np.random.choice(pc_no_ti, size=self.pc_no, replace=False)
                        pc_frame_ti = pc_xyzrvi[pc_list]

                    list_person_once_ti.append(pc_frame_ti)
                    list_person_once_key_xyz.append(pc_xyz_key)
                    list_person_once_imu.append(imu_save_l)
                    list_person_once_R_R0R.append(R_R0R)
                    list_person_once_t_R0R.append(t_R0R)
                    list_person_once_ground.append(abcd_ground_3)
                    list_person_once_foot_contact.append(foot_contact)
                    list_person_once_R_RtW.append(R_RtW)

                while len(list_person_once_ti) >= self.frame_no:
                    list_person_once_ti_25 = list_person_once_ti[-self.frame_no:]
                    list_person_once_key_xyz_25 = list_person_once_key_xyz[-self.frame_no:]
                    list_person_once_imu_25 = list_person_once_imu[-self.frame_no:]
                    list_person_once_ground_25 = list_person_once_ground[-self.frame_no:]
                    list_person_once_foot_contact_25 = list_person_once_foot_contact[-self.frame_no:]
                    list_person_once_R_R0R_25 = list_person_once_R_R0R[-self.frame_no:]
                    list_person_once_t_R0R_25 = list_person_once_t_R0R[-self.frame_no:]
                    list_person_once_R_RtW_25 = list_person_once_R_RtW[-self.frame_no:]

                    list_all_ti.append(list_person_once_ti_25)
                    list_all_key_xyz.append(list_person_once_key_xyz_25)
                    list_all_imu.append(list_person_once_imu_25)
                    list_all_R_R0R.append(list_person_once_R_R0R_25)
                    list_all_t_R0R.append(list_person_once_t_R0R_25)
                    list_all_skl.append(body_length)
                    list_all_ground.append(list_person_once_ground_25)
                    list_all_foot_contact.append(list_person_once_foot_contact_25)
                    list_all_R_RtW.append(list_person_once_R_RtW_25)

                    list_person_once_ti = list_person_once_ti[0:-self.frame_no]
                    list_person_once_key_xyz = list_person_once_key_xyz[0:-self.frame_no]
                    list_person_once_imu = list_person_once_imu[0:-self.frame_no]
                    list_person_once_R_R0R = list_person_once_R_R0R[0:-self.frame_no]
                    list_person_once_t_R0R = list_person_once_t_R0R[0:-self.frame_no]
                    list_person_once_ground = list_person_once_ground[0:-self.frame_no]
                    list_person_once_foot_contact = list_person_once_foot_contact[0:-self.frame_no]
                    list_person_once_R_RtW = list_person_once_R_RtW[0:-self.frame_no]

        self.body_length_all.append(body_length)

        print("data load end")
        list_all_ti = np.asarray(list_all_ti)
        list_all_key_xyz = np.asarray(list_all_key_xyz)
        list_all_skl = np.asarray(list_all_skl)
        list_all_imu = np.asarray(list_all_imu)
        list_all_R_R0R = np.asarray(list_all_R_R0R)
        list_all_t_R0R = np.asarray(list_all_t_R0R)
        list_all_ground = np.asarray(list_all_ground)
        list_all_foot_contact = np.asarray(list_all_foot_contact)
        list_all_R_RtW = np.asarray(list_all_R_RtW)

        return list_all_ti, list_all_key_xyz, list_all_imu, list_all_skl, list_all_ground, list_all_foot_contact, list_all_R_R0R, list_all_t_R0R, list_all_R_RtW


if __name__ == '__main__':
    vis_data = PosePC(train=False, vis=True, batch_length=50)
    B, L, N, D = vis_data.imu_.shape
    # accleration = vis_data.imu_[:, :, :, 9:12].reshape(B*L*N, 3)
    accleration = vis_data.imu_[:, :, :, 12:].reshape(B * L * N, 3)
    angular_velocity = vis_data.imu_[:, :, :, 9:12].reshape(B * L * N, 3)
    accleration_x = accleration[:, 0]
    H_pos = vis_data.data_key_[:, :, 20, :].reshape(B * L, 3)
    import matplotlib.pyplot as plt

    x1 = range(1, len(accleration) + 1)
    plt.figure(figsize=(20, 5))
    colors = ['#1f77b4', '#EE2C2C', '#FF8C00']
    # plt.title('Eval loss vs. frame', fontsize=20)
    for i in range(3):
        plt.plot(x1, accleration[:, i], '.-', color=colors[i], lw=6)
    # plt.plot(x1, accleration_x, '.-', color='b')

    # x1 = range(1, len(H_pos) + 1)
    # plt.figure(figsize=(15, 5))
    # plt.title('Eval loss vs. frame', fontsize=20)
    # plt.plot(x1, H_pos, '.-', label=['x', 'y', 'z'])

    # plt.legend()
    plt.axis('off')
    plt.show()
