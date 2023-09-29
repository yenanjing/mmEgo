r"""
    Config for paths, joint set, and parameters.
"""
import os

import numpy as np

_current_path = os.path.dirname(__file__)


class Config:
    Idx = 1  # 实验序号
    pb = 10  # 绘图去除前缀长度

    frame_no = 20
    pc_no = 128
    lower_pc_no = 64
    joint_num_all = 21
    joint_num_upper = 15
    joint_num_lower = 8
    num_action = 13
    IMU_used = True
    IMU_pretrained = False
    Upper_pretrained = False
    Lower_pretrained = False

    skeleton_all = np.asarray(
        [[20, 3], [3, 2], [2, 1], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
         [1, 0], [0, 12], [0, 16], [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]])
    skeleton_upper_body = np.asarray(
        [[20, 3], [3, 2], [2, 1], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
         [1, 0], [0, 12], [0, 16]])
    skeleton_lower_body = np.asarray([[12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]])

    kinect_upper_gragh = [(0, 12), (0, 13), (0, 1), (1, 2), (2, 3), (2, 4),
                          (2, 8), (3, 14), (4, 5), (5, 6), (6, 7), (8, 9),
                          (9, 10), (10, 11)]
    kinect_joint_selection = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    upper_joint_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20]

    lower_joint_map = [12, 13, 14, 15, 16, 17, 18, 19]

    hand_joint_map = [7, 6, 11, 10]  # 左手腕，左手肘，右手腕，右手肘

    model_IMU_path = os.path.join(_current_path,
                                  '../Resource/Pretrained_model/IMU_Net/epoch173_batch20frame20lr3e-05.pth')
    model_upper_path = os.path.join(_current_path,
                                    '../Resource/Pretrained_model/Upper_Net/epoch451_batch20frame20lr3e-05.pth')
    model_lower_path = os.path.join(_current_path,
                                    '../Resource/Pretrained_model/Lower_Net/epoch161_batch20frame20lr0.0003.pth')

    dataset_random_seed = 1
