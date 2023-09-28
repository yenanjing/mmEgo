import torch
import numpy as np
from torch.utils.data import DataLoader

from lowerNet import LowerNet
from Util.Universal_Util.Dataset import PosePC
from tqdm import tqdm
import draw3Dpose
import os
from Net.Upper_Net import UpperNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if torch.cuda.is_available():
    device = 'cuda:%d' % 0
else:
    device = 'cpu'
print(device)
frame_no = 30
batchsize = 13  #13个动作，每个动作取30帧
joint_num = 21
Lower_net=LowerNet().to(device)
Upper_net=UpperNet().to(device)
Upper_net.load('./model/14/299.pth')
Upper_net.eval()
Lower_net.load('./model/15/299.pth')
Lower_net.eval()
upper_joint_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20]
lower_joint_map = [12, 13, 14, 15, 16, 17, 18, 19]
pb = 10
Idx = 16
# skeleton = np.asarray([[0, 1], [0, 3], [0, 5], [0, 7], [1, 2], [3, 4], [5, 6], [7, 8]])
skeleton = np.asarray([[20, 3], [3, 2], [2, 1], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                          [1, 0], [0, 12], [0, 16], [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]])
root_kp = skeleton[:, 0]
leaf_kp = skeleton[:, 1]
root_kp = torch.tensor(root_kp, dtype=torch.long, device=device)
leaf_kp = torch.tensor(leaf_kp, dtype=torch.long, device=device)
targetPath = './gif/%d' % (Idx)
if not os.path.exists(targetPath):
    os.makedirs(targetPath)
else:
    print('路径已经存在！')

def angle_loss(pred_ske, true_ske):
    pred_vec = pred_ske[:, :, leaf_kp, :] - pred_ske[:, :, root_kp, :]
    true_vec = true_ske[:, :, leaf_kp, :] - true_ske[:, :, root_kp, :]
    cos_sim = torch.nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
    angle = torch.mean(torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0))
    return angle

vis_data=PosePC(train=False, vis=True, batch_length=frame_no)
vis_loader=DataLoader(vis_data, batch_size=batchsize, shuffle=False)

with torch.no_grad():
    for batch_idx, (data, target, skl, imu, init_head, ground, foot_contact) in tqdm(enumerate(vis_loader)):
        data = np.asarray(data)
        target = np.asarray(target)
        imu = np.asarray(imu)
        skl = np.asarray(skl)
        init_head = np.asarray(init_head)
        ground = np.asarray(ground)
        foot_contact = np.asarray(foot_contact)
        batch_size, seq_len, point_num, dim = data.shape

        data_ti = torch.tensor(data, dtype=torch.float32, device=device)
        target_lower = target[:, :, lower_joint_map, :]
        target_lower = torch.tensor(target_lower, dtype=torch.float32, device=device)
        target = torch.tensor(target, dtype=torch.float32, device=device)
        ground = torch.tensor(ground, dtype=torch.float32, device=device).squeeze()
        foot_contact = torch.tensor(foot_contact, dtype=torch.float32, device=device).squeeze()
        init_head = torch.tensor(init_head, dtype=torch.float32, device=device)
        init_head = init_head[0]

        h0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
        c0_g = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
        h0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
        c0_a = torch.zeros((6, batch_size, 64), dtype=torch.float32, device=device)
        initial_body = torch.tensor(skl, dtype=torch.float32, device=device)
        upper, _, pred_ground, _, _, _, _, _, _ = Upper_net(data_ti, h0_g, c0_g, h0_a, c0_a, initial_body)
        upper_l = upper.detach()
        lower_l, upper_q, contact, _, _, _, _, _, _ = Lower_net(upper_l, data_ti, h0_g, c0_g, h0_a, c0_a,
                                                                initial_body)
        pred_l = torch.zeros((batch_size, seq_len, 21, 3), dtype=torch.float32, device=device)
        pred_l[:, :, upper_joint_map, :] = upper_l
        pred_l[:, :, lower_joint_map, :] = lower_l
        data = data_ti.view(batch_size * seq_len, point_num, dim).cpu().numpy()
        data_key = target.view(batch_size * seq_len, joint_num, 3).cpu().numpy()
        show_s = pred_l.view(batch_size * seq_len, joint_num, 3).cpu().numpy()
        # show_s_2=torso_l.view(batch_size * seq_len, joint_num, 3).cpu().numpy()
        draw3Dpose.draw3Dpose_frames(data, show_s, data_key, Idx)
