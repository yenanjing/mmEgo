import matplotlib.pyplot as plt
import matplotlib.cm
import os
import numpy as np
import itertools
import math
import torch
import seaborn as sns
from Config.config import Config

_current_path = os.path.dirname(__file__)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): 当验证集性能不再提升时，等待的轮次数量。
            delta (float): 验证集性能提升的阈值。小于delta的性能改善将被认为是无效的。
            verbose (bool): 是否打印出每次早停时的信息。
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): 当前验证集的损失。
        Returns:
            early_stop (bool): 如果达到早停条件，返回True；否则返回False。
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss increased [{self.counter}/{self.patience}]')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop


def plot_confusion(matrix):
    targetPath = './svg/%d' % (Config.Idx)
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)
    a = ['({})'.format(i) for i in range(1, Config.num_action + 1)]
    sns.set()
    f, ax = plt.subplots(figsize=(10, 6))
    print(matrix)  # 打印出来看看
    sns.set_style({"font.sans-serif": "Times New Roman"})
    sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax, fmt='g', xticklabels=a, yticklabels=a,
                annot_kws={'size': 15, 'weight': 'bold'}, cbar=False)  # 画热力图
    # ax.set_title('Confusion Matrix') #标题xt
    ax.set_xlabel('True Action', fontproperties="Times New Roman", fontsize=25, weight='bold')  # x轴 15
    ax.set_ylabel('Predicted Action', fontproperties="Times New Roman", fontsize=25, weight='bold')  # y轴 15
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontproperties="Times New Roman", fontsize=12,
                       weight='bold')  # 9
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontproperties="Times New Roman", fontsize=12,
                       weight='bold')  # 9
    # plt.savefig('./actionRecog.pdf',bbox_inches='tight')
    plt.savefig("./svg/{}/{}.svg".format(Config.Idx, 'action_rec'))

    plt.show()


def project_skeleton_to_ground(skeleton_coords, light_pos, ground_height):
    # 计算光线方向向量
    # light_dir = light_pos - skeleton_coords
    #
    # # 计算射线和平面相交点
    # n = np.array([0, 0, 1])
    # P0 = skeleton_coords
    # d = light_dir
    # h = ground_height
    # t = (h - P0.dot(n)) / d.dot(n)
    # P = P0 + t * d

    # 用于保存每个骨架点的投影位置
    projection_coords = []

    # 遍历每个骨架点
    for i in range(len(skeleton_coords)):
        # 计算该点和光源坐标之间的向量
        vec = light_pos - skeleton_coords[i]

        # 计算光线方程
        def line_equation(t):
            return skeleton_coords[i] + t * vec

        # 计算光线和地平面的交点
        t = (ground_height - skeleton_coords[i, 2]) / vec[2]
        projection_coords.append(line_equation(t))
    # 返回投影点的x和y坐标
    projection_coords = np.asarray(projection_coords)
    return projection_coords[:, :2]


# 绘制单帧上半身pose
def draw3Dupper_pose(pose_3d, ax, floor):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    # 计算骨架投影到地面上的位置
    floor_level = -1 * floor
    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    # light_pos = [0, -5, -12]
    # project_ske_2d = project_skeleton_to_ground(pose_3d, light_pos, floor_level)
    # joints_2d = np.concatenate([project_ske_2d, np.full((21, 1), floor_level)], axis=-1)
    # joints_2d = np.concatenate([pose_3d[:, :2], np.full((21, 1), floor_level)], axis=-1)

    # 绘制骨架投影
    # for connection in config.skeleton_all:
    #     joint1 = joints_2d[connection[0]]
    #     joint2 = joints_2d[connection[1]]
    #     ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], 'k--')
    # 绘制地面
    x = np.arange(-RADIUS + xroot, RADIUS + xroot, 0.1)
    y = np.arange(-RADIUS + yroot, RADIUS + yroot, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, floor_level)
    # 绘制平面
    # ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), color='gray', shade=True, alpha=0.8)
    # ax.plot_surface(X, Y, Z, rcount=1, ccount=1, color='grey', shade=False, alpha=0.4, zorder=1)

    for i in Config.skeleton_upper_body:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=6, c='black', zorder=2)

    pose_3d = pose_3d[Config.upper_joint_map, :]
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='green', s=60, marker='o', zorder=3, alpha=1.0)

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.view_init(elev=16, azim=-107)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


# 绘制单帧pose
def draw3Dpose(pose_3d, ax, floor):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    pose_3d[:, 1] -= 0.2
    # 计算骨架投影到地面上的位置
    floor_level = -1 * floor
    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    # light_pos = [0, -5, -12]
    # project_ske_2d = project_skeleton_to_ground(pose_3d, light_pos, floor_level)
    # joints_2d = np.concatenate([project_ske_2d, np.full((21, 1), floor_level)], axis=-1)
    # joints_2d = np.concatenate([pose_3d[:, :2], np.full((21, 1), floor_level)], axis=-1)

    # 绘制骨架投影
    # for connection in config.skeleton_all:
    #     joint1 = joints_2d[connection[0]]
    #     joint2 = joints_2d[connection[1]]
    #     ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], 'k--')
    # 绘制地面
    x = np.arange(-RADIUS + xroot, RADIUS + xroot, 0.1)
    y = np.arange(-RADIUS + yroot, RADIUS + yroot, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, floor_level)
    # 绘制平面
    # ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), color='gray', shade=True, alpha=0.8)

    # ax.plot_surface(X, Y, Z, rcount=1, ccount=1, color='grey', shade=False, alpha=0.4, zorder=1)  #地平面

    for i in Config.skeleton_all:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=6, c='black', zorder=2)

    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='green', s=60, marker='o', zorder=3, alpha=1.0)

    ax.set_xlim3d([-RADIUS + xroot + 0.5, RADIUS + xroot - 0.5])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot - 0.2])
    ax.set_ylim3d([-RADIUS + yroot + 0.5, RADIUS + yroot - 0.5])
    ax.view_init(elev=16, azim=-107)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# 绘制多帧pose
def draw3Dpose_frames(pred, real, index, floor):
    # 分别绘制预测骨架和真实骨架
    targetPath = './svg/%d' % (Config.Idx)
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)
    fig1 = plt.figure(1, figsize=(5, 5))
    fig2 = plt.figure(2, figsize=(5, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_box_aspect([1.5, 1.5, 1.5])  # 设置缩放比例
    # ax1.set_title('Predicted Skeleton')
    # ax1.axis('off')
    ax1.axis('on')
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_box_aspect([1.5, 1.5, 1.5])  # 设置缩放比例
    # ax2.set_title('Real Skeleton')
    # ax2.axis('off')
    ax2.axis('on')
    i = 0
    while i < pred.shape[0]:
        ax1 = fig1.add_subplot(111, projection='3d')
        # ax1.set_title('Predicted Skeleton')
        # ax1.axis('off')
        ax1.axis('on')
        ax1.set_box_aspect([1.5, 1.5, 1.5])  # 设置缩放比例
        ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.set_title('Real Skeleton')
        # ax2.axis('off')
        ax2.axis('on')
        ax2.set_box_aspect([1.5, 1.5, 1.5])  # 设置缩放比例
        draw3Dpose(pred[i], ax1, floor[i])
        draw3Dpose(real[i], ax2, floor[i])
        # draw3Dupper_pose(pred[i], ax1, floor[i])
        # draw3Dupper_pose(real[i], ax2, floor[i])
        # fig1.show()
        # fig2.show()
        # plt.pause(3)
        # plt.ion()
        targetPath = './svg/%d/%d' % (Config.Idx, i + index)
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)

        fig1.savefig('./svg/{}/{}/pred_frame_{}.svg'.format(Config.Idx, i + index, i + index))
        fig2.savefig('./svg/{}/{}/real_frame_{}.svg'.format(Config.Idx, i + index, i + index))
        # plt.close()
        # print(ax.lines)
        plt.clf()
        # ax.lines = []
        i += 1
    plt.close(1)
    plt.close(2)


def radian_to_degree(q):
    r"""
    Convert radians to degrees.
    """
    return q * 180.0 / np.pi


def degree_to_radian(q):
    r"""
    Convert degrees to radians.
    """
    return q / 180.0 * np.pi


def angle_minus(m1, m2):
    eps = 1e-7
    m1 = m1.view(-1, 3, 3)
    m2 = m2.view(-1, 3, 3)
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))
    return theta / 3.14159265358 * 180


def Transform2R(points, batch_size, length_size, pc_no, R, t):
    points = points.view(batch_size * length_size, pc_no, -1, 1)
    R_l = R.view(batch_size * length_size, 1, 3, 3).repeat(1, pc_no, 1, 1).contiguous()
    t_l = t.view(batch_size * length_size, 1, 3, 1).repeat(1, pc_no, 1, 1).contiguous()
    if Config.IMU_used:
        points = torch.matmul(R_l.permute(0, 1, 3, 2), points) + t_l
    points = points.view(batch_size, length_size, pc_no, -1)
    return points


def Transform2H(points, batch_size, length_size, pc_no, R, t):
    points = points.view(batch_size * length_size, pc_no, -1, 1)
    R_r = R.view(batch_size * length_size, 1, 3, 3).repeat(1, pc_no, 1, 1).contiguous()
    t_r = t.view(batch_size * length_size, 1, 3, 1).repeat(1, pc_no, 1, 1).contiguous()
    xyz = points[:, :, :3, :].contiguous()
    if Config.IMU_used:
        points[:, :, :3, :] = torch.matmul(R_r, xyz - t_r)
    points = points.view(batch_size * length_size, pc_no, -1)
    return points


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def draw_loss_frame(loss_angle, loss_H_pos, name, idx=Config.Idx):
    x1 = range(1, len(loss_angle) + 1)
    plt.figure(figsize=(15, 5))
    plt.title('Eval loss vs. frame', fontsize=20)
    plt.plot(x1, loss_angle, '.-', label='angle')
    plt.plot(x1, loss_H_pos, '.-', label='H_pos')
    plt.xlabel('Frame', fontsize=15)
    plt.ylabel('Eval loss', fontsize=15)
    plt.grid()
    plt.legend(loc=0, prop={'size': 20})
    plt.savefig("./lossAndacc/{}/{}.svg".format(idx, name))
    plt.show()


def draw_fig(lis, name, epoch, begin, idx=Config.Idx, lis_1=None):
    x1 = range(1 + begin, epoch + 1)
    y1 = lis[begin:]
    if lis_1 is not None:
        y2 = lis_1[begin:]
    if name == "loss":
        plt.cla()
        plt.title('Eval loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('Eval loss', fontsize=15)
        plt.grid()
        plt.savefig(os.path.join(_current_path, "../../Processor/Train/lossAndacc/{}/Eval_loss.png".format(idx)))
        plt.show()
    elif name == "acc":
        plt.cla()
        plt.title('Average Joint Localization Error vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('Average Joint Localization Error (m)', fontsize=15)
        plt.grid()
        plt.savefig(os.path.join(_current_path, "../../Processor/Train/lossAndacc/{}/Eval_accuracy.png".format(idx)))
        plt.show()


def draw_bar(l, idx, name):
    skeleton_all = ['20—3', '3—2', '2—1', '2—4', '2—8', '4—5', '5—6', '6—7', '8—9', '9—10', '10—11',
                    '1—0', '0—12', '0—16', '12—13', '13—14', '14—15', '16—17', '17—18', '18—19']
    if name == 'pos':
        plt.cla()
        plt.xlabel('Joint Index', fontsize=15)
        plt.ylabel('Average Localization Error (m)', fontsize=15)
        plt.bar(range(0, len(l), 1), l)
        plt.xticks(range(0, len(l), 1))
        plt.savefig(os.path.join(_current_path, "../../Processor/Train/lossAndacc/{}/Eval_joint_accuracy.png".format(idx)))

        plt.show()
    elif name == 'angle':
        # plt.cla()
        # 设置figsize的大小
        plt.figure(figsize=(15, 5))
        plt.xlabel('Connectivity', fontsize=15)
        plt.ylabel('Rotation Error (°)', fontsize=15)
        plt.bar(range(0, len(l), 1), l, width=0.6, tick_label=skeleton_all)
        # plt.xticks(skeleton_all)
        plt.savefig(os.path.join(_current_path, "../../Processor/Train/lossAndacc/{}/Eval_joint_angle.png".format(idx)))
        plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, title, idx, normalize=True, cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.cla()
    # plt.figure(figsize=(9, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto', )

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label', fontsize=15)
    plt.xlabel('True label', fontsize=15)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(9, 6)
    plt.savefig("./lossAndacc/{}/Eval_{}foot_confmatrix.png".format(idx, title))
    plt.show()


if __name__ == '__main__':
    res = np.array([[1., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 1., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0.97, 0, 0.03, 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 1., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 1., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 1.,
                     0., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     1., 0., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     0., 1., 0., 0., 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 1., 0., 0., 0.,
                     0., ],
                    [0.01, 0., 0., 0., 0., 0.,
                     0., 0., 0., 0.99, 0., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 1., 0.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 1.,
                     0., ],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     1.]])
    plot_confusion(res)
