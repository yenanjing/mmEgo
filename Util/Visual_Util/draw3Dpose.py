import os
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import matplotlib.animation as animation
#pose关键点连接矩阵
connectivity_dict=[[20, 3], [3, 2], [2, 1], [2, 4], [2, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
                          [1, 0], [0, 12], [0, 16], [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]]

#绘制单帧pose
def draw3Dpose(pose_3d,pose_3d2,ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, pose_3d3=None):  # blue, orange
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.grid(False)
    for i in connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(y, z, -x, lw=2, c=lcolor)
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(y2, z2, -x2, lw=2, c=rcolor)
        if pose_3d3 is not None:
            x3, y3, z3 = [np.array([pose_3d3[i[0], j], pose_3d3[i[1], j]]) for j in range(3)]
            ax.plot(y3, z3, -x3, lw=2, c='k')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = -pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(32, 32)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("x")
    # plt.show()

def draw3Dpose_1(pose_3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.grid(False)
    for i in connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(y, z, -x, lw=2, c=lcolor)
        # ax = fig.add_subplot(111, projection='3d')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = -pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(32, 32)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("x")
    # plt.show()

#绘制多帧pose
def draw3Dpose_frames(ti,show_s,data_key_in, idx, show_s_2=None):
    # 绘制连贯的骨架
    # show_s: 预测的骨架  data_key_in：真实骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    i = 0
    j=0
    image_list = []
    while i < show_s.shape[0]:
        if show_s_2 is not None:
            draw3Dpose(show_s[i], data_key_in[i], ax, pose_3d3=show_s_2[i])
        else:
            draw3Dpose(show_s[i], data_key_in[i], ax)
        ax.scatter(ti[i,:, 1], ti[i,:, 2], -ti[i,:, 0], c=['green'], s=15)
        if show_s_2 is not None:
            ax.legend(['pred','gt', 'pred_2'], loc=1, fontsize=15)
        else:
            ax.legend(['pred', 'gt'], loc=1, fontsize=15)
        # plt.pause(0.3)
        plt.savefig('./gif/temp.png')
        # plt.close()
        image_list.append(imageio.imread('./gif/temp.png'))
        imageio.mimsave('./gif/{}/Dis.gif'.format(idx), image_list, duration=0.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1

    plt.ioff()
    # plt.show()
    os.remove('../../gif/temp.png')
    # ani = animation.ArtistAnimation(fig, image_list, interval=200, repeat_delay=1000)
    # ani.save("./gif/pic.gif", writer='pillow')

def draw3Dpose_frames_1(ti,show_s,idx):
    # 绘制连贯的骨架
    # show_s: 预测的骨架  data_key_in：真实骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    i = 0
    j=0
    image_list = []
    while i < show_s.shape[0]:
        draw3Dpose_1(show_s[i], ax)
        ax.scatter(ti[i,:, 1], ti[i,:, 2], -ti[i,:, 0], c=['green'], s=15)
        # ax.legend(['pred', 'gt'], loc=1, fontsize=15)
        plt.pause(0.3)
        time.sleep(0.3)
        plt.savefig('./gif/temp.png')
        # plt.close()
        image_list.append(imageio.imread('./gif/temp.png'))
        imageio.mimsave('./gif/{}/Dis.gif'.format(idx), image_list, duration=0.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1

    plt.ioff()
    plt.close()
    # plt.show()
    os.remove('../../gif/temp.png')
    # ani = animation.ArtistAnimation(fig, image_list, interval=200, repeat_delay=1000)
    # ani.save("./gif/pic.gif", writer='pillow')

if __name__ == '__main__':
    np.random.seed(1234)
    data_ti_in=np.random.random((80,64,3))
    #print(data_ti_in)
    np.random.seed(22)
    show_s = np.random.random((80,24,3))
    np.random.seed(33)
    data_key_in = np.random.random((80,24,3))
    np.random.seed(12)
    show_s2 = np.random.random((80,24,3))
    draw3Dpose_frames(data_ti_in, show_s, data_key_in, 30, show_s2)