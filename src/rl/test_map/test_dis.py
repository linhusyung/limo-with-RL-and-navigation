#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
import torch
from torch import optim
from Env import environment
from test_net import *
import matplotlib.pyplot as plt
import csv
import cv2
import time


class agent():
    def __init__(self, num_state, num_action, imitate_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(imitate_path))

    def get_state(self, scan_, taget) -> torch.tensor:
        return torch.cat((self.data_to_tensor(scan_), self.data_to_tensor(taget)), -1)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def image_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 1, 0))).float().to(self.device)

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

    def np_to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)


if __name__ == '__main__':
    # pre_model_path = './model/map_2/sac_24_2.pth'
    pre_model_path = './model/map_1/imitate_model.pth'
    scan_num = 24
    num_action = 2
    num_state = scan_num + 1
    rospy.init_node('text_DRL', anonymous=True)
    rate = rospy.Rate(30)
    a = agent(num_state, num_action, pre_model_path)
    env = environment()
    odom_x = []
    odom_y = []

    r_list = []
    for i in range(3):
        # print('第', i, '次游戏')
        action_index = 0
        episode_step = 0
        r = 0
        while True:
            print('第', i, '次游戏')
            print('第', action_index, '个动作')
            action_index += 1
            Target, scan_, pose, finish_pose, heading, finish_distance = env.get_state()
            odom_x.append(env.position.x)
            odom_y.append(env.position.y)
            print(type(odom_x))
            re_data = []
            for _ in range(scan_num):
                re_data.append(scan_[_ * (len(scan_) // scan_num)])

            state = torch.cat(
                (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            with torch.no_grad():
                mean, std, action, log_std = a.actor(state)
            print(mean.shape)
            mean, std = mean[0][0].cpu().numpy(), std[0][0].cpu().numpy()
            print('mean', mean, 'sta', std)

            x = np.linspace(-100, 100, 1000)  # x轴范围
            y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))  # 概率密度函数

            tanh_x = np.tanh(0.1 * mean) * 2.5
            # tanh_x = np.tanh(mean) * 2.5
            # plt.axvline(x=tanh_x, color='r', linestyle='--', label='Mean')
            # plt.plot(x, y, label='Gaussian distribution')

            action = a.tensor_to_numpy(action.squeeze())
            if Target != -1:
                action[0] = tanh_x
            else:
                pass
            print('w=', action[0], 'v=', action[1])
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_heading, next_finish_distance = env.step(
                action, train=True)

            # print(next_state.shape)
            episode_step += 1
            if episode_step == 200:
                env.get_bummper = True
                reward = -50
            r += reward

            if env.get_goalbox:
                env.chage_finish()
                r_list.append(r)
                break

            if env.get_bummper:
                env.init_word()
                r_list.append(r)
                break

            rate.sleep()
    # 坐标点

    # plt.plot(odom_x, odom_y)
    # plt.show()
    odom_x = np.array(odom_x)
    np.save('./path/odom_x.npy', odom_x)
    # np.save('./path/odom_x_no.npy', odom_x)
    odom_y = np.array(odom_y)
    np.save('./path/odom_y.npy', odom_y)
    # np.save('./path/odom_y_no.npy', odom_y)
    # plt.xlim(-100, 100)
    # plt.xlim(-2.5, 2.5)
    # plt.show()
    print(r_list)
    print('平均', np.mean(r_list))
