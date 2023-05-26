#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
import numpy as np
import torch
from torch import optim
from Env import environment
from net_Lin import *
import matplotlib.pyplot as plt
import csv
import cv2
import time
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class agent():
    def __init__(self, num_state, num_action, imitate_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor_net(num_state, num_action).to(self.device)
        self.actor.load_state_dict(torch.load(imitate_path))
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.path_x=[]
        self.path_y=[]
    def odom_callback(self, msg):
        # Extract the x and y position from the message
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        # Append the position to the path list
        self.path_x.append(self.pose_x)
        self.path_y.append(self.pose_y)

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
    pre_model_path = './model/imitate_model.pth'
    scan_num = 24
    num_action = 2
    num_state = scan_num + 1
    rospy.init_node('text_DRL', anonymous=True)
    rate = rospy.Rate(30)
    a = agent(num_state, num_action, pre_model_path)
    env = environment()
    pose_list = []
    finish_pose_list = []
    for i in range(1):
        # print('第', i, '次游戏')
        action_index = 0
        episode_step = 0
        while True:
            print('第', i, '次游戏')
            print('第', action_index, '个动作')
            action_index += 1
            Target, scan_, pose, finish_pose, heading, finish_distance = env.get_state()

            re_data = []
            for _ in range(scan_num):
                re_data.append(scan_[_ * (len(scan_) // scan_num)])

            state = torch.cat(
                (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
            action, _ = a.actor(state)
            # action, _ = a.actor.sample(state)
            action = a.tensor_to_numpy(action.squeeze())
            print('w=', action[0], 'v=', action[1])
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_heading, next_finish_distance = env.step(
                action, )

            # print(next_state.shape)
            # episode_step += 1
            # if episode_step == 200:
            #     env.get_bummper = True
            #     reward = -300
            # print('奖励', reward)

            if env.get_goalbox:
                env.chage_finish()
                break

            if env.get_bummper:
                env.init_word()
                break

            rate.sleep()
    plt.plot(a.path_x, a.path_y)
    plt.show()