#!/usr/bin/env python
# license removed for brevity
import numpy as np
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from Env import environment
from geometry_msgs.msg import Twist
import cv2
import csv
from geometry_msgs.msg import PoseWithCovarianceStamped
import torch
from net_Lin import *
from actionlib_msgs.msg import GoalID

def movebase_client(finish_pose):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = finish_pose[0]
    goal.target_pose.pose.position.y = finish_pose[1]
    goal.target_pose.pose.orientation.z = 0
    goal.target_pose.pose.orientation.w = 1

    client.send_goal(goal)

    # client.wait_for_result()

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
class sub():
    def __init__(self):
        self.x = 0
        self.z = 0

    def get_target_vel(self, data):
        self.x = data.linear.x
        self.z = data.angular.z

def noisy_finish(finish):
    stddev = 0.1
    noise = tuple(np.random.normal(scale=stddev, size=2))
    noisy_tuple = tuple(map(sum, zip(finish, noise)))
    return noisy_tuple

if __name__ == '__main__':
    rospy.init_node('movebase_client_py')

    pre_model_path = './model/map_1/imitate_model.pth'
    scan_num = 24
    num_action = 2
    num_state = scan_num + 1
    a = agent(num_state, num_action, pre_model_path)

    env = environment()
    """
    给个位置
    """
    pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(1)
    pose_x = PoseWithCovarianceStamped()
    pose_x.pose.pose.position.x = env.position.x  # x坐标
    pose_x.pose.pose.position.y = env.position.y  # y坐标
    pose_x.pose.pose.orientation.w = env.orientation.w  # 四元数
    pose_x.pose.pose.orientation.z = env.orientation.z  # 四元数
    pose_pub.publish(pose_x)
    """
    停止
    """
    cancel_msg = GoalID()
    cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)

    rate = rospy.Rate(30)
    sub = sub()
    r_list=[]
    for i in range(10):
        r = 0
        episode_step = 0
        while True:
            Target, scan_, pose, finish_pose, heading, finish_distance = env.get_state()
            if Target==-1:
                print('导航包决策')
                noisy_tuple = noisy_finish(finish_pose)
                movebase_client(noisy_tuple)
                cmd_vel = rospy.Subscriber("cmd_vel", Twist, sub.get_target_vel)
                action = [sub.z, sub.x]
            else:
                print('神经网路决策')
                cancel_pub.publish(cancel_msg)
                re_data = []
                for _ in range(scan_num):
                    re_data.append(scan_[_ * (len(scan_) // scan_num)])
                state = torch.cat(
                    (a.data_to_tensor(Target).unsqueeze(0).unsqueeze(0), a.data_to_tensor(re_data).unsqueeze(0)), 1)
                action, _ = a.actor(state)
                action = a.tensor_to_numpy(action.squeeze())
            next_Target, next_scan_, next_pose, next_finish_pose, reward, done, next_heading, next_finish_distance = env.step(
                action, train=True)

            r += reward

            episode_step += 1
            if episode_step == 100:
                env.get_bummper = True
                reward = -50
            if env.get_goalbox:
                env.chage_finish()
                r_list.append(r)
                pose_x = PoseWithCovarianceStamped()
                rospy.sleep(1)
                pose_x.pose.pose.position.x = env.position.x  # x坐标
                pose_x.pose.pose.position.y = env.position.y  # y坐标
                pose_x.pose.pose.orientation.w = env.orientation.w  # 四元数
                pose_x.pose.pose.orientation.z = env.orientation.z  # 四元数
                pose_pub.publish(pose_x)
                # print(pose_x)
                break

            if env.get_bummper:
                env.init_word()
                r_list.append(r)
                # for _ in range(50):
                pose_x = PoseWithCovarianceStamped()
                rospy.sleep(1)
                pose_x.pose.pose.position.x = env.position.x  # x坐标
                pose_x.pose.pose.position.y = env.position.y  # y坐标
                pose_x.pose.pose.orientation.w = env.orientation.w  # 四元数
                pose_x.pose.pose.orientation.z = env.orientation.z  # 四元数
                pose_pub.publish(pose_x)
                # print(pose_x)
                break
            rate.sleep()
    print(r_list)
    print('平均',np.mean(r_list))
