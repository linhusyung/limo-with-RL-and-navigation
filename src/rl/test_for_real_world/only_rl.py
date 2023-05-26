#!/usr/bin/env python
# license removed for brevity
import rospy
import cv2
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from net import Actor_net
from std_msgs.msg import String
import matplotlib.pyplot as plt


class c():
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge = CvBridge()
        # self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/my_data_tiny_new/best.pt',
        #                              source='local',
        #                              force_reload=False)#红色方块
        self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/yolo_model/cap/weights/best.pt',
                                     source='local',
                                     force_reload=False)  # 奖杯
        self.RL = True
        self.actor = Actor_net(25, 2).to(self.device)
        self.actor.load_state_dict(torch.load('./imitate_model.pth'))
        self.Target_in = -1
        self.distributed_x_ = []
        self.distributed_y_ = []
        self.distributed_tanhx_ = []
        min = np.linspace(0, 2.5, 50)
        max = np.linspace(-2.5, 0, 50)
        self.compensate_tabel = np.concatenate((min, max), axis=0)

    def sub_topic(self):
        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.get_scan)

        self.msg = Twist()
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_done = rospy.Subscriber('/chatter', String, self.get_done)

    def get_done(self, done):
        self.done = done.data

    def get_scan(self, scan):
        """
        scan咨询
        """
        self.re_data = []
        self.no_re_data = []
        for _ in range(24):
            self.re_data.append(scan.ranges[_ * (len(scan.ranges) // 24)] / 12)

        for _ in range(100):
            self.no_re_data.append(scan.ranges[_ * (len(scan.ranges) // 100)])

    def get_image(self, image):
        """
        摄影机咨询
        """
        self.x_f = -1
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        with torch.no_grad():
            results = self.yolov7(self.cv_im)
            bboxes = np.array(results.pandas().xyxy[0])

        for i in bboxes:
            if i[4] > 0.5:
                x_f = (i[2] - i[0]) / 2 + i[0]
                y_f = (i[3] - i[1]) / 2 + i[1]
                self.bboxes_show(self.cv_im, i, (int(x_f), int(y_f)))
                self.x_f = int(x_f * (100 / 640))

        cv2.imshow('img', self.cv_im)
        cv2.waitKey(1)
        if self.RL:
            action = self.sac()
            print(action)
            self.perform_action(action)

        if self.done == '1':
            print('f')
            self.perform_action([0, 0])
            self.RL = False
            self.stop__ = False
            # np.save('distributed_x_.npy', np.asanyarray(self.distributed_x_))
            # np.save('distributed_y_.npy', np.asanyarray(self.distributed_y_))
            # np.save('distributed_tanhx_.npy', np.asanyarray(self.distributed_tanhx_))

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 4)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)

    def sac(self):
        # if self.x_f != -1:
        #     self.Target_in = self.x_f
        # print('Target_in', self.Target_in)
        # self.x_f = 1
        state = torch.cat(
            (self.data_to_tensor(self.x_f).unsqueeze(0).unsqueeze(0),
             self.data_to_tensor(self.re_data).unsqueeze(0)),
            1)

        with torch.no_grad():
            action, _, mean, std = self.actor(state)
        action = self.tensor_to_numpy(action.squeeze())
        mean, std = mean[0][0].cpu().numpy(), std[0][0].cpu().numpy()
        # x = np.linspace(-100, 100, 1000)  # x轴范围
        # y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))  # 概率密度函数
        # tanh_x = np.tanh(0.3 * mean) * 2.5
        # self.distributed_x_.append(x)
        # self.distributed_y_.append(y)
        # self.distributed_tanhx_.append(tanh_x)

        # plt.axvline(x=tanh_x, color='r', linestyle='--', label='Mean')
        # plt.plot(x, y, label='Gaussian distribution')

        replacement_number = 12  # 替換的數字
        new_list = [x if x != 0 else replacement_number for x in self.no_re_data]
        compensate = 0
        if min(new_list) < 0.3:
            print("scan", new_list.index(min(new_list)))
            compensate = self.compensate_tabel[new_list.index(min(new_list))]

        if self.x_f != -1:
            # tanh_x = np.tanh(1 * mean) * 2.5
            tanh_x = np.tanh(0.1 * mean) * 2.5
            action[0] = tanh_x
            if compensate != 0:
                action[0] = compensate
                action[1] = 0.25
                return action
            else:
                return action
        else:
            if compensate != 0:
                action[0] = compensate
                action[1] = 0.25
                return action
            else:
                action[0] = 0
                action[1] = 0.25
                return action

    def tensor_to_numpy(self, data):
        return data.detach().cpu().numpy()

    def perform_action(self, action):
        '''
        连续动作空间
        v=[0~0.22]
        W=[-2.5~2.5]
        action=(w,v)
        '''
        self.msg.linear.x = float(action[1])
        self.msg.angular.z = float(action[0])
        self.pub.publish(self.msg)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)

    def main(self):
        self.sub_topic()


if __name__ == '__main__':
    rospy.init_node('rl_real_world')
    c = c()
    c.main()
    rospy.spin()
