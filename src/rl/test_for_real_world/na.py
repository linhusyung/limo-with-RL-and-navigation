#!/usr/bin/env python
# license removed for brevity
import rospy
import actionlib
import cv2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalID
import time
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from net import Actor_net
from std_msgs.msg import String
import math


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


class c():
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_ = []
        self.cancel_ = 1
        self.bridge = CvBridge()
        # self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/my_data_tiny_new/best.pt',
        #                              source='local',
        #                              force_reload=False)#红色方块
        self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/yolo_model/cap/weights/best.pt',
                                     source='local',
                                     force_reload=False)  # 奖杯
        self.RL = False
        self.once = True
        self.stop__ = True

        self.actor = Actor_net(25, 2).to(self.device)
        self.actor.load_state_dict(torch.load('./imitate_model.pth'))
        self.Target_in = -1
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
        self.stop()

    def sub_topic(self):
        self.sub_goal = rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal)
        self.pub_stop = rospy.Subscriber('/move_base/cancel', GoalID, self.cancel)
        self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.get_scan)
        self.sub_acml_pose = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_pose)

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
        for _ in range(24):
            self.re_data.append(scan.ranges[_ * (len(scan.ranges) // 24)] / 12)

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
            if i[4] > 0.1:
                x_f = (i[2] - i[0]) / 2 + i[0]
                y_f = (i[3] - i[1]) / 2 + i[1]
                self.bboxes_show(self.cv_im, i, (int(x_f), int(y_f)))
                self.x_f = int(x_f * (100 / 640))

        cv2.imshow('img', self.cv_im)
        cv2.waitKey(1)
        dis = math.dist(self.amcl_pose, [5, -0.3])
        if dis <= 3 and self.stop__ == True:
            self.RL = True
        # print(dis)
        if self.once:
            self.navigation_point([5, -0.3])
            self.once = False
        if self.RL:
            self.stop()
            action = self.sac()
            print(action)
            self.perform_action(action)
        if self.done == '1':
            print('f')
            self.stop()
            self.perform_action([0, 0])
            self.RL = False
            self.stop__ = False
    def get_pose(self, data):
        self.amcl_pose = [data.pose.pose.position.x, data.pose.pose.position.y]

    def goal(self, data):
        self.goal_ = [data.goal.target_pose.pose.position.x, data.goal.target_pose.pose.position.y]

    def cancel(self, data):
        self.cancel_ = data.stamp

    def navigation_point(self, point: list):
        '''
        发布导航dian
        '''
        while True:
            movebase_client(point)
            if len(c.goal_) != 0:
                break
            self.rate.sleep()

    def stop(self):
        '''
        结束导航
        '''
        cancel_msg = GoalID()
        while True:
            self.cancel_pub.publish(cancel_msg)
            if c.cancel_ != 1:
                break
            self.rate.sleep()

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 4)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)

    def sac(self):
        state = torch.cat(
            (self.data_to_tensor(self.x_f).unsqueeze(0).unsqueeze(0),
             self.data_to_tensor(self.re_data).unsqueeze(0)),
            1)
        with torch.no_grad():
            action, _, mean, std = self.actor(state)
        action = self.tensor_to_numpy(action.squeeze())
        mean, std = mean[0][0].cpu().numpy(), std[0][0].cpu().numpy()

        if self.x_f != -1:
            tanh_x = np.tanh(0.25 * mean) * 2.5
            action[0] = tanh_x
            return action
        else:
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
    rospy.init_node('movebase_client_py')
    c = c()
    c.main()
    rospy.spin()
