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


class stop():
    def __init__(self):
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=100)
        self.cancel_msg = GoalID()
        self.cancel_=0
    def sub(self):
        rospy.Subscriber('/move_base/cancel', GoalID, self.cancel)

    def cancel(self, data):
        self.cancel_ = data.stamp


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    rate = rospy.Rate(30)
    s = stop()
    while True:
        s.sub()
        s.cancel_pub.publish(s.cancel_msg)
        print(s.cancel_)
        # while True:
        #
        #     if len(s.cancel_) != 1:
        #         break
        #     rate.sleep()
        rate.sleep()
