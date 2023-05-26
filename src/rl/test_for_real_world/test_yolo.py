#!/usr/bin/env python
import rospy
import cv2
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np


class c():
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge = CvBridge()
        self.yolov7 = torch.hub.load('/home/a/yolov7', 'custom', '/home/a/yolov7/yolo_model/cap/weights/best.pt',
                                     source='local',
                                     force_reload=False)

    def sub(self):
        # self.sub_image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.get_image)
        self.sub_image = rospy.Subscriber('/camera/color/image_raw', Image, self.Binarization)
        # self.sub_image = rospy.Subscriber('/camera/color/image_raw', Image, self.yolo)

    def yolo(self, image):
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

    def Binarization(self, image):
        self.cv_im = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        mask = self.img_filter(self.cv_im)
        a = np.where(mask == 255)
        print(len(a[0]))
        cv2.imshow('mask', mask)
        cv2.waitKey(1)

    def bboxes_show(self, img, bbox, midpoint):
        cv2.circle(img, midpoint, 1, (0, 0, 255), 4)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 7)
        cv2.putText(img, bbox[6], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(bbox[4]), (int(bbox[2]), int(bbox[3]) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 1, cv2.LINE_AA)

    def img_filter(self, img):
        '''
        把影像处理成看到背景全部过滤掉只留下红色终点
        '''
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([160, 50, 0])
        upper_red1 = np.array([179, 255, 255])
        lower_red2 = np.array([0, 50, 0])
        upper_red2 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # cv2.imshow('img', mask)
        # cv2.waitKey(1)
        return mask


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    c = c()
    c.sub()
    rospy.spin()
