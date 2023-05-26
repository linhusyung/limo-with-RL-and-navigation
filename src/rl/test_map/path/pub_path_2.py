import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# odom_x = np.load('./odom_x.npy')
# odom_y = np.load('./odom_y.npy')
odom_x = np.load('./odom_x_no.npy')
odom_y = np.load('./odom_y_no.npy')

# rospy.init_node('path_publisher_1')
rospy.init_node('path_publisher_2')
# 创建路径消息
path_msg = Path()
path_msg.header.frame_id = 'map'  # 设置路径坐标系为地图坐标系

# 填充路径消息的数据
for i in range(len(odom_x)):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = odom_x[i]
    pose.pose.position.y = odom_y[i]
    path_msg.poses.append(pose)

# 创建路径发布者
path_pub = rospy.Publisher('b', Path, queue_size=10)

# 发布路径消息
rate = rospy.Rate(10)  # 设置发布频率为10Hz
while not rospy.is_shutdown():
    path_msg.header.stamp = rospy.Time.now()  # 更新路径消息的时间戳
    path_pub.publish(path_msg)
    rate.sleep()
