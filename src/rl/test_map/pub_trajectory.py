import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# 回调函数，当接收到odom消息时调用
def odom_callback(msg):
    # 提取机器人的位置信息
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    # 创建一个Point消息，用于发布机器人的位置信息
    point_msg = Point()
    point_msg.x = x
    point_msg.y = y
    point_msg.z = 0.0  # 在二维平面上，z坐标设为0

    # 发布机器人的位置信息到自定义的话题上
    pub.publish(point_msg)

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('odom_listener')

    # 创建一个发布器，用于发布机器人的位置信息到自定义话题上
    pub = rospy.Publisher('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', Point, queue_size=10)

    # 创建一个订阅器，订阅odom话题，并指定回调函数
    rospy.Subscriber('odom', Odometry, odom_callback)

    # 进入ROS循环
    rospy.spin()