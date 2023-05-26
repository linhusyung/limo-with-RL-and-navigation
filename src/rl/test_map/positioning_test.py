from geometry_msgs.msg import PoseWithCovarianceStamped
import rospy

rospy.init_node('movebase_client_py')

pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
rospy.sleep(1)
pose_x = PoseWithCovarianceStamped()
pose_x.pose.pose.position.x = -2.5  # x坐标
pose_x.pose.pose.position.y = 0.0  # y坐标
pose_x.pose.pose.orientation.w = 0  # 四元数
pose_pub.publish(pose_x)

rospy.spin()