#!/home/a/anaconda3/envs/torch/bin/python3
import rospy
from gazebo_msgs.msg import ContactsState


class listener():
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        rospy.set_param('/done',0)
        rospy.Subscriber("/limo/bumper_states", ContactsState, self.callback)
        rospy.Subscriber("/limo/bumper_states_1", ContactsState, self.callback_1)
        rospy.Subscriber("/limo/bumper_states_2", ContactsState, self.callback_2)
        rospy.Subscriber("/limo/bumper_states_3", ContactsState, self.callback_3)
        rospy.spin()

    def callback(self,data):
        if len(data.states)!=0:
            rospy.loginfo('bumper')
            rospy.set_param('/done',1)
    def callback_1(self,data):
        if len(data.states) != 0:
            rospy.loginfo('bumper_1')
            rospy.set_param('/done', 1)
    def callback_2(self,data):
        if len(data.states) != 0:
            rospy.loginfo('bumper_2')
            rospy.set_param('/done', 1)
    def callback_3(self,data):
        if len(data.states) != 0:
            rospy.loginfo('bumper_3')
            rospy.set_param('/done', 1)


if __name__ == '__main__':
    lis = listener()
