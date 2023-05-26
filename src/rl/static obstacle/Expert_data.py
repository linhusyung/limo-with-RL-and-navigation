import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from Env import environment
from geometry_msgs.msg import Twist
import numpy as np


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


class sub():
    def __init__(self):
        self.x = 0
        self.z = 0

    def get_target_vel(self, data):
        self.x = data.linear.x
        self.z = data.angular.z


if __name__ == '__main__':
    rospy.init_node('movebase_client_py')
    env = environment()
    rate = rospy.Rate(30)
    sub = sub()
    b = 0
    for i in range(20):
        # movebase_client(env.finish_pose)
        print('第', i, '次')
        while True:
            Target, scan, _, _, _, _ = env.get_state()
            cmd_vel = rospy.Subscriber("cmd_vel", Twist, sub.get_target_vel)
            action = np.zeros((1, 2))
            action[0][0] = sub.x
            action[0][1] = sub.z
            _ = env.step(action[0], train=False)

            Target = np.array(Target)
            # heading = np.array(heading)
            # state = np.hstack((Target, scan, heading))

            if action[0][0] != 0 or action[0][1] != 0:
                print('储存动作')
                np.save('./Expert_data/action/action_' + str(b) + '.npy', action)
                np.save('./Expert_data/target/target_' + str(b) + '.npy', Target)
                # np.save('./Expert_data/heading/heading_' + str(b) + '.npy', heading)
                np.save('./Expert_data/scan/scan_' + str(b) + '.npy', scan)
                b += 1

            if env.get_goalbox:
                env.chage_finish()
                break

            if env.get_bummper:
                env.init_word()
                break
            rate.sleep()
