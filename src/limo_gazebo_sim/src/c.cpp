// #include <ros/ros.h>                             // 引用 ros.h 檔
// #include<gazebo_msgs/ContactsState.h>
// void chatterCallback(const std_msgs::ContactsState::ConstPtr& msg)
// {
//   ROS_INFO("I heard: [%s]", msg->data.c_str());
// }
// int main(int argc, char** argv){
//     ros::init(argc, argv, "bumper");     // 初始化 hello_cpp_node
//     ros::NodeHandle n;                     // node 的 handler
//     ros::Subscriber sub = n.subscribe("/limo/bumper_states", 1000, chatterCallback);
//      ros::spin();
//      return 0
// }