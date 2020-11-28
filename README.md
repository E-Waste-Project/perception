# realsense_ros_custom_driver
A simple and fast driver for using Intel Realsense D400 series cameras with ROS and OpenCV.

## Dependencies
```
ros kinetic
OpenCV >= 4
realsense2
```

## Installations
```shell
$ cd ~
$ git clone https://github.com/E-waste-Project/realsense-ros-custom-driver.git
$ cp -r realsense-ros-custom-driver/src/ ~/your_catkin_ws/src/
$ sudo rm -r realsense-ros-custom-driver
$ cd ~/your_catkin_ws
$ catkin build
```

## Sample Usage
If needed change parameter values in the stream_publisher.launch launch file in realsense_ros_custom_driver/launch folder.

First open a terminal and launch the stream_publisher.launch file:
```shell
$ roslaunch realsense_ros_custom_driver stream_publisher.launch
```
Then in another terminal you can run a subscriber node to the published image stream topic:
```shell
$ rosrun realsense_ros_custom_driver_test test_node
```


