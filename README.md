# Disassembly-Perception
Perception tasks needed for robots to disassemble laptops.

## Dependencies
```
Python >= 3
OpenCV >= 4
ros melodic
```

## Installation
Step 1: Download Repoistory to your catkin workspace and build it.
```shell
$ cd your_catkin_ws/src/
$ git clone https://github.com/E-Waste-Project/perception.git
$ cd .. && catkin build perception
```
Step 2: Create 'models' folder and download detection models inside it.
```shell
$ cd your_catkin_ws/src/perception && mkdir models
$ wget https://drive.google.com/file/d/1oWRE2vGRF8ScWP111Tp1CPUFbzeSc5vF/view?usp=sharing
```

## Usage
To run any of the scripts, simply rosrun it.
```
$ rosrun perception script_name.py
```
