# Disassembly-Perception
Perception tasks needed for robots to disassemble laptops.

## Dependencies
```
Python >= 3
OpenCV >= 4
ros melodic
```

## Installation
```shell
$ cd ~
$ git clone https://github.com/E-Waste-Project/Disassembly-Perception.git
$ cp -r Disassembly-Perception/src/ your_catkin_ws/src/
$ sudo rm -r Disassembly-Perception
$ cd your_catkin_ws && catkin build perception
```

## Usage
To run any of the scripts, simply rosrun it.
```
$ rosrun perception script_name.py
```
