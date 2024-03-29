# Disassembly-Perception
Perception tasks needed for robots to disassemble laptops.
It has Deep learning models based on Yolov5 and ssd trained on [laptop components dataset](https://drive.google.com/drive/folders/1MSZEYLth2RzEBamaif_onqErxO3oXc81) that we collected.

Also It has image processing pipelines for planning cutting operations to detach components.


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
Step 2: Create 'models' folder.
```shell
$ cd your_catkin_ws/src/perception && mkdir models && cd models
```
Step 3: Download model from here https://drive.google.com/file/d/1oWRE2vGRF8ScWP111Tp1CPUFbzeSc5vF/view?usp=sharing to 'models' folder.

## Usage
To run any of the scripts, simply rosrun it.
```
$ rosrun perception script_name.py
```
Used for Robothon Grand challenge (Our Team RAND-E ranked 3rd World Wide):

[Demonstration Video](https://www.youtube.com/watch?v=4QgSiGciNaM&t)

Also Used for Autonomous Semi-Destructive Disassembly of Laptops:

[Demonstration Video](https://youtu.be/DrsZcyIvMZc)

If useful to you, please cite our paper:

```bibtex
@INPROCEEDINGS{9447637,  author={Bassiouny, Abdelrhman M. and Farhan, Abdelrahman S. and Maged, Shady A. and Awaad, Mohammed I.},
booktitle={2021 International Mobile, Intelligent, and Ubiquitous Computing Conference (MIUCC)},
title={Comparison of Different Computer Vision Approaches for E-waste Components Detection to Automate E-waste Disassembly},
year={2021}, pages={17-23},  doi={10.1109/MIUCC52538.2021.9447637}}
```
## Sample Results
![1](https://user-images.githubusercontent.com/36744004/198827769-0698e458-4faf-4493-b31a-4863189c5e25.jpg)
![11](https://user-images.githubusercontent.com/36744004/198827778-a8c3e502-47d9-47aa-9c96-4cc758a7b204.jpg)
![1_Color](https://user-images.githubusercontent.com/36744004/198827816-20911b40-4cf2-4ef1-930f-ddec82abbaff.png)
