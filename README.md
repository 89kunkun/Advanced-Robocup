### Grasp Pipline
Pre Grasp
```bash
    roslaunch grasp grasp_try.launch
```

Open YOLO(v8)
```bash
    roslaunch yolo_v8_detector yolo_v8.launch 
```

Plane Segmentation
```bash
    roslaunch plane_segmentation plane_segmentation_seg.launch
```

Object Labeling
```bash
    roslaunch object_labeling object_labeling_seg.launch
```

Calculate PCA Axis and Run Grasp
```bash
    roslaunch grasp grasp_cola.launch/grasp_sprite.launch
```
#########################################################################
This is a readme file about our final project for advanced RoboCup@Home. In our task, we compute TiaGo to finish the Restaurant task from Rule Book and TiaGO acts here as a server and serves customers with objects they ordered.
The video of demo is on Youtube and can be visited via the link https://studio.youtube.com/video/4dQ4f4bH9A4/edit
The code is also uploaded to Github via the link https://github.com/89kunkun/Advanced-Robocup.git

# Requirements
0. Ubuntu 20.04
1. ROS1 Noetic is downloaded and installed. The ROS Path is added to environment parameters.
2. Python >= 3.8
3. Clone or copy the project to the workspace
4. Copy all the TiaGo relevant packages to the workspace
5. Download the YOLOv8 relevant package with `pip install -U ultralytics`, the used weights are already saved under ~/Advanced-Robocup/yolo_v8_detector/weights/yolov8n-pose.pt
## TODO: Add the steps to install wit.ros

# Compile the project
1. Compile the whole project with the following instructions
```
cd ~/Advanced-Robocup
catkin build
```
2. Source the working directory with
```
source devel/setup.bash
```

# Start the demo
## Activate the correct map on TiaGo PC
The scanned map of the lab is saved on the computer of TiaGo.
ssh to TiaGo computer with `ssh pal@192.168.1.200` with password "pal". And switch to the map with `rosservice call /pal_map_manager/change_map "input: '2025-12-26_115430'"`

## Start ROS Nodes
Before starting the state machine, the following nodes should be started in seperate terminals.
```
roslaunch yolo_v8_detector yolo_v8.launch 
roslaunch wave_customer_detect detect_wave.launch 
roslaunch tiago_wave_customer_localizer waving_person_localizer.launch 
```
## TODO: Add above the robot speaking and witros node

## Start the state machine
In the last step, you may start the state machine with following instruction
``
roslaunch state_machine party_main.launch
```