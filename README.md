# Advanced RoboCup@Home - Restaurant Task

This is a readme file about our final project for advanced RoboCup@Home. In our task, we program TiaGo to finish the Restaurant task from Rule Book and TiaGO acts here as a server and serves customers with objects they ordered.

The video of demo is on Youtube and can be visited via the link https://studio.youtube.com/video/4dQ4f4bH9A4/edit

The code is also uploaded to Github via the link https://github.com/89kunkun/Advanced-Robocup.git

# Requirements
0. Ubuntu 20.04
1. ROS1 Noetic is downloaded and installed. The ROS Path is added to environment parameters.
2. Python >= 3.8
3. Clone or copy the project to the workspace
4. Copy all the TiaGo relevant packages to the workspace
5. Download the YOLOv8 relevant package with `pip install -U ultralytics`, the used weights are already saved under ~/Advanced-Robocup/yolo_v8_detector/weights/yolov8n-pose.pt and ~/Advanced-Robocup/yolo_v8_detector/weights/yolov8s-seg_70epoch.pt
**TODO:** Add the steps to install wit.ros

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
Before starting the state machine, the following nodes should be started in separate terminals.
```
roslaunch yolo_v8_detector yolo_v8.launch 
roslaunch wave_customer_detect detect_wave.launch 
roslaunch tiago_wave_customer_localizer waving_person_localizer.launch 
```
**TODO:** Add above the robot speaking and witros node

**Note:** For the grasp nodes, they are added and activated directly via State Machine with subprocesses. If you just want to test the grasp function separately, please refer to the [Grasp Pipeline](#grasp-pipeline) section below.

## Start the state machine
In the last step, you may start the state machine with following instruction
```
roslaunch state_machine party_main.launch
```

# Train weights based on YOLOv8 models
1. First label the objects on labelling websites. At the same time you also need to mark their bounding boxes or masks as your object detection needs.
2. After the final export, you will get label information in a txt file, which contains the label information of each picture including the labels in this picture, the position of the mask/bounding box.
3. The training workspace should have the following directory structure. Put the corresponding photos and label information txt files into these folders:

```
TrainYourYOLO/
├── data/
│   ├── labels/
│   │   ├── validate/
│   │   └── train/
│   └── images/
│       ├── validate/
│       └── train/
└── image-segmentation-yolov8-main/
    └── Here are the training python scripts and also I use Jupyter Notebooks here
```

You may import all these files and structures to Google Drive if you may use Google Colab to train the model. Thus you can easily run the training python file on your PC. The folder /image-segmentation-yolov8.main is also added to this project repository.

# Grasp Pipeline

This section describes the step-by-step process for object grasping:

## Pre Grasp
```bash
roslaunch grasp grasp_try.launch
```

## Open YOLO(v8)
```bash
roslaunch yolo_v8_detector yolo_v8.launch 
```

## Plane Segmentation
```bash
roslaunch plane_segmentation plane_segmentation_seg.launch
```

## Object Labeling
```bash
roslaunch object_labeling object_labeling_seg.launch
```

## Calculate PCA Axis and Run Grasp
```bash
roslaunch grasp grasp_cola.launch
# or
roslaunch grasp grasp_sprite.launch
# or
roslaunch grasp grasp_plate.launch
```