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
6. Install speech interaction dependencies: install Python dependency with `pip install requests wit`; install Audio recording tools with `sudo apt-get install alsa-utils`; install PulseAudio tools with `sudo apt-get install pulseaudio`


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
rosrun say_something say.py
roslaunch wit_ros audio_recognizer.launch
rosrun grasp plate_overhang_node.py
```
## Add Your Wit.ai API Key

Create the file `wit_ros/param/api.yaml` and insert:

```yaml
api_key: "YOUR_WIT_AI_API_KEY"
```

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

# Acknowledgments and Third-Party Components

This project incorporates code and components from the following open-source projects:

## YOLOv8 Image Segmentation
- **Source:** [image-segmentation-yolov8](https://github.com/computervisioneng/image-segmentation-yolov8.git)
- **Usage:** Training scripts and Jupyter Notebooks for YOLOv8 model training
- **Location:** `image-segmentation-yolov8-main/` directory
- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **⚠️ Important:** This component requires that any derivative work (including this project) be licensed under AGPL-3.0

## Ultralytics YOLOv8
- **Source:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Usage:** Object detection and segmentation models
- **Installation:** `pip install -U ultralytics`
- **License:** AGPL-3.0

## Wit.ai Speech API
- **Source:** https://wit.ai
- **Usage:** Cloud-based speech recognition for order understanding

## ALSA (arecord)
- **Source:** https://www.alsa-project.org/
- **Usage:** Audio recording on Linux systems

## PulseAudio
- **Source:** https://www.freedesktop.org/wiki/Software/PulseAudio/
- **Usage:** Audio streaming between robot and development machine


# License

**⚠️ IMPORTANT LICENSE NOTICE:**

This project incorporates components licensed under GNU Affero General Public License v3.0 (AGPL-3.0), which requires that:

1. **The entire project must be licensed under AGPL-3.0**
2. **Complete source code must be made available** to anyone who receives the software
3. **If you provide this software as a service over a network** (e.g., web service), you must make the complete source code available to users
4. **All copyright and license notices must be preserved**

**This project is therefore licensed under AGPL-3.0** - see the [LICENSE](../LICENSE) file for details.

**Note:** Due to AGPL-3.0 requirements, if you use this project in any commercial or network-based service, you must make the complete source code available to your users.
