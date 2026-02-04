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

map is saved on the computer of TiaGo.
ssh to TiaGo computer with `ssh pal@192.168.1.200` with password "pal".
and switch to the map with `rosservice call /pal_map_manager/change_map "input: '2025-12-26_115430'"`

Before starting the state machine, the following nodes should be started in seperate terminals.
```
roslaunch yolo_v8_detector yolo_v8.launch 
roslaunch wave_customer_detect detect_wave.launch 
roslaunch tiago_wave_customer_localizer waving_person_localizer.launch 
```

Start Yolo
roslaunch yolo_v8_detector yolo_v8.launch 

Wave hand detection
roslaunch wave_customer_detect detect_wave.launch 
roslaunch tiago_wave_customer_localizer waving_person_localizer.launch 

State machine
roslaunch state_machine party_main.launch