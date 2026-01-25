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
