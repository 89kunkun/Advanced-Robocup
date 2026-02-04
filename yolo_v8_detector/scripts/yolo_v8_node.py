#!/usr/bin/env python3

import rospy
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolo_v8_detector.msg import BoundingBoxes, BoundingBox, Skeletons, Skeleton, Keypoint
from std_msgs.msg import Bool


class YOLOv8Detector:
    def __init__(self):
        rospy.init_node("yolo_v8_node")

        # Load params
        self.image_topic = rospy.get_param("~image_topic", "")
        self.weights_path = rospy.get_param("~weights_path", "")
        self.publish_debug = rospy.get_param("~publish_debug_image", True)
        self.skeleton_weights_path = rospy.get_param("~skeleton_weights_path", "")
        rospy.loginfo(f"[YOLOv8] Loading model: {self.weights_path}")

        # Load YOLOv8 model (correct API)
        self.model = YOLO(self.weights_path)
        
        # Load skeleton model only if path is provided
        self.skeleton_model = None
        if self.skeleton_weights_path:
            rospy.loginfo(f"[YOLOv8] Loading skeleton model: {self.skeleton_weights_path}")
            self.skeleton_model = YOLO(self.skeleton_weights_path)
        else:
            rospy.loginfo("[YOLOv8] Skeleton detection disabled (no weights path provided)")

        self.bridge = CvBridge()
        # Flag to control whether detection is running; initially False
        self.active = False
        # Listen to /startyolo and /stopyolo topics to control activation
        rospy.Subscriber("/startyolo", Bool, self.start_callback)
        rospy.Subscriber("/stopyolo", Bool, self.stop_callback)
        # Subscribe to image topic
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback, queue_size=1)

        self.pub_boxes = rospy.Publisher(
            "yolo_v8_detector/bounding_boxes", 
            BoundingBoxes, 
            queue_size=1
        )
        self.pub_skeletons = rospy.Publisher(
            "/yolo_v8_detector/skeletons",
            Skeletons,
            queue_size=1
        )

        if self.publish_debug:
            self.pub_debug = rospy.Publisher(
                "/yolo_v8_detector/debug_image",
                Image,
                queue_size=1
            )

        rospy.loginfo("[YOLOv8] Node started. Waiting for /startyolo signal... Detection is initially off.")

    def start_callback(self, msg):
        if msg.data and not self.active:
            self.active = True
            rospy.loginfo("YOLOv8 detection started.")

    def stop_callback(self, msg):
        if msg.data and self.active:
            self.active = False
            rospy.loginfo("YOLOv8 detection stopped.")


    def callback(self, msg):
        if not self.active:
            # Detection is not active, do nothing
            return
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Run YOLOv8 object detection
        results = self.model(frame, verbose=False)

        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header

        # Copy frame for annotation
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls)
                conf = float(box.conf)

                # Fill BoundingBox message
                bb = BoundingBox()
                bb.xmin = int(xyxy[0])
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bb.Class = self.model.names[cls_id]
                bb.probability = conf

                boxes_msg.bounding_boxes.append(bb)

                # Draw bounding box on image
                cv2.rectangle(annotated_frame,
                              (bb.xmin, bb.ymin),
                              (bb.xmax, bb.ymax),
                              (0, 255, 0), 2)
                cv2.putText(annotated_frame,
                            f"{bb.Class} {conf:.2f}",
                            (bb.xmin, bb.ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # Run YOLOv8 skeleton detection (only if skeleton model is loaded)
        skeletons_msg = Skeletons()
        skeletons_msg.header = msg.header
        
        if self.skeleton_model is not None:
            skeleton_results = self.skeleton_model(frame, verbose=False)

            keypoint_types = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]

            for r in skeleton_results:
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    # Get keypoint data (shape: [num_people, 17, 2])
                    xy = r.keypoints.xy.cpu().numpy()  # coordinates
                    conf = r.keypoints.conf.cpu().numpy()  # confidence
                    
                    for person_idx in range(len(xy)):
                        sk = Skeleton()
                        person_kps = xy[person_idx]  # shape: [17, 2]
                        person_conf = conf[person_idx]  # shape: [17]
                        
                        # Add all keypoints for this person
                        for i in range(len(person_kps)):
                            keypoint = Keypoint()
                            keypoint.x = float(person_kps[i][0])
                            keypoint.y = float(person_kps[i][1])
                            keypoint.confidence = float(person_conf[i])
                            keypoint.type = keypoint_types[i]
                            sk.keypoints.append(keypoint)
                        
                        skeletons_msg.skeletons.append(sk)
                        
                        # Draw skeleton keypoints on image (only high confidence)
                        for i in range(len(person_kps)):
                            x, y = int(person_kps[i][0]), int(person_kps[i][1])
                            if person_conf[i] > 0.5:  # Only draw high confidence points
                                cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
                        
                        # Draw skeleton connections (only if both keypoints are high confidence)
                        skeleton_connections = [
                            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Upper body
                            (5, 11), (6, 12), (11, 12),  # Torso
                            (11, 13), (13, 15), (12, 14), (14, 16)  # Lower body
                        ]
                        for conn in skeleton_connections:
                            if person_conf[conn[0]] > 0.5 and person_conf[conn[1]] > 0.5:
                                pt1 = (int(person_kps[conn[0]][0]), int(person_kps[conn[0]][1]))
                                pt2 = (int(person_kps[conn[1]][0]), int(person_kps[conn[1]][1]))
                                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 2)
        rospy.loginfo(f"[YOLOv8] Detected {len(boxes_msg.bounding_boxes)} boxes and {len(skeletons_msg.skeletons)} skeletons.")
        # Publish detected bounding boxes
        self.pub_boxes.publish(boxes_msg)

        # Publish detected skeletons
        self.pub_skeletons.publish(skeletons_msg)

        # Publish debug image (annotated)
        if self.publish_debug:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            img_msg.header = msg.header
            self.pub_debug.publish(img_msg)

if __name__ == "__main__":
    YOLOv8Detector()
    rospy.spin()


