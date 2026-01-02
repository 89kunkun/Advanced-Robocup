#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolo_v8_detector.msg import BoundingBoxes, BoundingBox

class YOLOv8Detector:
    def __init__(self):
        rospy.init_node("yolo_v8_node")

        # -----------------------
        # Params
        # -----------------------
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.weights_path = rospy.get_param("~weights_path", "")
        self.publish_debug = rospy.get_param("~publish_debug_image", True)

        rospy.loginfo(f"[YOLOv8-Seg] Loading model: {self.weights_path}")

        # -----------------------
        # Model
        # -----------------------
        self.model = YOLO(self.weights_path)

        self.bridge = CvBridge()

        # -----------------------
        # ROS IO
        # -----------------------
        self.sub = rospy.Subscriber(
            self.image_topic, Image, self.callback, queue_size=1
        )

        self.pub_boxes = rospy.Publisher(
            "/yolo_v8_detector/bounding_boxes", 
            BoundingBoxes, 
            queue_size=1
        )

        self.pub_mask = rospy.Publisher(
            "/yolo_v8_detector/mask_image", 
            Image, 
            queue_size=1
        )

        if self.publish_debug:
            self.pub_debug = rospy.Publisher(
                "/yolo_v8_detector/debug_image",
                Image,
                queue_size=1
            )

        rospy.loginfo("[YOLOv8-Seg] Node started.")

    # =====================================================
    def callback(self, msg):
        # Convert ROS -> CV2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run YOLOv8 Seg inference
        results = self.model(frame, conf=0.4, iou=0.5, verbose=False)
        r = results[0]

        # -----------------------
        # BoundingBoxes msg
        # -----------------------
        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header

        # -----------------------
        # Mask image (uint8)
        # -----------------------
        msk_img = np.zeros(frame.shape[:2], dtype=np.uint8)  # (H, W)
        annotated_frame = frame.copy()

        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls_id]

                # -------- BoundingBox --------
                bb = BoundingBox()
                bb.xmin = int(xyxy[0])
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bb.Class = class_name
                bb.probability = conf
                boxes_msg.bounding_boxes.append(bb)

                # -------- Mask --------
                if r.masks is not None:
                    mask = r.masks.data[i].cpu().numpy()  # (H, W), float32
                    mask_bin = (mask > 0.5).astype(np.uint8) * 255  # Unique ID per instance
                    msk_img = np.maximum(msk_img, mask_bin)

                    # draw contour
                    contours, _ = cv2.findContours(
                        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)

                # -------- BBox visualization --------
                cv2.rectangle(
                    annotated_frame,
                    (bb.xmin, bb.ymin),
                    (bb.xmax, bb.ymax),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    annotated_frame,
                    f"{bb.Class} {conf:.2f}",
                    (bb.xmin, bb.ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 
                    (0, 255, 0), 2
                )

        # -----------------------
        # Publish
        # -----------------------
        self.pub_boxes.publish(boxes_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(msk_img, "mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # Publish debug image
        if self.publish_debug:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            img_msg.header = msg.header
            self.pub_debug.publish(img_msg)

        cv2.imshow("YOLOv8-Seg Detection", annotated_frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    YOLOv8Detector()
    rospy.spin()


