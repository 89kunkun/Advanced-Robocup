#!/usr/bin/env python3
import rospy
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolo_v8_detector.msg import BoundingBoxes, BoundingBox

class YOLOv8Detector:
    def __init__(self):
        rospy.init_node("yolo_v8_node")

        # Load params
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.weights_path = rospy.get_param("~weights_path", "")
        self.publish_debug = rospy.get_param("~publish_debug_image", True)

        rospy.loginfo(f"[YOLOv8] Loading model: {self.weights_path}")

        # Load YOLOv8 model (correct API)
        self.model = YOLO(self.weights_path)

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback, queue_size=1)

        self.pub_boxes = rospy.Publisher(
            "/yolo_v8_detector/bounding_boxes", 
            BoundingBoxes, 
            queue_size=1
        )

        if self.publish_debug:
            self.pub_debug = rospy.Publisher(
                "/yolo_v8_detector/debug_image",
                Image,
                queue_size=1
            )

        rospy.loginfo("[YOLOv8] Node started.")

    def callback(self, msg):
        # Convert ROS -> CV2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run YOLOv8
        results = self.model(frame, verbose=False)

        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header

        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls)
                conf = float(box.conf)

                # Fill BoundingBox msg
                bb = BoundingBox()
                bb.xmin = int(xyxy[0])
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bb.Class = self.model.names[cls_id]
                bb.probability = conf

                boxes_msg.bounding_boxes.append(bb)

                # Draw box on image
                cv2.rectangle(annotated_frame,
                              (bb.xmin, bb.ymin),
                              (bb.xmax, bb.ymax),
                              (0, 255, 0), 2)
                cv2.putText(annotated_frame,
                            f"{bb.Class} {conf:.2f}",
                            (bb.xmin, bb.ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # Publish boxes
        self.pub_boxes.publish(boxes_msg)

        # Publish debug image
        if self.publish_debug:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            img_msg.header = msg.header
            self.pub_debug.publish(img_msg)

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    YOLOv8Detector()
    rospy.spin()


