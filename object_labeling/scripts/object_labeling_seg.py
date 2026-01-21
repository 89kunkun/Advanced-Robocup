import rospy
import os
import rospkg
import numpy as np
import open3d as o3d
import struct

from ultralytics import YOLO
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, Image, CameraInfo, PointField
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs 

# ============================================================
# PointCloud2 layouts
# ============================================================

FIELDS_XYZRGB = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("rgb", 12, PointField.FLOAT32, 1),
]

FIELDS_XYZL = [
    PointField("x",     0,  PointField.FLOAT32, 1),
    PointField("y",     4,  PointField.FLOAT32, 1),
    PointField("z",     8,  PointField.FLOAT32, 1),
    PointField("label", 12, PointField.UINT32,  1),
]

CLASS_DICT = {
    "unknown":  0,
    "cola":     1,
    "sprite":   2,
    "tray":     3,
    "cereal":   4,
    "person":   5,
}

# ============================================================
# Utils
# ============================================================

def pack_rgb(r, g, b):
    """Pack uint8 RGB into a single float32 (ROS PointCloud2 convention)."""
    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint))[0]

def pc2_to_xyz(msg: PointCloud2):
    """Convert PointCloud2 to Nx3 numpy array (skip NaNs)."""
    pts = np.array([[p[0], p[1], p[2]]
                    for p in pc2.read_points(msg, ("x", "y", "z"), skip_nans=True)],
                   dtype=np.float32)
    return pts

def xyzrgb_to_pc2(pts, rgbs, frame_id, stamp):
    """Create an XYZRGB PointCloud2 message from numpy arrays."""
    header = Header(frame_id=frame_id, stamp=stamp)
    cloud = [(float(pts[i,0]), float(pts[i,1]), float(pts[i,2]), float(rgbs[i]))
        for i in range(len(pts))]
    return pc2.create_cloud(header, FIELDS_XYZRGB, cloud)

def xyzl_to_pc2(pts, labels, frame_id, stamp):
    """Create XYZL PointCloud2 message (PointXYZL style)."""
    header = Header(frame_id=frame_id, stamp=stamp)
    cloud = [(float(pts[i,0]), float(pts[i,1]), float(pts[i,2]), int(labels[i]))
             for i in range(len(pts))]
    return pc2.create_cloud(header, FIELDS_XYZL, cloud)

# ============================================================
# ObjectLabelingPy
# ============================================================
class ObjectLabelingPy:
    """
    Inputs:
      - /objects_point_cloud (XYZRGB from plane_segmentation)
      - RGB image + depth_image + camera_info for YOLO mask + plate recovery
      - /table_point_cloud (for table center + plate z-band)

    Outputs:
      - /labeled_object_point_cloud : XYZL (label per point, per cluster)
      - /text_markers               : MarkerArray (class name per cluster)
      - /cluster_centroid           : PointStamped
      - /table_center + marker
      - /plate_point_cloud (XYZRGB) + /plate_center
    """

    def __init__(self):
        rospy.init_node("object_labeling_seg")

        # -----------------------
        # Topics parameters
        # -----------------------
        self.rgb_topic      = rospy.get_param("~rgb_topic", "/xtion/rgb/image_raw")
        self.objects_topic  = rospy.get_param("~objects_topic", "/objects_point_cloud")
        self.table_topic    = rospy.get_param("~table_topic", "/table_point_cloud")

        self.depth_img_topic = rospy.get_param("~depth_image_topic", "/xtion/depth_registered/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/xtion/depth_registered/camera_info")

        # Plate recovery target class name (default laptop)
        self.plate_class = rospy.get_param("~plate_class", "tray")

        # DBSCAN params
        self.dbscan_eps = rospy.get_param("~dbscan_eps", 0.03)
        self.dbscan_min_points = rospy.get_param("~dbscan_min_points", 80)

        # ----------------------------------------------------
        # YOLOv8 segmentation model
        # ----------------------------------------------------
        model_path = rospy.get_param("~model_path", "yolov8s_seg_70epoch.pt")
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            rp = rospkg.RosPack()
            weights_dir = os.path.join(rp.get_path("yolo_v8_detector"), "weights")
            candidate = os.path.join(weights_dir, model_path)
            if os.path.exists(candidate):
                model_path = candidate
        rospy.loginfo(f"[ObjectLabelingPy] Loading YOLOv8-seg: {model_path}")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # -----------------------
        # Subscribers
        # -----------------------
        rospy.Subscriber(self.rgb_topic, Image, self.cb_rgb, queue_size=1)
        rospy.Subscriber(self.objects_topic, PointCloud2, self.cb_objects, queue_size=1)
        rospy.Subscriber(self.table_topic, PointCloud2, self.cb_table, queue_size=1)

        rospy.Subscriber(self.depth_img_topic, Image, self.cb_depth_img, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info, queue_size=1)

        # -----------------------
        # Publishers
        # -----------------------
        # Geometry + labels
        self.pub_labeled = rospy.Publisher("/labeled_object_point_cloud", PointCloud2, queue_size=1)

        # Semantic visualization
        self.pub_text = rospy.Publisher("/text_markers", MarkerArray, queue_size=1)
        self.pub_centroid = rospy.Publisher("/cluster_centroid", PointStamped, queue_size=10)

        # Table info
        self.pub_table_center = rospy.Publisher("/table_center", PointStamped, queue_size=1)
        self.pub_table_marker = rospy.Publisher("/table_center_marker", Marker, queue_size=1)

        # Plate recovery output
        self.pub_plate_cloud = rospy.Publisher("/plate_point_cloud", PointCloud2, queue_size=1)
        self.pub_plate_center = rospy.Publisher("/plate_center", PointStamped, queue_size=1)

        # ----------------------------------------------------
        # Internal buffers
        # ----------------------------------------------------
        self.rgb = None
        self.depth_img = None        
        self.camera_info = None

        self.objects_pts = None
        self.objects_header = None

        self.table_pts = None
        self.table_header = None

        self.yolo_masks = []        # Cached YOLO masks (2D)

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("[ObjectLabelingPy] Node Started.")

    # ========================================================
    # YOLO cache
    # ========================================================
    def update_yolo_masks(self):
        """Run YOLO on latest RGB and cache masks (shared by cluster + plate)."""
        if self.rgb is None:
            return
        # results = self.model(self.rgb, verbose=False)
        # r = results[0]
        r = self.model(self.rgb, verbose=False)[0]
        if r.masks is None:
            self.yolo_masks = []
            return
        
        self.yolo_masks = []
        for i in range(len(r.masks.data)):
            cls_id = int(r.boxes.cls[i])
            name = self.model.names[cls_id]
            mask = r.masks.data[i].cpu().numpy() > 0.5
            self.yolo_masks.append({"name": name, "mask": mask})

    # ========================================================
    # Callbacks: raw sensor data
    # ========================================================
    def cb_rgb(self, msg):
        """Store latest RGB image."""
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.update_yolo_masks()

    def cb_depth_img(self, msg):
        """
        Depth image aligned with RGB.
        Converted to meters if needed.
        """
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001  # mm → m
        self.depth_img = depth
        self.plate_recovery()

    def cb_camera_info(self, msg):
        self.camera_info = msg

    # ========================================================
    # Callbacks: scene geometry
    # ========================================================
    def cb_objects(self, msg):
        """
        Object point cloud after plane segmentation.

        Important design choice:
        - We DO NOT modify the point cloud.
        - It is forwarded directly as labeled_object_point_cloud.
        - Semantic labels are published separately as markers.
        """
        pts = pc2_to_xyz(msg)
        if pts.size == 0:
            return
        self.objects_pts = pts
        self.objects_header = msg.header
        # Semantic processing
        self.cluster_objects_to_xyzl()

    def cb_table(self, msg):
        """Table plane point cloud and its centroid."""
        pts = pc2_to_xyz(msg)
        if pts.size == 0:
            return
        self.table_pts = pts
        self.table_header = msg.header

        center = pts.mean(axis=0)
        ps = PointStamped(header=msg.header)
        ps.point.x, ps.point.y, ps.point.z = center
        self.pub_table_center.publish(ps)

        mk = Marker()
        mk.header = msg.header
        mk.ns = "table_center"
        mk.id = 0
        mk.type = Marker.SPHERE
        mk.scale.x = mk.scale.y = mk.scale.z = 0.05
        mk.color.g = mk.color.a = 1.0
        mk.pose.position = ps.point
        self.pub_table_marker.publish(mk)

    # ========================================================
    # Object clustering and semantic labeling (XYZL labeling)
    # ========================================================
    def cluster_objects_to_xyzl(self):
        """
        - DBSCAN clustering (3D)
        - For each cluster, compute centroid
        - centroid TF -> rgb frame, project, lookup YOLO mask -> label_name
        - Assign label_id to ALL points in the cluster
        - Publish:
            /labeled_object_point_cloud (XYZL)
            /text_markers (same as before)
        """
        if self.objects_pts is None or self.camera_info is None:
            return

        # --- DBSCAN clustering in 3D ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.objects_pts)
        cluster_ids = np.array(
            pcd.cluster_dbscan(eps=float(self.dbscan_eps), min_points=int(self.dbscan_min_points),
                               print_progress=False)
        )

        labels_out = np.zeros(len(self.objects_pts), dtype=np.uint32)
        markers = MarkerArray()

        # Camera intrinsics
        K = np.array(self.camera_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        rgb_frame = self.camera_info.header.frame_id

        for cid in np.unique(cluster_ids):
            if cid < 0:
                continue
            
            idxs = np.where(cluster_ids == cid)[0]
            pts = self.objects_pts[idxs]
            centroid = pts.mean(axis=0)
            label_name = "unknown"

            # --- centroid TF -> rgb frame -> project -> mask lookup ---            
            if self.yolo_masks:
                try:
                    pt = PointStamped()
                    pt.header.frame_id = self.objects_header.frame_id   # base_footprint
                    pt.header.stamp = rospy.Time(0)
                    pt.point.x, pt.point.y, pt.point.z = centroid

                    pt_rgb = self.tf_buffer.transform(pt, rgb_frame, timeout=rospy.Duration(0.1))
                    X, Y, Z = pt_rgb.point.x, pt_rgb.point.y, pt_rgb.point.z
                    if Z > 0:
                        u = int(fx * X / Z + cx)
                        v = int(fy * Y / Z + cy)
                        for m in self.yolo_masks:
                            mask = m["mask"]
                            if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
                                if mask[v, u]:
                                    label_name = m["name"]
                                    break
                except Exception as e:
                    rospy.logwarn(f"[cluster_objects] TF/project failed: {e}")

            label_id = CLASS_DICT.get(label_name, 0)
            labels_out[idxs] = label_id

            # --- Publish text marker ---
            mk = Marker()
            mk.header = self.objects_header
            mk.ns = "objects"
            mk.id = int(cid)
            mk.type = Marker.TEXT_VIEW_FACING
            mk.text = label_name
            mk.pose.position.x = centroid[0]
            mk.pose.position.y = centroid[1]
            mk.pose.position.z = centroid[2] + 0.1
            mk.scale.z = 0.1
            mk.color.b = mk.color.a = 1.0
            markers.markers.append(mk)

            # --- Publish centroid ---
            # cps = PointStamped(header=self.objects_header).
            cps = PointStamped()
            cps.header.frame_id = self.objects_header.frame_id
            cps.header.stamp = self.objects_header.stamp
            cps.point.x, cps.point.y, cps.point.z = centroid
            self.pub_centroid.publish(cps)

        # publish markers
        self.pub_text.publish(markers)
        # publish XYZL labeled cloud
        self.pub_labeled.publish(
            xyzl_to_pc2(self.objects_pts, labels_out,
                        self.objects_header.frame_id,
                        self.objects_header.stamp)
        )

    # ========================================================
    # YOLOv8 segmentation → plate recovery (unchanged logic)
    # ========================================================
    def plate_recovery(self):
        """
        Recover a plate (or laptop) point cloud by fusing:
        - YOLOv8 segmentation (2D mask)
        - Depth image (per-pixel depth)
        - Camera intrinsics (back-projection)
        - Table plane constraint (z-band filtering)
        - Plane fitting (RANSAC)
        - Spatial consistency (DBSCAN)

        Output:
        - /plate_point_cloud  (XYZRGB)
        - /plate_center       (PointStamped)
        """

        # Safety checks: required inputs
        if (self.rgb is None or self.depth_img is None or
            self.camera_info is None or not self.yolo_masks or
            self.table_pts is None):
            return

        # table z band (in table/base frame)
        table_z = np.median(self.table_pts[:, 2])
        z_low, z_high = table_z - 0.01, table_z + 0.06

        # Camera intrinsics
        K = np.array(self.camera_info.K).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        plate_pts_all = []
        plate_rgbs_all = []

        for m in self.yolo_masks:
            if m["name"] != self.plate_class:
                continue

            vs, us = np.where(m["mask"])
            zs = self.depth_img[vs, us]
            valid = np.isfinite(zs) & (zs > 0.1)
            if np.count_nonzero(valid) < 100:
                continue

            vs, us, zs = vs[valid], us[valid], zs[valid]

            pts_base, rgbs_base = [], []

            for u, v, z in zip(us, vs, zs):
                X = (u - cx) * z / fx
                Y = (v - cy) * z / fy

                ps = PointStamped()
                ps.header.frame_id = self.camera_info.header.frame_id
                ps.header.stamp = rospy.Time(0)
                ps.point.x, ps.point.y, ps.point.z = X, Y, z

                try:
                    ps_b = self.tf_buffer.transform(ps, self.table_header.frame_id, rospy.Duration(0.1))
                except:
                    continue

                if z_low < ps_b.point.z < z_high:
                    pts_base.append([ps_b.point.x, ps_b.point.y, ps_b.point.z])
                    b, g, r = self.rgb[v, u]
                    rgbs_base.append(pack_rgb(r, g, b))

            if len(pts_base) < 80:
                continue

            plate_pts_all.append(np.asarray(pts_base))
            plate_rgbs_all.append(np.asarray(rgbs_base))

        if not plate_pts_all:
            return

        plate_pts = np.vstack(plate_pts_all)
        plate_rgbs = np.hstack(plate_rgbs_all)

        stamp = rospy.Time.now()
        frame_id = self.table_header.frame_id

        self.pub_plate_cloud.publish(
            xyzrgb_to_pc2(plate_pts, plate_rgbs, frame_id, stamp)
        )

        center = plate_pts.mean(axis=0)
        ps = PointStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.point.x, ps.point.y, ps.point.z = center
        self.pub_plate_center.publish(ps)

if __name__ == "__main__":
    ObjectLabelingPy()
    rospy.spin()
        
