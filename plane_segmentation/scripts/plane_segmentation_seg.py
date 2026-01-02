#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
import tf

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

# ============================================================
# PointCloud2 utils
# ============================================================
FIELDS_XYZRGB = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("rgb", 12, PointField.FLOAT32, 1)
]

def pc2_to_xyzrgb(msg: PointCloud2):
    pts = []
    rgbs = []

    for p in pc2.read_points(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=False
    ):
        if not np.isfinite(p[2]):
            pts.append([np.nan, np.nan, np.nan])
            rgbs.append(np.nan)
        else:
            pts.append([p[0], p[1], p[2]])
            rgbs.append(p[3])
        
    return np.asarray(pts, dtype=np.float32), np.asarray(rgbs, dtype=np.float32)

def xyzrgb_to_pc2(pts, rgbs, frame_id, stamp):
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    cloud = [
        (float(pts[i,0]), float(pts[i,1]), float(pts[i,2]), rgbs[i])
        for i in range(pts.shape[0])
    ]

    return pc2.create_cloud(header, FIELDS_XYZRGB, cloud)

# ============================================================
# PlaneSegmentationPy
# ============================================================
class PlaneSegmentationPy:
    def __init__(self):
        rospy.init_node("plane_segmentation_seg")

        # Parameters
        self.pointcloud_topic = rospy.get_param("point_cloud_topic", "/xtion/depth_registered/points")
        self.base_frame = rospy.get_param("base_frame", "base_footprint")

        pre_pass = rospy.get_param("pre_pass_filter", [0.35, 1.0])
        seg_pass = rospy.get_param("seg_pass_filter", [0.01, 0.3])
        self.pre_pass_low, self.pre_pass_high = pre_pass
        self.seg_pass_low, self.seg_pass_high = seg_pass

        self.ransac_thresh = rospy.get_param("ransac_threshold", 0.015)
        self.voxel_size = rospy.get_param("voxel_size", 0.01)

        self.tf_listener = tf.TransformListener()

        # ROS I/O
        self.sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cb, queue_size=1)

        self.pub_plane = rospy.Publisher("/table_point_cloud", PointCloud2, queue_size=1)
        self.pub_objects = rospy.Publisher("/objects_point_cloud", PointCloud2, queue_size=1)

        rospy.loginfo("[PlaneSegmentationPy] Started")

    # ========================================================
    # TF: transform (cloud -> base_frame)
    # ========================================================
    def transform_points(self, pts, src_frame):
        if pts.shape[0] == 0:
            return pts

        self.tf_listener.waitForTransform(
            self.base_frame, src_frame, rospy.Time(0), rospy.Duration(1.0)
        )
        trans, rot = self.tf_listener.lookupTransform(
            self.base_frame, src_frame, rospy.Time(0)
        )

        R = tf.transformations.quaternion_matrix(rot)[:3, :3]
        t = np.array(trans)

        valid = np.isfinite(pts[:,2])
        pts_out = pts.copy()

        pts_out[valid] = (pts[valid] @ R.T) + t
        return pts_out

    # ========================================================
    # preprocessCloud() (z pass-through only)
    # ========================================================
    def preprocess(self, pts, rgbs):
        valid = np.isfinite(pts[:,2])
        pts, rgbs = pts[valid], rgbs[valid]

        z = pts[:, 2]
        keep = (z >= self.pre_pass_low) & (z <= self.pre_pass_high)
        return pts[keep], rgbs[keep]

    # ========================================================
    # segmentCloud()
    # ========================================================
    def segment(self, pts, rgbs):
        if pts.shape[0] < 200:
            return (
                np.zeros((0, 3)), np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3)), np.zeros((0,), dtype=np.float32)
            )

        # --- RANSAC ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_thresh,
            ransac_n=3,
            num_iterations=200
        )

        if len(inliers) == 0:
            return (
                np.zeros((0, 3)), np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3)), np.zeros((0,), dtype=np.float32)
            )

        # --- ExtractIndices (EXACT) ---
        table_pts = pts[inliers]
        table_rgbs = rgbs[inliers]

        mask = np.ones(len(pts), dtype=bool)
        mask[inliers] = False

        obj_pts = pts[mask]
        obj_rgbs = rgbs[mask]

        if obj_pts.shape[0] == 0:
            return table_pts, table_rgbs, np.zeros((0, 3)), np.zeros((0,), dtype=np.float32)

        # --- plane-aligned filter for objects ---
        a, b, c, d = plane_model
        n = np.array([a, b, c])
        n /= (np.linalg.norm(n) + 1e-12)

        R = self.rotation_from_vectors(n, np.array([0, 0, 1]))
        t = d * n

        obj_plane = (obj_pts @ R.T) + t
        z = obj_plane[:, 2]
        keep = (z >= self.seg_pass_low) & (z <= self.seg_pass_high)

        obj_back = (obj_plane[keep] - t) @ R
        obj_rgbs_f = obj_rgbs[keep]

        # # Debug (color)
        # rospy.loginfo_once(
        #     f"RGB sample before: {rgbs[:5]}, after: {obj_rgbs_f[:5]}"
        # )

        return table_pts, table_rgbs, obj_back, obj_rgbs_f

    # ========================================================
    # Eigen::Quaternionf::FromTwoVectors equivalent
    # ========================================================
    def rotation_from_vectors(self, src, dst):
        src = src / (np.linalg.norm(src) + 1e-12)
        dst = dst / (np.linalg.norm(dst) + 1e-12)
        v = np.cross(src, dst)
        c = np.dot(src, dst)
        s = np.linalg.norm(v)

        if s < 1e-8:
            return np.eye(3)

        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # ========================================================
    # ROS callback
    # ========================================================
    def cb(self, msg: PointCloud2):
        pts, rgbs = pc2_to_xyzrgb(msg)

        pts_base = self.transform_points(pts, msg.header.frame_id)
        pts_pre, rgbs_pre = self.preprocess(pts_base, rgbs)

        table_pts, table_rgbs, obj_pts, obj_rgbs = self.segment(
            pts_pre, rgbs_pre
        )

        stamp = msg.header.stamp
        self.pub_plane.publish(
            xyzrgb_to_pc2(table_pts, table_rgbs, self.base_frame, stamp)
        )
        self.pub_objects.publish(
            xyzrgb_to_pc2(obj_pts, obj_rgbs, self.base_frame, stamp)
        )

# ============================================================
if __name__ == "__main__":
    PlaneSegmentationPy()
    rospy.spin()
