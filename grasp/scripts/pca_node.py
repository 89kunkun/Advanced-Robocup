import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray

import sensor_msgs.point_cloud2 as pc2

# ============================================================
# PCA math utils
# ============================================================

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n

def compute_pca(pts: np.ndarray):
    """
    pts: (N,3)
    return: eigvals, axis_thin, axis_mid, axis_long
    """
    if pts.shape[0] < 3:
        return None
    
    centroid = pts.mean(axis=0)
    M = pts - centroid
    cov = (M.T @ M) / float(len(pts))

    eigvals, eigvecs = np.linalg.eigh(cov) # ascending
    axis_thin = eigvecs[:, 0]
    axis_mid  = eigvecs[:, 1]
    axis_long = eigvecs[:, 2]

    return centroid,eigvals, axis_thin, axis_mid, axis_long

# ============================================================
# PCA Node
# ============================================================

class PCANodePy:

    def __init__(self):
        rospy.init_node("pca_node")

        # Parameters
        self.target_label = rospy.get_param("~target_label", 1)
        self.vertical_cos_threshold = rospy.get_param("~vertical_cos_threshold", 0.8)
        self.horizontal_cos_threshold = rospy.get_param("~horizontal_cos_threshold", 0.8)
        self.axis_scale = rospy.get_param("~axis_scale", 0.25)

        self.last_centroid = None

        # Subscribers / Publishers
        rospy.Subscriber("/labeled_object_point_cloud", PointCloud2, self.cb_cloud, queue_size=1)

        self.pub_marker = rospy.Publisher("/pca_axes", MarkerArray, queue_size=1)
        self.pub_axis = rospy.Publisher("/calculated_pca_axis", Float64MultiArray, queue_size=1)

        rospy.loginfo("[PCANodePy] Started. target_label=%d", self.target_label)

    # ========================================================
    # Callbacks
    # ========================================================

    def cb_cloud(self, msg: PointCloud2):

        # 1) Filter points by label
        pts = []
        for p in pc2.read_points(msg, ("x", "y", "z", "label"), skip_nans=True):
            if int(p[3]) == self.target_label:
                pts.append([p[0], p[1], p[2]])

        if len(pts) < 10:
            return
        
        pts = np.asarray(pts, dtype=np.float64)

        # 2) PCA
        result = compute_pca(pts)
        if result is None:
            return
        
        centroid, _, axis_thin_raw, axis_mid_raw, axis_long_raw = result
        axis_thin_raw = normalize(axis_thin_raw)
        axis_long_raw = normalize(axis_long_raw)

        # 3) Orientation judgment
        Z = np.array([0, 0, 1.0])

        cos_long_z = abs(np.dot(axis_long_raw, Z))  # vertical test
        cos_thin_z = abs(np.dot(axis_thin_raw, Z))  # horizontal test

        is_vertical = cos_long_z > self.vertical_cos_threshold
        is_horizontal = cos_thin_z > self.horizontal_cos_threshold

        if is_horizontal and not is_vertical:
            state = "HORIZONTAL"
        elif is_vertical and not is_horizontal:
            state = "VERTICAL"
        elif is_vertical and is_horizontal:
            state = "HORIZONTAL" if cos_thin_z > cos_long_z else "VERTICAL"
        else:
            state = "TILTED"

        rospy.loginfo(
            "[PCA] label=%d | cos(long,Z)=%.3f | cos(thin,Z)=%.3f → %s",
            self.target_label,
            cos_long_z,
            cos_thin_z,
            state
        )

        # 4) Axis regularization based on state
        if state == "HORIZONTAL":
            axis_thin = -Z
            axis_long = axis_long_raw.copy()
            axis_long[2] = 0.0
            if np.linalg.norm(axis_long) < 1e-6:
                axis_long = np.array([1.0, 0.0, 0.0])
            axis_long = normalize(axis_long)
            axis_mid = normalize(np.cross(axis_thin, axis_long))

            # ===== 【新增】确保 axis_long 朝“物体外侧” =====
            ref_dir = centroid.copy()
            ref_dir[2] = 0.0
            ref_dir = normalize(ref_dir)

            if np.dot(axis_long, ref_dir) < 0:
                axis_long = -axis_long
                axis_mid = -axis_mid

        elif state == "VERTICAL":
            axis_long = Z
            axis_mid = np.array([0.0, -1.0, 0.0])
            axis_thin = normalize(np.cross(axis_long, axis_mid))

        else:  # TILTED
            axis_long = axis_long_raw
            axis_thin = axis_thin_raw
            axis_mid = normalize(np.cross(axis_thin, axis_long))

        # 5) publish PCA matrix (row-major)
        arr = Float64MultiArray()
        arr.data = [
            axis_long[0], axis_long[1], axis_long[2],
            axis_mid[0],  axis_mid[1],  axis_mid[2],
            axis_thin[0], axis_thin[1], axis_thin[2],
        ]
        self.pub_axis.publish(arr)

        # 6) Rviz visualization
        self.publish_markers(
            msg.header.frame_id,
            centroid,
            axis_long, axis_mid, axis_thin
        )
        
    # ========================================================
    # Visualization
    # ========================================================

    def publish_markers(self, frame, c, aL, aM, aT):

        def arrow(i, axis, color):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = rospy.Time.now()
            m.ns = "pca_axes"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            p0 = Point(x=c[0], y=c[1], z=c[2])
            p1 = Point(
                x=c[0] + axis[0] * self.axis_scale,
                y=c[1] + axis[1] * self.axis_scale,
                z=c[2] + axis[2] * self.axis_scale,
            )
            m.points = [p0, p1]

            m.scale.x = 0.01
            m.scale.y = 0.02
            m.scale.z = 0.03

            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = color
            return m
        
        arr = MarkerArray()
        arr.markers.append(arrow(0, aL, (1.0, 0.0, 0.0))) # long(red)
        arr.markers.append(arrow(1, aM, (0.0, 1.0, 0.0))) # mid(green)
        arr.markers.append(arrow(2, aT, (0.0, 0.0, 1.0))) # thin(blue)

        self.pub_marker.publish(arr)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    PCANodePy()
    rospy.spin()