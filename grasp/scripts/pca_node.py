#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray

import sensor_msgs.point_cloud2 as pc2

# ============================================================
# Basic math utils
# ============================================================

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n

def project_to_xy(v):
    vv = v.copy()
    vv[2] = 0.0
    return vv

def project_to_yz(v):
    vv = v.copy()
    vv[0] = 0.0
    return vv

def project_to_xz(v):
    vv = v.copy()
    vv[1] = 0.0
    return vv

def right_handed_fix(red, green, blue):
    # ensure red × green points to blue (dot > 0)
    c = np.cross(red, green)
    if np.linalg.norm(c) < 1e-9:
        return red, green, blue
    if np.dot(c, blue) < 0:
        blue = -blue
    return red, green, blue

def outward_fix_xy(axis, centroid_xy):
    """
    Make axis (assumed roughly in XY) point to "outside":
    centroid_xy is vector from origin to object centroid projected on XY.
    We want dot(axis, centroid_xy) > 0.
    """
    ref = normalize(centroid_xy)
    if np.linalg.norm(ref) < 1e-9:
        return axis
    if np.dot(axis, ref) < 0:
        return -axis
    return axis

# ============================================================
# PCA
# ============================================================

def compute_pca(pts: np.ndarray):
    """
    pts: (N,3)
    return: centroid, eigvals, axis_thin, axis_mid, axis_long
    """
    if pts.shape[0] < 3:
        return None

    centroid = pts.mean(axis=0)
    M = pts - centroid
    cov = (M.T @ M) / float(len(pts))

    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    axis_thin = eigvecs[:, 0]
    axis_mid  = eigvecs[:, 1]
    axis_long = eigvecs[:, 2]

    return centroid, eigvals, axis_thin, axis_mid, axis_long


# ============================================================
# cereal-specific logic (with "blue must be outward" in vertical)
# ============================================================

def compute_axes_cereal(pts, vertical_cos_threshold):
    res = compute_pca(pts)
    if res is None:
        return None

    centroid, _, thin_raw, mid_raw, long_raw = res

    long_raw = normalize(long_raw)
    mid_raw  = normalize(mid_raw)
    thin_raw = normalize(thin_raw)

    Z = np.array([0.0, 0.0, 1.0])
    cos_long_z = abs(np.dot(long_raw, Z))

    # "outside" reference direction on XY plane (from origin -> object)
    cxy = np.array([centroid[0], centroid[1], 0.0])
    ref_dir = normalize(cxy)

    # ========================================================
    # CASE 1: long ≈ vertical  (red axis should be +Z)
    # ========================================================
    if cos_long_z > vertical_cos_threshold:
        state = "CEREAL_LONG_VERTICAL"

        # (A) red: force to +Z (z-up)
        axis_red = np.array([0.0, 0.0, 1.0])

        # (B) project mid/thin to XY
        axis_mid  = normalize(project_to_xy(mid_raw))
        axis_thin = normalize(project_to_xy(thin_raw))

        # (C) swap mid and thin (你要求的：中轴/短轴对换)
        axis_mid, axis_thin = axis_thin, axis_mid

        # (D) pick green seed from swapped axis_mid, then make it outward
        axis_green = axis_mid
        if np.linalg.norm(axis_green) < 1e-6:
            axis_green = np.array([1.0, 0.0, 0.0])
        axis_green = normalize(axis_green)
        axis_green = outward_fix_xy(axis_green, cxy)

        # (E) build blue by right-hand rule, then re-orthogonalize green
        axis_blue = normalize(np.cross(axis_red, axis_green))
        axis_green = normalize(np.cross(axis_blue, axis_red))

        # (F) output projection again (XY), then re-orthogonalize
        axis_green = normalize(project_to_xy(axis_green))
        if np.linalg.norm(axis_green) < 1e-6:
            axis_green = np.array([1.0, 0.0, 0.0])
        axis_green = outward_fix_xy(axis_green, cxy)

        axis_blue = normalize(np.cross(axis_red, axis_green))
        axis_green = normalize(np.cross(axis_blue, axis_red))

        # ======= NEW: force BLUE to be outward (not toward robot) =======
        # If blue points opposite to ref_dir, flip BOTH green and blue to keep right-handed
        if np.linalg.norm(ref_dir) > 1e-6:
            if np.dot(axis_blue, ref_dir) < 0:
                axis_blue = -axis_blue
                axis_green = -axis_green

        # final right-hand safety
        axis_red, axis_green, axis_blue = right_handed_fix(axis_red, axis_green, axis_blue)

        # publish order = (long, mid, thin) = (red, green, blue)
        axis_long = axis_red
        axis_mid  = axis_green
        axis_thin = axis_blue
        return centroid, axis_long, axis_mid, axis_thin, state

    # ========================================================
    # CASE 2: long ≈ horizontal (red axis in XY)
    # ========================================================
    state = "CEREAL_LONG_HORIZONTAL"

    # (A) red: project long to XY
    axis_red = project_to_xy(long_raw)
    if np.linalg.norm(axis_red) < 1e-6:
        axis_red = np.array([1.0, 0.0, 0.0])
    axis_red = normalize(axis_red)

    # (B) blue: force to numeric downward (-Z)
    axis_blue = np.array([0.0, 0.0, -1.0])

    # (C) green: derived from blue and red to keep right-handed
    axis_green = normalize(np.cross(axis_blue, axis_red))
    if np.linalg.norm(axis_green) < 1e-6:
        axis_green = np.array([0.0, 1.0, 0.0])
    axis_green = normalize(axis_green)

    # (D) re-orthogonalize to be safe
    axis_red = normalize(project_to_xy(axis_red))
    axis_green = normalize(np.cross(axis_blue, axis_red))
    if np.linalg.norm(axis_green) < 1e-6:
        axis_green = np.array([0.0, 1.0, 0.0])
    axis_green = normalize(axis_green)
    axis_red, axis_green, axis_blue = right_handed_fix(axis_red, axis_green, axis_blue)

    axis_long = axis_red
    axis_mid  = axis_green
    axis_thin = axis_blue
    return centroid, axis_long, axis_mid, axis_thin, state


# ============================================================
# PCA Node
# ============================================================

class PCANodePy:
    def __init__(self):
        rospy.init_node("pca_node")

        self.prev_state = None

        self.target_label = rospy.get_param("~target_label", 1)
        self.vertical_cos_threshold = rospy.get_param("~vertical_cos_threshold", 0.8)
        self.horizontal_cos_threshold = rospy.get_param("~horizontal_cos_threshold", 0.8)
        self.axis_scale = rospy.get_param("~axis_scale", 0.25)

        rospy.Subscriber("/labeled_object_point_cloud", PointCloud2, self.cb_cloud, queue_size=1)

        self.pub_marker = rospy.Publisher("/pca_axes", MarkerArray, queue_size=1)
        self.pub_axis   = rospy.Publisher("/calculated_pca_axis", Float64MultiArray, queue_size=1)
        self.pub_pac_centroid = rospy.Publisher("/pca_object_centroid", PointStamped, queue_size=1)

        rospy.loginfo("[PCANodePy] Started. target_label=%d", self.target_label)

    def cb_cloud(self, msg: PointCloud2):
        pts = []
        for p in pc2.read_points(msg, ("x", "y", "z", "label"), skip_nans=True):
            if int(p[3]) == self.target_label:
                pts.append([p[0], p[1], p[2]])

        if len(pts) < 10:
            return

        pts = np.asarray(pts, dtype=np.float64)

        # cereal
        if self.target_label == 4:
            res = compute_axes_cereal(pts, self.vertical_cos_threshold)
            if res is None:
                return
            centroid, axis_long, axis_mid, axis_thin, state = res
            if state != self.prev_state:
                rospy.loginfo("[PCA-CEREAL] %s", state)
                self.prev_state = state

        # other classes (UNCHANGED)
        else:
            res = compute_pca(pts)
            if res is None:
                return

            centroid, _, thin_raw, mid_raw, long_raw = res
            thin_raw = normalize(thin_raw)
            long_raw = normalize(long_raw)

            Z = np.array([0.0, 0.0, 1.0])

            cos_long_z = abs(np.dot(long_raw, Z))
            cos_thin_z = abs(np.dot(thin_raw, Z))

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

            if state == "HORIZONTAL":
                axis_thin = -Z
                axis_long = long_raw.copy()
                axis_long[2] = 0.0
                axis_long = normalize(axis_long)
                axis_mid = normalize(np.cross(axis_thin, axis_long))

            elif state == "VERTICAL":
                axis_long = Z
                axis_mid = np.array([0.0, -1.0, 0.0])
                axis_thin = normalize(np.cross(axis_long, axis_mid))

            else:
                axis_long = long_raw
                axis_thin = thin_raw
                axis_mid = normalize(np.cross(axis_thin, axis_long))

            if state != self.prev_state:
                rospy.loginfo("[PCA] %s", state)
                self.prev_state = state

        # publish PCA matrix (row-major): [red; green; blue]
        arr = Float64MultiArray()
        arr.data = [
            axis_long[0], axis_long[1], axis_long[2],   # red
            axis_mid[0],  axis_mid[1],  axis_mid[2],    # green
            axis_thin[0], axis_thin[1], axis_thin[2],   # blue
        ]
        self.pub_axis.publish(arr)

        # rviz visualization
        self.publish_markers(msg.header.frame_id, centroid, axis_long, axis_mid, axis_thin)

        # publish centroid
        ps = PointStamped()
        ps.header.frame_id = msg.header.frame_id
        ps.header.stamp = rospy.Time.now()
        ps.point.x, ps.point.y, ps.point.z = centroid
        self.pub_pac_centroid.publish(ps)

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
        arr.markers.append(arrow(0, aL, (1.0, 0.0, 0.0)))  # red
        arr.markers.append(arrow(1, aM, (0.0, 1.0, 0.0)))  # green
        arr.markers.append(arrow(2, aT, (0.0, 0.0, 1.0)))  # blue
        self.pub_marker.publish(arr)


if __name__ == "__main__":
    PCANodePy()
    rospy.spin()
