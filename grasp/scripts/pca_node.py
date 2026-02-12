#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS node: compute PCA axes from a labeled point cloud and publish both numeric axes + RViz markers.

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
    """
    Normalize a vector v.
    If its norm is too small, return a copy of v unchanged to avoid division by zero.
    """
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n

def project_to_xy(v):
    """
    Project a 3D vector onto the XY plane by zeroing its Z component.
    """
    vv = v.copy()
    vv[2] = 0.0
    return vv

def project_to_yz(v):
    """
    Project a 3D vector onto the YZ plane by zeroing its X component.
    """
    vv = v.copy()
    vv[0] = 0.0
    return vv

def project_to_xz(v):
    """
    Project a 3D vector onto the XZ plane by zeroing its Y component.
    """
    vv = v.copy()
    vv[1] = 0.0
    return vv

def right_handed_fix(red, green, blue):
    """
    Enforce a right-handed coordinate system:
      red x green should point in the same general direction as blue.

    If cross(red, green) points opposite to blue, flip blue.
    (If you need to keep the triad strictly orthonormal, you may also re-orthogonalize elsewhere.)
    """
    # ensure red × green points to blue (dot > 0)
    c = np.cross(red, green)
    if np.linalg.norm(c) < 1e-9:
        return red, green, blue
    if np.dot(c, blue) < 0:
        blue = -blue
    return red, green, blue

def outward_fix_xy(axis, centroid_xy):
    """
    Make an axis (assumed roughly in XY plane) point "outward" from the robot/origin.

    centroid_xy: vector from origin -> object centroid projected on XY.
    We want dot(axis, centroid_xy) > 0.
    If not, flip axis.
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
    Perform PCA on a set of 3D points.

    Args:
        pts: (N, 3) numpy array.

    Returns:
        centroid: (3,) mean of points
        eigvals:  (3,) eigenvalues (ascending)
        axis_thin: eigenvector for smallest eigenvalue  (least variance)
        axis_mid:  eigenvector for middle eigenvalue
        axis_long: eigenvector for largest eigenvalue   (most variance)

    Note:
        np.linalg.eigh returns eigenvalues in ascending order for symmetric matrices.
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
    """
    Special axis selection rules for the "cereal" class (target_label == 4).

    Behavior:
      - Compute PCA axes as usual.
      - Decide if the long axis is approximately vertical by comparing with +Z.
      - If vertical: force red axis to +Z, swap mid/thin after projection, enforce outward directions.
      - If horizontal: force blue to -Z, red to projected long axis in XY, green from right-handed rule.

    Returns:
        centroid, axis_long(red), axis_mid(green), axis_thin(blue), state_string
    """
    res = compute_pca(pts)
    if res is None:
        return None

    centroid, _, thin_raw, mid_raw, long_raw = res

    long_raw = normalize(long_raw)
    mid_raw  = normalize(mid_raw)
    thin_raw = normalize(thin_raw)

    Z = np.array([0.0, 0.0, 1.0])
    cos_long_z = abs(np.dot(long_raw, Z))

    # "Outside" reference direction on XY plane (from origin -> object centroid)
    cxy = np.array([centroid[0], centroid[1], 0.0])
    ref_dir = normalize(cxy)

    # ========================================================
    # CASE 1: long axis is approximately vertical
    # ========================================================
    if cos_long_z > vertical_cos_threshold:
        state = "CEREAL_LONG_VERTICAL"

        # (A) Force red axis to be exactly +Z (up direction)
        axis_red = np.array([0.0, 0.0, 1.0])

        # (B) Project PCA mid/thin axes to XY plane (remove Z component)
        axis_mid  = normalize(project_to_xy(mid_raw))
        axis_thin = normalize(project_to_xy(thin_raw))

        # (C) Swap mid and thin axes (as requested)
        axis_mid, axis_thin = axis_thin, axis_mid

        # (D) Use the swapped axis_mid as "green" candidate and make it point outward
        axis_green = axis_mid
        if np.linalg.norm(axis_green) < 1e-6:
            axis_green = np.array([1.0, 0.0, 0.0])
        axis_green = normalize(axis_green)
        axis_green = outward_fix_xy(axis_green, cxy)

        # (E) Build blue using right-hand rule, then re-orthogonalize green
        axis_blue = normalize(np.cross(axis_red, axis_green))
        axis_green = normalize(np.cross(axis_blue, axis_red))

        # (F) Re-project green to XY again (guard against numeric drift), then re-orthogonalize again
        axis_green = normalize(project_to_xy(axis_green))
        if np.linalg.norm(axis_green) < 1e-6:
            axis_green = np.array([1.0, 0.0, 0.0])
        axis_green = outward_fix_xy(axis_green, cxy)

        axis_blue = normalize(np.cross(axis_red, axis_green))
        axis_green = normalize(np.cross(axis_blue, axis_red))

        # NEW RULE:
        # Force BLUE to point outward (same general direction as ref_dir).
        # If blue points inward, flip BOTH green and blue so the frame remains right-handed.
        if np.linalg.norm(ref_dir) > 1e-6:
            if np.dot(axis_blue, ref_dir) < 0:
                axis_blue = -axis_blue
                axis_green = -axis_green

        # Final right-hand safety check
        axis_red, axis_green, axis_blue = right_handed_fix(axis_red, axis_green, axis_blue)

        # Publish order: (long, mid, thin) mapped to (red, green, blue)
        axis_long = axis_red
        axis_mid  = axis_green
        axis_thin = axis_blue
        return centroid, axis_long, axis_mid, axis_thin, state

    # ========================================================
    # CASE 2: long axis is approximately horizontal
    # ========================================================
    state = "CEREAL_LONG_HORIZONTAL"

    # (A) Red: project long axis to XY plane
    axis_red = project_to_xy(long_raw)
    if np.linalg.norm(axis_red) < 1e-6:
        axis_red = np.array([1.0, 0.0, 0.0])
    axis_red = normalize(axis_red)

    # (B) Blue: force to numeric downward (-Z)
    axis_blue = np.array([0.0, 0.0, -1.0])

    # (C) Green: derived from blue and red to keep a right-handed frame
    axis_green = normalize(np.cross(axis_blue, axis_red))
    if np.linalg.norm(axis_green) < 1e-6:
        axis_green = np.array([0.0, 1.0, 0.0])
    axis_green = normalize(axis_green)

    # (D) Re-orthogonalize again for safety
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
# PCA Node (ROS)
# ============================================================

class PCANodePy:
    """
    ROS node that:
      - Subscribes to /labeled_object_point_cloud (PointCloud2 with fields x,y,z,label)
      - Extracts points belonging to a specific label (target_label)
      - Computes PCA axes (with special rules for cereal label=4)
      - Publishes:
          * /calculated_pca_axis (Float64MultiArray) as 3 axes stacked row-major [red;green;blue]
          * /pca_axes (MarkerArray) for RViz visualization
          * /pca_object_centroid (PointStamped)
    """

    def __init__(self):
        # Initialize node
        rospy.init_node("pca_node")

        # Track previous state string to avoid spamming logs
        self.prev_state = None

        # Parameters
        self.target_label = rospy.get_param("~target_label", 1)
        self.vertical_cos_threshold = rospy.get_param("~vertical_cos_threshold", 0.8)
        self.horizontal_cos_threshold = rospy.get_param("~horizontal_cos_threshold", 0.8)
        self.axis_scale = rospy.get_param("~axis_scale", 0.25)

        # Subscriber: labeled point cloud
        rospy.Subscriber("/labeled_object_point_cloud", PointCloud2, self.cb_cloud, queue_size=1)

        # Publishers:
        # RViz axis markers
        self.pub_marker = rospy.Publisher("/pca_axes", MarkerArray, queue_size=1)
        # Numeric PCA axes (Float64MultiArray)
        self.pub_axis   = rospy.Publisher("/calculated_pca_axis", Float64MultiArray, queue_size=1)
        # Object centroid
        self.pub_pac_centroid = rospy.Publisher("/pca_object_centroid", PointStamped, queue_size=1)

        rospy.loginfo("[PCANodePy] Started. target_label=%d", self.target_label)

    def cb_cloud(self, msg: PointCloud2):
        """
        Callback for incoming PointCloud2 message.
        Extract points with matching label, compute axes, then publish results.
        """
        pts = []

        # Iterate over point cloud fields (x, y, z, label)
        for p in pc2.read_points(msg, ("x", "y", "z", "label"), skip_nans=True):
            if int(p[3]) == self.target_label:
                pts.append([p[0], p[1], p[2]])

        # Require a minimum number of points for stability
        if len(pts) < 10:
            return

        pts = np.asarray(pts, dtype=np.float64)

        # ====================================================
        # Special rule set for cereal
        # ====================================================
        if self.target_label == 4:
            res = compute_axes_cereal(pts, self.vertical_cos_threshold)
            if res is None:
                return

            centroid, axis_long, axis_mid, axis_thin, state = res

            # Log state transitions only
            if state != self.prev_state:
                rospy.loginfo("[PCA-CEREAL] %s", state)
                self.prev_state = state

        # ====================================================
        # Default logic for other classes (unchanged)
        # ====================================================
        else:
            res = compute_pca(pts)
            if res is None:
                return

            centroid, _, thin_raw, mid_raw, long_raw = res
            thin_raw = normalize(thin_raw)
            long_raw = normalize(long_raw)

            Z = np.array([0.0, 0.0, 1.0])

            # Measure how aligned the long/thin axes are with Z
            cos_long_z = abs(np.dot(long_raw, Z))
            cos_thin_z = abs(np.dot(thin_raw, Z))

            # Decide orientation mode by thresholds
            is_vertical = cos_long_z > self.vertical_cos_threshold
            is_horizontal = cos_thin_z > self.horizontal_cos_threshold

            # Resolve state
            if is_horizontal and not is_vertical:
                state = "HORIZONTAL"
            elif is_vertical and not is_horizontal:
                state = "VERTICAL"
            elif is_vertical and is_horizontal:
                # If both are "high", choose the one that is more aligned
                state = "HORIZONTAL" if cos_thin_z > cos_long_z else "VERTICAL"
            else:
                state = "TILTED"

            # Build axes by state
            if state == "HORIZONTAL":
                # Thin axis forced downward
                axis_thin = -Z
                # Long axis projected onto XY plane
                axis_long = long_raw.copy()
                axis_long[2] = 0.0
                axis_long = normalize(axis_long)
                # Mid axis via right-handed rule
                axis_mid = normalize(np.cross(axis_thin, axis_long))

            elif state == "VERTICAL":
                # Long axis forced upward
                axis_long = Z
                # Choose a fixed mid axis (points along -Y)
                axis_mid = np.array([0.0, -1.0, 0.0])
                # Thin axis via cross product
                axis_thin = normalize(np.cross(axis_long, axis_mid))

            else:
                # Tilted: use PCA long and thin as is
                axis_long = long_raw
                axis_thin = thin_raw
                axis_mid = normalize(np.cross(axis_thin, axis_long))

            # Log state transitions only
            if state != self.prev_state:
                rospy.loginfo("[PCA] %s", state)
                self.prev_state = state

        # ====================================================
        # Publish numeric PCA axes
        # ====================================================
        # Publish PCA axes matrix (row-major): [red; green; blue]
        arr = Float64MultiArray()
        arr.data = [
            axis_long[0], axis_long[1], axis_long[2],   # red (long axis)
            axis_mid[0],  axis_mid[1],  axis_mid[2],    # green (mid axis)
            axis_thin[0], axis_thin[1], axis_thin[2],   # blue (thin axis)
        ]
        self.pub_axis.publish(arr)

        # Publish RViz markers for visualization
        self.publish_markers(msg.header.frame_id, centroid, axis_long, axis_mid, axis_thin)

        # Publish centroid as PointStamped
        ps = PointStamped()
        ps.header.frame_id = msg.header.frame_id
        ps.header.stamp = rospy.Time.now()
        ps.point.x, ps.point.y, ps.point.z = centroid
        self.pub_pac_centroid.publish(ps)

    def publish_markers(self, frame, c, aL, aM, aT):
        """
        Publish 3 arrow markers (red/green/blue) representing PCA axes.
        """
        def arrow(i, axis, color):
            """
            Create an RViz arrow marker from centroid in direction 'axis'.
            color is (r,g,b).
            """
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = rospy.Time.now()
            m.ns = "pca_axes"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD

            # Arrow starts at centroid and ends at centroid + axis * scale
            p0 = Point(x=c[0], y=c[1], z=c[2])
            p1 = Point(
                x=c[0] + axis[0] * self.axis_scale,
                y=c[1] + axis[1] * self.axis_scale,
                z=c[2] + axis[2] * self.axis_scale,
            )
            m.points = [p0, p1]

            # Arrow thickness
            m.scale.x = 0.01
            m.scale.y = 0.02
            m.scale.z = 0.03

            # Marker color + alpha
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = color
            return m

        # Build and publish marker array
        arr = MarkerArray()
        arr.markers.append(arrow(0, aL, (1.0, 0.0, 0.0)))  # red axis (long)
        arr.markers.append(arrow(1, aM, (0.0, 1.0, 0.0)))  # green axis (mid)
        arr.markers.append(arrow(2, aT, (0.0, 0.0, 1.0)))  # blue axis (thin)
        self.pub_marker.publish(arr)


if __name__ == "__main__":
    # Instantiate node and spin to process callbacks
    PCANodePy()
    rospy.spin()
