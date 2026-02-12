#!/usr/bin/env python3
# ROS node: detect plate overhang points by comparing plate point cloud against a cleaned table hull.
# Also provides a "check grasp point" handshake:
#   - Receives /check_plate_grasp_point (PointStamped)
#   - Waits until overhang cloud is "ready"
#   - Searches along +X (then -X fallback) to find a point near overhang cloud
#   - Publishes /updated_grasp_point and /if_in_plate_area (Bool)

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

# -------------------------
# PointCloud2 layouts
# -------------------------

# PointCloud2 field definition for (x,y,z,rgb) packed as float32
# NOTE: rgb is stored as float32 (common PCL convention), not three separate uint8 fields.
FIELDS_XYZRGB = [
    PointField("x", 0,   PointField.FLOAT32, 1),
    PointField("y", 4,   PointField.FLOAT32, 1),
    PointField("z", 8,   PointField.FLOAT32, 1),
    PointField("rgb", 12, PointField.FLOAT32, 1),
]

def pc2_to_xyzrgb_skip_nans(msg: PointCloud2):
    """
    Convert PointCloud2 -> (pts, rgbs) numpy arrays, skipping NaNs.

    Returns:
        pts:  (N,3) float32
        rgbs: (N,)  float32 (packed rgb)
    """
    pts, rgbs = [], []
    for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
        rgbs.append(p[3])
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    return np.asarray(pts, np.float32), np.asarray(rgbs, np.float32)

def xyzrgb_to_pc2(pts, rgbs, frame_id, stamp):
    """
    Convert (pts, rgbs) numpy arrays back into a PointCloud2 message.
    """
    header = Header(frame_id=frame_id, stamp=stamp)
    cloud = [(float(pts[i,0]), float(pts[i,1]), float(pts[i,2]), float(rgbs[i]))
             for i in range(len(pts))]
    return pc2.create_cloud(header, FIELDS_XYZRGB, cloud)

# -------------------------
# Geometry helpers
# -------------------------

def fit_plane_from_points(P: np.ndarray):
    """
    Fit a plane to 3D points using SVD.

    Plane form: n·x + d = 0
    Returns:
        n: (3,) unit normal vector (float32)
        d: float offset
    """
    if P.shape[0] < 3:
        # Default plane normal if insufficient points
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0
    c = P.mean(axis=0)
    X = P - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[-1, :]  # smallest singular vector
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(np.dot(n, c))
    return n.astype(np.float32), float(d)

def rotation_from_vectors(src, dst):
    """
    Compute a rotation matrix that rotates vector 'src' to vector 'dst'
    using Rodrigues' rotation formula.

    Returns:
        R: (3,3) rotation matrix (float32)
    """
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        # src and dst are (almost) parallel -> identity rotation
        return np.eye(3, dtype=np.float32)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

def convex_hull_2d(points_xy: np.ndarray):
    """
    Compute the convex hull of 2D points using a monotonic chain algorithm.

    Args:
        points_xy: (N,2)

    Returns:
        hull: (M,2) vertices in CCW order (no repeated last point)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts

    # Sort points by x then y
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]

    def cross(o, a, b):
        # 2D cross product of OA and OB (sign indicates turn direction)
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper to form full hull
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)

def expand_polygon(poly: np.ndarray, margin: float):
    """
    Expand a convex polygon outward approximately by scaling relative to its centroid.

    This is a simple heuristic expansion:
      - Compute centroid
      - Scale vertex offsets so average radius increases by 'margin'
    """
    if poly.shape[0] < 3 or margin <= 0.0:
        return poly
    c = poly.mean(axis=0)
    v = poly - c
    r = np.linalg.norm(v, axis=1).mean()
    if r < 1e-6:
        return poly
    s = 1.0 + (margin / r)
    return c + v * s

def points_inside_convex_polygon(points: np.ndarray, poly: np.ndarray):
    """
    Check whether each point lies inside (or on the boundary of) a convex polygon.

    Args:
        points: (N,2)
        poly:   (M,2) convex polygon in CCW order

    Returns:
        inside_mask: (N,) boolean mask
    """
    if poly.shape[0] < 3:
        return np.zeros((points.shape[0],), dtype=bool)

    P = np.asarray(points, dtype=np.float64)
    V = np.asarray(poly, dtype=np.float64)
    Vi = V
    Vj = np.roll(V, -1, axis=0)
    E = Vj - Vi                      # edge vectors
    W = P[:, None, :] - Vi[None, :, :]  # point relative to each vertex

    # For CCW polygon, points are inside if cross(edge, point-vertex) >= 0 for all edges
    cross_vals = E[None, :, 0] * W[:, :, 1] - E[None, :, 1] * W[:, :, 0]
    return np.all(cross_vals >= -1e-9, axis=1)

# -------------------------
# Cleaning helpers
# -------------------------

def filter_table_by_z_quantile(table_pts: np.ndarray, keep_quantile: float):
    """
    Keep only lower part of table points based on Z quantile in table-aligned coordinates.
    Useful to remove tall clutter/noise (e.g., objects on the table) from hull building.

    keep_quantile:
        e.g., 0.6 means keep points with z <= 60th percentile.
    """
    if table_pts.shape[0] < 10:
        return table_pts
    z = table_pts[:, 2]
    q = float(np.clip(keep_quantile, 0.05, 0.95))
    thr = np.quantile(z, q)
    return table_pts[z <= thr]

def remove_table_cells_overlapping_plate(table_xy: np.ndarray, plate_xy: np.ndarray, cell: float):
    """
    Remove table points in grid cells that overlap with plate points.

    This helps prevent the hull from shrinking inward near the plate area by
    erasing table samples under/near the plate.

    Args:
        table_xy: (Nt,2) table points in plane coordinates
        plate_xy: (Np,2) plate points in plane coordinates
        cell: cell size (meters)

    Returns:
        keep_mask: (Nt,) boolean mask; True means keep that table point.
    """
    if table_xy.shape[0] == 0 or plate_xy.shape[0] == 0:
        return np.ones((table_xy.shape[0],), dtype=bool)

    c = float(max(cell, 1e-4))
    tp = np.floor(table_xy / c).astype(np.int64)
    pp = np.floor(plate_xy / c).astype(np.int64)

    plate_cells = set((int(a), int(b)) for a, b in pp)
    keep = np.array([ (int(a), int(b)) not in plate_cells for a, b in tp ], dtype=bool)
    return keep

# -------------------------
# Compute hull (cached) + overhang
# -------------------------

def build_clean_table_hull(table_pts, plate_pts, table_margin, table_keep_quantile, erase_cell,
                           min_clean_table_pts=300, min_hull_vertices=12):
    """
    Build a stable table boundary hull in a table-aligned coordinate frame.

    Steps:
      1) Fit table plane from table_pts.
      2) Compute rotation R that aligns plane normal to +Z.
      3) Transform points into this aligned frame.
      4) Keep only lower z-quantile of table points (remove clutter).
      5) Remove table cells that overlap with plate points (optional cleanup).
      6) Compute 2D convex hull on XY.
      7) Expand hull outward by table_margin.

    Returns:
        hull: (M,2) polygon vertices in plane XY coords (expanded)
        R:    (3,3) rotation matrix used to align plane to +Z
        t:    (3,) translation vector used in transform
        ok:   bool success
    """
    if table_pts.shape[0] < 80:
        return None, None, None, False

    # Plane fit for table
    n, d = fit_plane_from_points(table_pts)

    # Rotation that maps n -> +Z
    R = rotation_from_vectors(n, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Translation that brings plane to z=0 after rotation:
    # x' = x R^T + t, where t = -d * n (in original frame) in this formulation.
    t = (-d) * n

    # Transform plate points into plane-aligned coordinates
    plate_plane = (plate_pts @ R.T) + t

    # Keep the lower portion of table points (to reduce influence of objects)
    table_low = filter_table_by_z_quantile(table_pts, keep_quantile=table_keep_quantile)
    if table_low.shape[0] < 50:
        table_low = table_pts

    table_low_plane = (table_low @ R.T) + t

    # Remove table points in grid cells that overlap with plate points
    if plate_plane.shape[0] > 0:
        keep_mask = remove_table_cells_overlapping_plate(
            table_low_plane[:, :2], plate_plane[:, :2], cell=erase_cell
        )
        table_clean_plane = table_low_plane[keep_mask]
    else:
        table_clean_plane = table_low_plane

    # Require enough cleaned points
    if table_clean_plane.shape[0] < min_clean_table_pts:
        return None, None, None, False

    # Build convex hull in plane XY
    hull = convex_hull_2d(table_clean_plane[:, :2])
    if hull.shape[0] < min_hull_vertices:
        return None, None, None, False

    # Expand hull outward slightly for safety margin
    hull = expand_polygon(hull, float(table_margin))
    return hull, R, t, True

def compute_overhang_with_cached_hull(plate_pts, plate_rgbs, hull, R, t, z_min):
    """
    Compute plate overhang points using a precomputed table hull (in plane coords).

    Definition here:
      - Transform plate points into plane coords.
      - Determine which points are OUTSIDE the table hull polygon in XY.
      - Also require the point to be above z_min in plane coords (filter near-plane noise).
      - Return those points transformed back into original coordinates.

    Returns:
        overhang_pts: (K,3) float32 in original frame
        overhang_rgbs: (K,) float32 packed rgb
    """
    plate_plane = (plate_pts @ R.T) + t
    inside = points_inside_convex_polygon(plate_plane[:, :2], hull)
    high = plate_plane[:, 2] > float(z_min)
    keep = (~inside) & high
    if not np.any(keep):
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    overhang_plane = plate_plane[keep]
    overhang_back = (overhang_plane - t) @ R
    return overhang_back.astype(np.float32), plate_rgbs[keep].astype(np.float32)

# -------------------------
# ROS Node
# -------------------------

class PlateOverhangNode:
    """
    ROS node that:
      - Subscribes to:
          * table point cloud (/table_point_cloud)
          * plate point cloud (/plate_point_cloud)
          * check point (/check_plate_grasp_point)
      - Publishes:
          * overhang cloud (/plate_overhang_cloud)
          * boolean check result (/if_in_plate_area)
          * updated grasp point (/updated_grasp_point)

    Key concepts:
      - A cached convex hull is built from the cleaned table point cloud (aligned to table plane).
      - Overhang points are plate points outside this hull and above z_min in plane coords.
      - Stability: can hold last_good overhang result if current frame is weak.
      - READY logic: defined by last_good overhang point count >= min_ready_overhang_pts.
      - FREEZE option: once ready, freeze the overhang cloud for consistency.
      - Grasp check: if overhang ready, do NN search along +X then -X to find a nearby valid point.
    """

    def __init__(self):
        rospy.init_node("plate_overhang_node")

        # Input topics
        self.table_topic = rospy.get_param("~table_topic", "/table_point_cloud")
        self.plate_topic = rospy.get_param("~plate_topic", "/plate_point_cloud")

        # Output topic for overhang cloud
        self.out_topic   = rospy.get_param("~out_topic", "/plate_overhang_cloud")

        # -------------------------
        # Geometry parameters
        # -------------------------
        # Minimum z (in plane coords) to accept as "above the table plane"
        self.z_min = float(rospy.get_param("~z_min", 0.006))

        # Expand table hull outward by this margin (meters)
        self.table_margin = float(rospy.get_param("~table_margin", 0.02))

        # Keep only bottom quantile of table points when building hull (to remove clutter)
        self.table_keep_quantile = float(rospy.get_param("~table_keep_quantile", 0.6))

        # Cell size for removing table samples overlapping plate area (meters)
        self.erase_cell = float(rospy.get_param("~erase_cell", 0.01))

        # -------------------------
        # Minimum point count gates
        # -------------------------
        self.min_table_pts = int(rospy.get_param("~min_table_pts", 80))
        self.min_plate_pts = int(rospy.get_param("~min_plate_pts", 30))
        self.min_clean_table_pts = int(rospy.get_param("~min_clean_table_pts", 300))
        self.min_hull_vertices   = int(rospy.get_param("~min_hull_vertices", 12))

        # -------------------------
        # Stability / caching settings
        # -------------------------
        # Minimum overhang points to consider the output "ok" for this frame
        self.min_overhang_pts = int(rospy.get_param("~min_overhang_pts", 300))

        # If True, publish last_good overhang cloud when current frame is not ok
        self.hold_last_good   = bool(rospy.get_param("~hold_last_good", True))

        # How often to rebuild the table hull (seconds)
        self.hull_update_period = float(rospy.get_param("~hull_update_period", 1.5))

        # -------------------------
        # Check topics for grasp handshake
        # -------------------------
        self.check_topic = rospy.get_param("~check_topic", "/check_plate_grasp_point")
        self.if_topic = rospy.get_param("~if_topic", "/if_in_plate_area")
        self.updated_point_topic = rospy.get_param("~updated_point_topic", "/updated_grasp_point")

        # Nearest neighbor radius threshold (meters)
        self.nn_radius = float(rospy.get_param("~nn_radius", 0.02))   # comment says 1cm, value is 2cm

        # Step size for searching along x (meters)
        self.step_x = float(rospy.get_param("~step_x", 0.01))         # x += 0.01

        # Maximum search steps (max displacement = max_steps * step_x)
        self.max_steps = int(rospy.get_param("~max_steps", 200))      # 200*0.01 = 2m

        # -------------------------
        # Pending behavior when overhang is not ready yet
        # -------------------------
        # Ready threshold: last_good overhang points >= this value
        self.min_ready_overhang_pts = int(rospy.get_param("~min_ready_overhang_pts", 200))

        # How long to keep a pending check point before dropping it (seconds)
        self.pending_timeout = float(rospy.get_param("~pending_timeout", 4.0))
        self.pending_check_msg = None
        self.pending_start_time = rospy.Time(0)

        # Periodic printing of ready info (seconds)
        self.ready_print_period = float(rospy.get_param("~ready_print_period", 1.0))
        self._last_ready_print = rospy.Time(0)

        # -------------------------
        # FREEZE behavior: lock overhang cloud once ready
        # -------------------------
        self.freeze_on_ready = bool(rospy.get_param("~freeze_on_ready", True))
        self.frozen = False
        self.frozen_pts = np.zeros((0, 3), dtype=np.float32)
        self.frozen_rgbs = np.zeros((0,), dtype=np.float32)

        # -------------------------
        # Cached inputs
        # -------------------------
        self.table_pts = None         # latest table points (xyz only)
        self.table_frame = None       # frame id for table cloud

        # -------------------------
        # Cached hull representation
        # -------------------------
        self.hull = None              # (M,2) polygon in plane coords
        self.R = None                 # rotation to plane coords
        self.t = None                 # translation to plane coords
        self.last_hull_time = rospy.Time(0)

        # -------------------------
        # Last good overhang output (used for stability + NN search)
        # -------------------------
        self.last_good_pts = np.zeros((0, 3), dtype=np.float32)
        self.last_good_rgbs = np.zeros((0,), dtype=np.float32)

        # -------------------------
        # ROS publishers
        # -------------------------
        self.pub_overhang = rospy.Publisher(self.out_topic, PointCloud2, queue_size=1)
        self.pub_if = rospy.Publisher(self.if_topic, Bool, queue_size=10)
        self.pub_updated = rospy.Publisher(self.updated_point_topic, PointStamped, queue_size=10)

        # -------------------------
        # ROS subscribers
        # -------------------------
        rospy.Subscriber(self.table_topic, PointCloud2, self.cb_table, queue_size=1)
        rospy.Subscriber(self.plate_topic, PointCloud2, self.cb_plate, queue_size=1)
        rospy.Subscriber(self.check_topic, PointStamped, self.cb_check_point, queue_size=10)

        # Timer: handles pending check points and periodic ready prints
        self.timer = rospy.Timer(rospy.Duration(0.1), self.cb_timer)

        rospy.loginfo("[PlateOverhangNode] Started.")
        rospy.loginfo(f"check_topic={self.check_topic} -> if_topic={self.if_topic} -> updated_point={self.updated_point_topic}")
        rospy.loginfo(f"nn_radius={self.nn_radius}, step_x={self.step_x}, max_steps={self.max_steps}")
        rospy.loginfo(f"min_ready_overhang_pts={self.min_ready_overhang_pts}, pending_timeout={self.pending_timeout}s, ready_print_period={self.ready_print_period}s")
        rospy.logwarn(f"[FREEZE] freeze_on_ready={self.freeze_on_ready} (will freeze when ready=True)")

    def cb_table(self, msg: PointCloud2):
        """
        Receive table point cloud and cache its xyz points.
        If too few points, clear cache.
        """
        pts, _ = pc2_to_xyzrgb_skip_nans(msg)
        if pts.shape[0] < self.min_table_pts:
            self.table_pts = None
            return
        self.table_pts = pts
        self.table_frame = msg.header.frame_id

    def maybe_update_hull(self, now, plate_pts):
        """
        Update (or reuse) the cached table hull.
        Hull is rebuilt at most once per hull_update_period seconds.
        """
        if self.table_pts is None:
            return False
        if (now - self.last_hull_time).to_sec() < self.hull_update_period and self.hull is not None:
            return True

        hull, R, t, ok = build_clean_table_hull(
            table_pts=self.table_pts,
            plate_pts=plate_pts,
            table_margin=self.table_margin,
            table_keep_quantile=self.table_keep_quantile,
            erase_cell=self.erase_cell,
            min_clean_table_pts=self.min_clean_table_pts,
            min_hull_vertices=self.min_hull_vertices
        )
        if ok:
            self.hull, self.R, self.t = hull, R, t
            self.last_hull_time = now
            return True
        # If build fails, but we already have a hull cached, keep using it
        return self.hull is not None

    def _overhang_ready(self):
        """
        Define "ready" as last_good overhang point count >= min_ready_overhang_pts.
        Returns:
            (ready_bool, count_int)
        """
        if self.last_good_pts is None:
            return False, 0
        n = int(self.last_good_pts.shape[0])
        return (n >= self.min_ready_overhang_pts), n

    def _active_cloud(self):
        """
        The cloud used for NN checking:
          - If frozen: use frozen overhang cloud
          - Otherwise: use last_good overhang cloud
        """
        if self.frozen:
            return self.frozen_pts
        return self.last_good_pts

    def cb_plate(self, msg: PointCloud2):
        """
        Receive plate point cloud, compute overhang against cached table hull, then publish stable output.
        If frozen, do NOT update overhang anymore; just republish frozen cloud.
        """
        # === FREEZE: after frozen, do not update; only publish frozen cloud ===
        if self.frozen:
            self.pub_overhang.publish(
                xyzrgb_to_pc2(self.frozen_pts, self.frozen_rgbs, msg.header.frame_id, msg.header.stamp)
            )
            return

        # Need a valid cached table cloud
        if self.table_pts is None:
            return

        plate_pts, plate_rgbs = pc2_to_xyzrgb_skip_nans(msg)

        # Not enough plate points -> publish stable fallback
        if plate_pts.shape[0] < self.min_plate_pts:
            self.publish_stable(msg.header.frame_id, msg.header.stamp, ok=False, pts=None, rgbs=None)
            return

        # Frame consistency check
        if self.table_frame is not None and msg.header.frame_id != self.table_frame:
            rospy.logwarn_throttle(2.0, f"[PlateOverhangNode] frame mismatch: table={self.table_frame}, plate={msg.header.frame_id}")
            self.publish_stable(msg.header.frame_id, msg.header.stamp, ok=False, pts=None, rgbs=None)
            return

        now = rospy.Time.now()

        # Ensure hull is available (rebuild periodically)
        if not self.maybe_update_hull(now, plate_pts):
            self.publish_stable(msg.header.frame_id, msg.header.stamp, ok=False, pts=None, rgbs=None)
            return

        # Compute overhang points using cached hull
        overhang_pts, overhang_rgbs = compute_overhang_with_cached_hull(
            plate_pts, plate_rgbs, self.hull, self.R, self.t, z_min=self.z_min
        )

        # "ok" for this frame if enough overhang points
        ok = overhang_pts.shape[0] >= self.min_overhang_pts

        # Publish stable (possibly last_good) output
        self.publish_stable(msg.header.frame_id, msg.header.stamp, ok=ok, pts=overhang_pts, rgbs=overhang_rgbs)

    def publish_stable(self, frame_id, stamp, ok, pts, rgbs):
        """
        Publish overhang point cloud with stability rules:
          - If ok: update last_good and publish current
          - If not ok:
              * if hold_last_good: publish last_good
              * else publish empty cloud

        Also handles FREEZE:
          - If freeze_on_ready and ready becomes True:
              lock (freeze) the current last_good cloud and publish it immediately.
        """
        if ok and pts is not None:
            # Update "last good" cache
            self.last_good_pts = pts
            self.last_good_rgbs = rgbs
            out_pts, out_rgbs = pts, rgbs
        else:
            # Fall back to last_good or empty
            if self.hold_last_good:
                out_pts, out_rgbs = self.last_good_pts, self.last_good_rgbs
            else:
                out_pts = np.zeros((0, 3), np.float32)
                out_rgbs = np.zeros((0,), np.float32)

        # === FREEZE: once ready, lock current last_good and stop updating forever ===
        if (not self.frozen) and self.freeze_on_ready:
            ready, n = self._overhang_ready()
            if ready:
                self.frozen = True
                self.frozen_pts = self.last_good_pts.copy()
                self.frozen_rgbs = self.last_good_rgbs.copy()
                rospy.logwarn(f"[FREEZE] Overhang frozen! pts={self.frozen_pts.shape[0]} (threshold={self.min_ready_overhang_pts})")

                # Publish frozen version immediately to ensure consistency right away
                self.pub_overhang.publish(xyzrgb_to_pc2(self.frozen_pts, self.frozen_rgbs, frame_id, stamp))
                return

        # Publish the chosen output (current or last_good or empty)
        self.pub_overhang.publish(xyzrgb_to_pc2(out_pts, out_rgbs, frame_id, stamp))

    # --------- nearest neighbor check helpers ----------

    def _nn_inside(self, P: np.ndarray):
        """
        Nearest-neighbor distance query from point P to the active overhang cloud.

        Returns:
            inside (bool): True if nearest distance <= nn_radius
            dmin (float): nearest distance
        """
        cloud = self._active_cloud()
        if cloud is None or cloud.shape[0] == 0:
            return False, float("inf")
        diff = cloud - P[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        dmin = float(np.sqrt(np.min(d2)))
        return (dmin <= self.nn_radius), dmin

    # --------- RUN CHECK (UPDATED: +x then fallback -x) ----------

    def _run_check_and_publish(self, msg: PointStamped):
        """
        Given a check point, search along +X direction in steps to find a point
        that is within nn_radius of the overhang cloud.
        If +X fails, search along -X.

        When found:
          - publish /if_in_plate_area = True
          - publish /updated_grasp_point = found point
        Otherwise:
          - publish /if_in_plate_area = False
        """
        P0 = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)

        def _search_along(sign: float, label: str):
            """
            Search along X with sign (+1 or -1).
            """
            P = P0.copy()
            for k in range(self.max_steps + 1):
                inside, dmin = self._nn_inside(P)
                if inside:
                    # Found a point near overhang cloud
                    self.pub_if.publish(Bool(data=True))

                    out = PointStamped()
                    out.header = msg.header
                    out.point.x = float(P[0])
                    out.point.y = float(P[1])
                    out.point.z = float(P[2])
                    self.pub_updated.publish(out)

                    rospy.loginfo(
                        f"[AUTO-CHECK] FOUND dir={label} step={k}  "
                        f"P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f})  "
                        f"nearest={dmin:.4f} <= {self.nn_radius:.4f}  "
                        f"=> TRUE (published /updated_grasp_point)"
                    )
                    return True

                # Not inside yet
                self.pub_if.publish(Bool(data=False))
                rospy.loginfo(
                    f"[AUTO-CHECK] dir={label} step={k}  "
                    f"P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f})  "
                    f"nearest={dmin:.4f} > {self.nn_radius:.4f}  => FALSE"
                )

                # Move along X axis
                P[0] += sign * self.step_x
            return False

        # 1) Try +X first
        if _search_along(+1.0, "+x"):
            return True

        rospy.logwarn(
            f"[AUTO-CHECK] +x NOT FOUND within max_steps={self.max_steps}. Fallback to -x search..."
        )

        # 2) Fallback: try -X
        if _search_along(-1.0, "-x"):
            return True

        # 3) Never found
        self.pub_if.publish(Bool(data=False))
        rospy.logwarn(
            f"[AUTO-CHECK] NOT FOUND in both +x and -x within max_steps={self.max_steps}. "
            f"start=({P0[0]:.3f},{P0[1]:.3f},{P0[2]:.3f}), step_x={self.step_x}"
        )
        return False

    # --------- UPDATED: pending if not ready ----------

    def cb_check_point(self, msg: PointStamped):
        """
        Handle incoming check point.

        If overhang is not "ready":
          - store this point as pending and wait in timer callback
        If ready:
          - run NN search immediately and publish results
        """
        # Frame sanity: check point should be in the same frame as table/overhang
        if self.table_frame is not None and msg.header.frame_id and msg.header.frame_id != self.table_frame:
            rospy.logwarn_throttle(
                2.0,
                f"[AUTO-CHECK] frame mismatch: point={msg.header.frame_id}, overhang/table={self.table_frame}. "
                f"Please publish check point in {self.table_frame}."
            )
            return

        ready, n = self._overhang_ready()
        if not ready:
            # Store pending point and wait for readiness
            self.pending_check_msg = msg
            self.pending_start_time = rospy.Time.now()
            self.pub_if.publish(Bool(data=False))
            rospy.logwarn(
                f"[AUTO-CHECK] overhang NOT ready yet (last_good_pts={n}, need>={self.min_ready_overhang_pts}). "
                f"Stored pending point, will auto-run when ready."
            )
            return

        # Already ready -> run immediately
        self._run_check_and_publish(msg)

    # --------- TIMER: print ready info + run pending when ready ----------

    def cb_timer(self, event):
        """
        Periodic timer callback:
          (1) Print readiness status at a throttled rate
          (2) If a pending check point exists:
              - wait until ready then run it
              - drop it if pending_timeout is exceeded
        """
        now = rospy.Time.now()

        # (1) Print ready info periodically
        if (now - self._last_ready_print).to_sec() >= self.ready_print_period:
            self._last_ready_print = now
            ready, n = self._overhang_ready()
            pending = self.pending_check_msg is not None
            rospy.loginfo(
                f"[OVERHANG-READY] ready={ready}  frozen={self.frozen}  last_good_pts={n}  "
                f"threshold={self.min_ready_overhang_pts}  pending={pending}"
            )

        # (2) No pending message -> nothing to do
        if self.pending_check_msg is None:
            return

        # Pending timeout -> drop the pending request
        if (now - self.pending_start_time).to_sec() > self.pending_timeout:
            rospy.logwarn(
                f"[AUTO-CHECK] pending timeout ({self.pending_timeout}s). "
                f"overhang still not ready -> drop pending."
            )
            self.pub_if.publish(Bool(data=False))
            self.pending_check_msg = None
            return

        # Still not ready -> keep waiting
        ready, n = self._overhang_ready()
        if not ready:
            return

        # Ready now -> run pending request once
        msg = self.pending_check_msg
        self.pending_check_msg = None
        rospy.loginfo(
            f"[AUTO-CHECK] overhang ready now (last_good_pts={n}). Running pending check..."
        )
        self._run_check_and_publish(msg)

if __name__ == "__main__":
    # Create node instance and spin for callbacks
    PlateOverhangNode()
    rospy.spin()
