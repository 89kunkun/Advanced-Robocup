#include <object_labeling/object_labeling.h>

#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

/* =========================
 *  Helper: crop point cloud with YOLO bbox
 * ========================= */
static void cropPointCloudWithBBox(
  const ObjectLabeling::CloudPtr& input_cloud,
  const yolo_v8_detector::BoundingBox& bbox,
  const Eigen::Matrix3d& K,
  const Eigen::Affine3d& T_camera_base,
  ObjectLabeling::CloudPtr& output_cloud)
{
  output_cloud->clear();
  output_cloud->header = input_cloud->header;

  for (const auto& p : input_cloud->points)
  {
    Eigen::Vector3d point_base(p.x, p.y, p.z);

    // base -> camera
    Eigen::Vector3d point_cam = T_camera_base * point_base;
    if (point_cam.z() <= 0.0)
      continue; // point is behind the camera

    // project to image plane
    Eigen::Vector3d uvw = K * point_cam;
    double u = uvw.x() / uvw.z();
    double v = uvw.y() / uvw.z();

    // inside bbox
    if (u >= bbox.xmin && u <= bbox.xmax &&
        v >= bbox.ymin && v <= bbox.ymax)
    {
      output_cloud->points.push_back(p);
    }
  }
}

/* =========================
 *  ObjectLabeling
 * ========================= */

ObjectLabeling::ObjectLabeling(
    const std::string& objects_cloud_topic_, 
    const std::string& camera_info_topic,
    const std::string& camera_frame) :
  is_cloud_updated_(false),
  has_camera_info_(false),
  objects_cloud_topic_(objects_cloud_topic_),
  camera_info_topic_(camera_info_topic),
  camera_frame_(camera_frame),
  K_(Eigen::Matrix3d::Zero())
{
}

ObjectLabeling::~ObjectLabeling()
{
}

bool ObjectLabeling::initalize(ros::NodeHandle& nh)
{
  // Subscribe to objects pointcloud published by the plane_segmentation_node
  object_point_cloud_sub_ = nh.subscribe(objects_cloud_topic_, 10, &ObjectLabeling::cloudCallback, this);
  // Subscribe to bounding boxes from YOLOv8
  object_detections_sub_ = nh.subscribe("/yolo_v8_detector/bounding_boxes", 10, &ObjectLabeling::detectionCallback, this);
  // Subscribe to camera info from robot to obtain the camera matrix K
  camera_info_sub_ = nh.subscribe(camera_info_topic_, 10, &ObjectLabeling::cameraInfoCallback, this);
  // Subscribe to table point cloud to compute table center
  table_cloud_sub_ = nh.subscribe("/table_point_cloud", 10, &ObjectLabeling::tableCloudCallback, this);

  // Publish the labled objects as PointCloudl type (see typedefs in header)
  labeled_object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/labeled_object_point_cloud", 1);
  // Publish the LABELED object names as visulaization marker (http://wiki.ros.org/rviz/DisplayTypes/Marker)
  text_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/text_markers", 1);
  centroid_pub_ = nh.advertise<geometry_msgs::PointStamped>("cluster_centroid", 1);
  // Publish the table center as PointStamped
  table_center_pub_ = nh.advertise<geometry_msgs::PointStamped>("/table_center", 1);
  table_center_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/table_center_marker", 1);
  // Publish plate point cloud and center
  plate_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/plate_point_cloud", 1);
  plate_center_pub_ = nh.advertise<geometry_msgs::PointStamped>("/plate_center", 1);
  
  // init internal pointclouds for processing (again pcl uses pointers)
  object_point_cloud_.reset(new PointCloud);    // holds unlabled object point cloud
  labeled_point_cloud_.reset(new PointCloudl);  // holds labled object point cloud
  table_cloud_.reset(new PointCloud); // holds table point cloud
  plate_cloud_.reset(new PointCloud); // holds plate point cloud

  // Setup a mapping from class names the ones given by yolo (see yolo bounding_boxes message for classes)
  //#>>>>Note: We will use 0 as 'unkown' type
  dict_["bottle"] = 1;
  dict_["cup"] = 2;
  dict_["laptop"] = 3;
  dict_["person"] = 5;

  return true;
}

void ObjectLabeling::update(const ros::Time& time)
{
  // camera info and point cloud available
  if(is_cloud_updated_ && has_camera_info_)
  {
    is_cloud_updated_ = false;

    // label the objects in pointcloud based on 2d bounding boxes 
    if(!labelObjects(object_point_cloud_, labeled_point_cloud_))
    {
      ROS_WARN("Object labeling failed.");
      return;
    }
    
    // Publish labeled_point_cloud_ to ros
    sensor_msgs::PointCloud2 labeled_point_cloud_msg;
    pcl::toROSMsg(*labeled_point_cloud_, labeled_point_cloud_msg);
    labeled_point_cloud_msg.header.frame_id = object_point_cloud_->header.frame_id;
    labeled_point_cloud_msg.header.stamp = time;
    labeled_object_cloud_pub_.publish(labeled_point_cloud_msg);
    
    // Publish plate_cloud_ and plate_center_ to ros
    if (plate_cloud_ && !plate_cloud_->empty())
    {
      sensor_msgs::PointCloud2 plate_msg;
      pcl::toROSMsg(*plate_cloud_, plate_msg);
      plate_msg.header.frame_id = object_point_cloud_->header.frame_id;
      plate_msg.header.stamp = time;
      plate_cloud_pub_.publish(plate_msg);
    }

    if (has_plate_center_)
    {
      plate_center_.header.stamp = time;
      plate_center_pub_.publish(plate_center_);
    }

    // Publish text_markers_ to ros
    for (auto& marker : text_markers_.markers) 
    {
      marker.header.frame_id = object_point_cloud_->header.frame_id;
      marker.header.stamp = time;
    }
    text_marker_pub_.publish(text_markers_);

  }
}

bool ObjectLabeling::labelObjects(CloudPtr& input, CloudPtrl& output)
{
  //#>>>>GOAL: Split input pointcloud into seperate blobs, compute centroid,
  //#>>>>GOAL: project centorid into the camera image and match with bounding box
  //#>>>>GOAL: finally label the pointcloud with object type

// 1. Euclidean clustering
  // First we need to cluster the input cloud into seperated clusters,
  // each of them represents an object on the table.

  //#>>>>TODO: Use EuclideanClusterExtraction to seperate the pointcloud into clusters
  //#>>>>Hint: https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html?highlight=EuclideanClusterExtraction

  if (!input || input->empty()) {
    ROS_WARN("Input cloud is null or empty!");
    return false;
  }

  plate_cloud_->clear();

  /* ============================================================
  * SEMANTIC RECOVERY: recover plate from table cloud
  * ============================================================ */
  if (table_cloud_ && !table_cloud_->empty() && !detections_.empty())
  {
    tf::StampedTransform tf_base_camera;
    tfListener_.lookupTransform("base_footprint", camera_frame_, ros::Time(0), tf_base_camera);

    Eigen::Affine3d T_base_camera;
    tf::transformTFToEigen(tf_base_camera, T_base_camera);
    Eigen::Affine3d T_camera_base = T_base_camera.inverse();

    for (const auto& box : detections_)
    {
      if (box.Class == "laptop")
      {
        CloudPtr recovered(new PointCloud);

        cropPointCloudWithBBox(
          table_cloud_, box, K_, T_camera_base, recovered);

        if (!recovered->empty())
        {
          ROS_INFO("[ObjectLabeling] Recovered plate point cloud with %zu points", recovered->size());

          // Append plate_cloud to input cloud
          *plate_cloud_ += *recovered;
          *input += *recovered;
        }
      }
    }
  }

  /* ============================================================
  * Compute plate center (3D centroid)
  * ============================================================ */
  if (plate_cloud_ && !plate_cloud_->empty())
  {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*plate_cloud_, centroid);

    plate_center_.header.frame_id = input->header.frame_id;
    plate_center_.header.stamp = ros::Time::now();
    plate_center_.point.x = centroid[0];
    plate_center_.point.y = centroid[1];
    plate_center_.point.z = centroid[2];

    has_plate_center_ = true;

    ROS_INFO("[ObjectLabeling] Plate center at (%.2f, %.2f, %.2f)", 
              centroid[0], centroid[1], centroid[2]);
  }
  else
  {
    has_plate_center_ = false;
  }

  /* ========== Euclidean clustering ========== */
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(input);
  
  // holds the extracted cluster indices (just a integer for identifiction)
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(0.02);
  ec.setMinClusterSize(80);
  ec.setMaxClusterSize(25000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(input);
  ec.extract(cluster_indices);

  if (cluster_indices.empty()) {
    ROS_WARN("Euclidean clustering found no clusters!");
    return false;
  }

// 2. Calculate the centroid of each cluster
  //#>>>>TODO: Iterate over each cluster and compute its centroid point (= mean)
  //#>>>>TODO: Push the centroid into the vector of centroids
  std::vector<Eigen::Vector3d> centroids;
  for (const auto& indices : cluster_indices)
  {
    pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
    for (const auto& idx : indices.indices)
    {
      cloud_cluster->push_back(input->points[idx]);
    }
    Eigen::Vector4d centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    centroids.push_back(centroid.head<3>());

    ROS_INFO("[ObjectLabeling] Cluster centroid at (%.2f, %.2f, %.2f)", 
              centroid[0], centroid[1], centroid[2]);

    // Publish the centroid
    geometry_msgs::PointStamped centroid_msg;
    centroid_msg.header.frame_id = "base_footprint";
    centroid_msg.header.stamp = ros::Time::now();
    centroid_msg.point.x = centroid[0];
    centroid_msg.point.y = centroid[1];
    centroid_msg.point.z = centroid[2];

    centroid_pub_.publish(centroid_msg);
  }

  // Next we need to find the pixel coordinates of the centroids within the 2d
  // camera image. This projection is handled by the camera matrix
  // First, the centorids need to be transformed from the pointcloud frame into the
  // camera frame.

// 3. TF: base_link -> camera_frame
  //#>>>>TODO: Get the homogenous transformation matrix of the base frame with respect
  //#>>>>TODO: to the camera frame.
  //#>>>>Hint: look up the transformation through the tf tree: tfListener_.lookupTransform(...)
  //#>>>>TODO: Convert the tf::StampedTransform into an Eigen::Affine3d
  //#>>>>Hint: tf::transformTFToEigen(...) can do the job

  Eigen::Affine3d T_base_camera;
  tf::StampedTransform transform;
  tfListener_.lookupTransform("base_footprint", camera_frame_, ros::Time(0), transform);
  tf::transformTFToEigen(transform, T_base_camera);

// 4. Transform the centroids to the camera coordinate frame
  //#>>>>TODO: Transform the centorids into the camera frame by multiplying them 
  //#>>>>TODO: with the transformation that takes a point in the pointcloud frame and turns it
  //#>>>>TODO: into a point in the camera frame. Do this for all centroids.

  std::vector<Eigen::Vector3d> centroids_camera;
  Eigen::Affine3d T_camera_base = T_base_camera.inverse();
  for (const auto& c : centroids)
    centroids_camera.push_back(T_camera_base * c);

// 5. Camera projection: 3D -> 2D
  //#>>>>TODO: Project the transformed centorids into the camera plane by using the camera matrix K
  //#>>>>Hint: Multiplying a 3d vector with the 3x3 camera matrix gives a vector in R^3
  //#>>>>Hint: To get pixel coordinates in R^2 you need to convert them to homogenous 2d coordinates
  //#>>>>Hint: ( = divide by the last component and drop the one in third component.)
  std::vector<Eigen::Vector2d> pixel_centroids;
  for (const auto& c : centroids_camera)
  {
    Eigen::Vector3d homegenous_uvw = K_ * c;
    homegenous_uvw /= homegenous_uvw[2];
    pixel_centroids.emplace_back(homegenous_uvw[0], homegenous_uvw[1]);
  }

  // Now the centorids of each cluster are given as pixel coordinates in the 2d image
  // plane of the camera. What remains is to find the bounding box that matches to each of 
  // those controids.

// 6. Assign each detection box to the nearest centroid
  //#>>>>TODO: Find the best match between pixel_centroids and detections_
  //#>>>>Hint: Use the euclidian distance between the pixel_centroids and the boundingbox centers
  //#>>>>Hint: For each bounding box find the closest cenroid 
  //#>>>>TODO: If a cluster cant be matched (no bounding boxes left) assign 0 as label

  std::vector<int> assigned_labels(cluster_indices.size(), 0);                  // lables of each centroid
  std::vector<std::string> assigned_classes(cluster_indices.size(), "unknown"); // class names of each centroid

  for(size_t i = 0; i < detections_.size(); ++i)
  {
    // get the bounding box we want to find the closest cenroid 
    const yolo_v8_detector::BoundingBox& bounding_box = detections_[i];

    //#>>>>TODO: For all cenroids compute the distance to the boudning box
    //#>>>>TODO: select the clostes as match and get its index in pixel_centroids
    int match; // = ?
    double closest_distance = std::numeric_limits<double>::max();
    Eigen::Vector2d bounding_box_centroid{
      (bounding_box.xmax + bounding_box.xmin)/2.0,
      (bounding_box.ymax + bounding_box.ymin)/2.0
    };
    for (size_t j = 0; j < pixel_centroids.size(); j++)
    {
      double distance = (pixel_centroids[j] - bounding_box_centroid).norm();
      if (distance < closest_distance)
      {
        closest_distance = distance;
        match = j;
      }
    }

    // remember the label of match
    if(dict_.find(bounding_box.Class) != dict_.end())
    {
      assigned_labels[match] = dict_[bounding_box.Class]; // set match to defined class index
      assigned_classes[match] = bounding_box.Class;       // set match to class name
    }
  }

// 7. Reconstruct and build the labeled point cloud
  // relabel the point cloud
  output->points.clear();
  output->header = input->header;

  PointTl pt;
  size_t i = 0;
  auto cit = cluster_indices.begin();
  for(; cit != cluster_indices.end(); ++cit, ++i ) 
  {
    // relabel all the points inside cluster
    std::vector<int>::const_iterator it = cit->indices.begin();
    for(; it != cit->indices.end(); ++it ) 
    {
      PointT& cpt = input->points[*it];
      pt.x = cpt.x;
      pt.y = cpt.y;
      pt.z = cpt.z;
      pt.label = assigned_labels[i];    // Note: To test clustering without the matching stuff just use pt.lable = i
      output->points.push_back( pt );
    }
  }

// 8. Generate text annotation markers
  // create a text marker that displays the assigned class name (assigned_classes) 
  // at the 3d position of the corresponding centroid
  text_markers_.markers.resize(assigned_classes.size());
  for(size_t i = 0; i < assigned_classes.size(); ++i)
  {
    visualization_msgs::Marker marker;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.text = assigned_classes[i];
    marker.pose.position.x = centroids[i].x();
    marker.pose.position.y = centroids[i].y();
    marker.pose.position.z = centroids[i].z() + 0.1;
    marker.color.a = 1.0;
    marker.scale.z = 0.1;
    marker.id = i;
    marker.header.frame_id = input->header.frame_id;
    marker.header.stamp = ros::Time::now();
    text_markers_.markers[i] = marker;
  }

  return true;
}


void ObjectLabeling::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  // convert to pcl
  is_cloud_updated_ = true;
  //#>>>>TODO: convert to pcl and store in object_point_cloud_
  //#>>>>Hint: pcl::fromROSMsg()
  pcl::fromROSMsg(*msg, *object_point_cloud_);
}

void ObjectLabeling::detectionCallback(const yolo_v8_detector::BoundingBoxesConstPtr &msg)
{
  //#>>>>TODO: copy the YOLO bounding boxes
  detections_ = msg->bounding_boxes;

  // Debug
  // for (const auto& box : detections_) {
  //   ROS_INFO_STREAM("Detected class: " << box.Class);
  // }
}

void ObjectLabeling::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg)
{
  // copy camera info
  has_camera_info_ = true;
  Eigen::Matrix3d K = Eigen::Matrix3d::Zero();

  //#>>>>TODO: copy the 3x3 camera matrix to K_
  //#>>>>Hint: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
  for(size_t i = 0; i < 9; ++i)
  {
    K(i) = msg->K[i];
  }
  K_ = K.transpose();

  // K_ << msg->K[0], msg->K[1], msg->K[2],
  //     msg->K[3], msg->K[4], msg->K[5],
  //     msg->K[6], msg->K[7], msg->K[8];
}

void ObjectLabeling::tableCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  CloudPtr cloud(new PointCloud);
  pcl::fromROSMsg(*msg, *cloud);
  if (cloud->empty()) {
    ROS_WARN("Received empty table point cloud!");
    return;
  } 
  table_cloud_ = cloud;

  // 1. Compute AABB min/max
  PointT min_pt, max_pt;
  pcl::getMinMax3D(*table_cloud_, min_pt, max_pt);

  double center_x = (min_pt.x + max_pt.x) / 2.0;
  double center_y = (min_pt.y + max_pt.y) / 2.0;
  double center_z = max_pt.z;

  // 2. Publish table center
  table_center_.header.frame_id = msg->header.frame_id;
  table_center_.header.stamp = ros::Time::now();
  table_center_.point.x = center_x;
  table_center_.point.y = center_y;
  table_center_.point.z = center_z;

  table_center_pub_.publish(table_center_);
  has_table_center_ = true;

  ROS_INFO("[ObjectLabeling] Table center at (%.2f, %.2f, %.2f)", center_x, center_y, center_z);

  // 3. (Optional) Marker for RViz
  visualization_msgs::Marker mk;
  mk.header = table_center_.header;
  mk.ns = "table_center";
  mk.id = 0;
  mk.type = visualization_msgs::Marker::SPHERE;
  mk.scale.x = mk.scale.y = mk.scale.z = 0.05;
  mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 1.0;
  mk.pose.position = table_center_.point;

  table_center_marker_pub_.publish(mk);
}
