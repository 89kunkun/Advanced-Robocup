#include <plane_segmentation/plane_segmentation.h>

PlaneSegmentation::PlaneSegmentation(
    const std::string& pointcloud_topic, 
    const std::string& base_frame) :
  pointcloud_topic_(pointcloud_topic),
  base_frame_(base_frame),
  is_cloud_updated_(false)
{
}

PlaneSegmentation::~PlaneSegmentation()
{
}

bool PlaneSegmentation::initalize(ros::NodeHandle& nh)
{
  // load rosparams
  std::vector<float> pre_pass_limits;
  std::vector<float> seg_pass_limits;
  if (!ros::param::get("pre_pass_filter", pre_pass_limits))
  {
    return false;
  }
  if (!ros::param::get("seg_pass_filter", seg_pass_limits))
  {
    return false;
  }
  if (!ros::param::get("ransac_threshold", ransac_thresh_))
  {
    return false;
  }
  pre_pass_low_ = pre_pass_limits[0];
  pre_pass_high_ = pre_pass_limits[1];
  seg_pass_low_ = seg_pass_limits[0];
  seg_pass_high_ = seg_pass_limits[1];

  //#>>>>TODO: subscribe to the pointcloud_topic_ and link it to the right callback
  point_cloud_sub_ = nh.subscribe(pointcloud_topic_, 10, &PlaneSegmentation::cloudCallback, this);

  //#>>>>TODO: advertise the pointcloud for of the table plane
  plane_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/table_point_cloud", 1);

  //#>>>>TODO: advertise the pointcloud for the remaining points (objects)
  objects_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/objects_point_cloud", 1);

  // Most PCL functions accept pointers as their arguments, as such we first set
  // initalize these pointers, otherwise we will run into segmentation faults...
  raw_cloud_.reset(new PointCloud);
  preprocessed_cloud_.reset(new PointCloud);
  plane_cloud_.reset(new PointCloud);
  objects_cloud_.reset(new PointCloud);

  return true;
}

void PlaneSegmentation::update(const ros::Time& time)
{
  // update as soon as new pointcloud is available
  if(is_cloud_updated_)
  {
    is_cloud_updated_ = false;

    //#>>>>Note: To check preProcessCloud() you can publish its output for testing
    // apply all preprocessing steps
    if(!preProcessCloud(raw_cloud_, preprocessed_cloud_))
      return;

    // segment cloud into table and objects
    if(!segmentCloud(preprocessed_cloud_, plane_cloud_, objects_cloud_))
      return;

    //#>>>>TODO: publish both pointclouds obtained by segmentCloud()

    sensor_msgs::PointCloud2 plane_cloud_msg;
    sensor_msgs::PointCloud2 objects_cloud_msg;
    // plane_cloud_msg.header.frame_id = base_frame_;
    // objects_cloud_msg.header.frame_id = base_frame_;
    pcl::toROSMsg(*plane_cloud_, plane_cloud_msg);
    pcl::toROSMsg(*objects_cloud_, objects_cloud_msg);

    plane_cloud_pub_.publish(plane_cloud_msg);
    objects_cloud_pub_.publish(objects_cloud_msg);
  }
}

bool PlaneSegmentation::preProcessCloud(CloudPtr& input, CloudPtr& output)
{
  //#>>>>Goal: Subsample and Filter the pointcloud

  //#>>>>Note: Raw pointclouds are typically to dense and need to be made sparse
  //#>>>>TODO: Down sample the pointcloud using VoxelGrid, save result in ds_cloud
  //#>>>>Hint: See https://pcl.readthedocs.io/projects/tutorials/en/master/voxel_grid.html#voxelgrid 
  //#>>>>TODO: Set useful parameters

  CloudPtr ds_cloud(new PointCloud);            // downsampled pointcloud
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(input);
  voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
  voxel_filter.filter(*ds_cloud);

  
  //#>>>>Note: Its allways a good idea to get rid of useless points first (e.g. floor, ceiling, walls, etc.)
  //#>>>>TODO: Transform the point cloud to the base_frame of the robot. (A frame with z=0 at ground level)
  //#>>>>TODO: Transform the point cloud to the base_frame and store the result in transf_cloud
  //#>>>>Hint: use pcl_ros::transformPointCloud

  CloudPtr transf_cloud(new PointCloud);        // transformed pointcloud (expressed in base frame)

  ds_cloud->header = input->header;

  tfListener_.waitForTransform(base_frame_, ds_cloud->header.frame_id, ros::Time(0), ros::Duration(1.0));
  
  
  // Transform the point cloud to the base_frame link.
  pcl_ros::transformPointCloud(base_frame_, ros::Time(0), *ds_cloud, ds_cloud->header.frame_id, *transf_cloud, tfListener_);

  //#>>>>TODO: Trim points lower than some z_min to remove the floor from the point cloud.
  //#>>>>Hint: use pcl::PassThrough filter and save result in output
  //#>>>>Hint: https://pcl.readthedocs.io/projects/tutorials/en/master/passthrough.html#passthrough

  // creat pass through filtering object
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(transf_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(pre_pass_low_, pre_pass_high_);
  pass.filter(*output);
  
  // ROS_INFO_STREAM("Preprocessed cloud size: " << output->size());

  return true;
}

bool PlaneSegmentation::segmentCloud(CloudPtr& input, CloudPtr& plane_cloud, CloudPtr& objects_cloud)
{
  //#>>>>Goal: Remove every point that is not an object from the objects_cloud cloud

// 1. Segment the plane using RANSAC
  // We will use Ransac to segment the pointcloud, here we setup the objects we need for this
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  //#>>>>TODO: set parameters of the SACS segmentation
  //#>>>>TODO: set correct model, play with DistanceThreshold and the free outlier probability
  //#>>>>TODO: then segment the input point cloud
  //#>>>>Note: Checkout the pcl tutorials on plane segmentation

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(ransac_thresh_);
  seg.setProbability(0.99);
  seg.setInputCloud(input);
  seg.segment(*inliers, *coefficients);

  // // Debug
  // if (inliers->indices.empty()){
  //   ROS_WARN("Plane segmentation failed. No inliers found.");
  //   return false;
  // }

// 2. Extract plane points  
  //#>>>>TODO: save inliers in plane_cloud 
  //#>>>>Note: These sould be point that belong to the table 
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(input);
  extract.setIndices(inliers);

  extract.setNegative(false); // 提取 inliers
  extract.filter(*plane_cloud);

// 3. Extract non-plane points (objects)
  //#>>>>TODO: save outliers in the objects_cloud
  //#>>>>Note: This should be the rest
  extract.setNegative(true); // 提取 outliners
  extract.filter(*objects_cloud);

  // Next, we further refine the the objects_cloud by transforming it into the coordinate frame
  // of the fitted plane. In this transformed frame we remove everything below the table plane and 
  // everything more than 20 cm above the table.
  // Basically, a table aligned bounding box

// 4. Convert the plane parameters to the normal vector n and distance d
  // if the plane fit is correct it will result in the coefficients = [nx, ny, nz, d]
  // where n = [nx, ny, nz] is the 3d normal vector perpendicular to the plane
  // and d the distance to the origin

  if(coefficients->values.size() < 4)
    return false;

  //#>>>>TODO: extract the normal vector 'n' perpendicular to the plane and the scalar distance 'd'
  //#>>>>TODO: to the origin from the plane coefficions.
  //#>>>>Note: As always we use Eigen to represent vectors and matices 
  //#>>>>Note: https://eigen.tuxfamily.org/dox/GettingStarted.html
  Eigen::Vector3f n(coefficients->values[0],
                    coefficients->values[1],
                    coefficients->values[2]); // = ?
  double d = coefficients->values[3]; // = ?
  
  // Now we construct an Eigen::Affine3f transformation T_plane_base that describes the table plane 
  // with respect to the base_link frame using n and d

  //#>>>>TODO: Build the Rotation (Quaterion) from the table's normal vector n
  //#>>>>TODO: And the floor (world) normal vector: [0,0,1]
  //#>>>>Hint: Use Eigen::Quaternionf::FromTwoVectors()

// 5. Construct translation
  Eigen::Vector3f z_axis(0.0, 0.0, 1.0);
  Eigen::Quaternionf Q_plane_base = Eigen::Quaternionf::FromTwoVectors(n, z_axis); // = ?

// 6. Construct rotation
  Eigen::Vector3f t_plane_base = d * n; // = ?

// 7. Create the affine transformation T_plane_base
  // Finally we create the Homogenous transformation of the table
  Eigen::Affine3f T_plane_base = Eigen::Affine3f::Identity();
  T_plane_base.rotate(Q_plane_base.toRotationMatrix());
  T_plane_base.translate(t_plane_base);

// 8. Transform the objects_cloud to the plane-aligned coordinate frame
  //#>>>>TODO: Transform the objects_cloud into the table frame and store in transf_cloud
  //#>>>>Hint: Use the function pcl::transformPointCloud() with T_plane_base as input
  //#>>>>https://pcl.readthedocs.io/projects/tutorials/en/latest/matrix_transform.html
  CloudPtr transf_cloud(new PointCloud);
  pcl::transformPointCloud(*objects_cloud, *transf_cloud, T_plane_base);

// 9. Filter objects using PassThrough
  //#>>>>TODO: filter everything directly below the table and above it (z > 0.01 && < 0.15) 
  //#>>>>Hint: using pcl::PassThrough filter (same as before)
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(transf_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(seg_pass_low_, seg_pass_high_);
  
  CloudPtr filterd_cloud(new PointCloud);
  pass.filter(*filterd_cloud);

// 10.Transform the object points back to the original base_link coordinate frame
  //#>>>>TODO: transform back to base_link frame using the inverse transformation
  //#>>>>TODO: and store result in objects_cloud. Object cloud should only contain points associated to objects
  //#>>>>Hint: Eigen::Affine3f has an inverse function
  pcl::transformPointCloud(*filterd_cloud, *objects_cloud, T_plane_base.inverse());

  return true;
}

void PlaneSegmentation::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  // convert ros msg to pcl raw_cloud
  is_cloud_updated_ = true;

  //#>>>>TODO: Convert the msg to the internal variable raw_cloud_ that holds the raw input pointcloud 
  //#>>>>Hint: pcl::fromROSMsg() can do the job
  pcl::fromROSMsg(*msg, *raw_cloud_);
}
