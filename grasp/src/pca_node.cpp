#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>

#include <Eigen/Dense>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <grasp/pca_utils.h>

class PCANode
{
public:
  explicit PCANode(ros::NodeHandle& nh, ros::NodeHandle& nh_global)
  {
    // ====== 原有参数 ======
    nh.param<std::string>("input_topic", input_topic_, std::string("/cloud_labeled"));
    nh.param("target_label", target_label_, 1);
    nh.param("run_once", run_once_, false);

    nh.param("axis_scale", axis_scale_, 0.25);
    nh.param("axis_radius", axis_radius_, 0.01);

    nh.param("vertical_cos_threshold", vertical_cos_threshold_, 0.8);
    nh.param("horizontal_cos_threshold", horizontal_cos_threshold_, 0.4);

    // ====== 新增：desired_food topic（做法1）======
    // ✅ 这行是关键：强制订阅全局 /desired_food（不进 ~ 命名空间）
    nh.param<std::string>("desired_food_topic", desired_food_topic_, std::string("/desired_food"));

    // ====== 新增：名称->label 映射（与你 object_labeling 的 dict 一致）======
    // 你 object_labeling 里写的是：
    // milk=1, cola=2, sprite=5
    // 这里保持一致
    food_to_label_["milk"]   = 1;
    food_to_label_["cola"]   = 2;
    food_to_label_["sprite"] = 5;

    // ====== 订阅点云 ======
    sub_ = nh.subscribe(input_topic_, 1, &PCANode::cb, this);

    // ====== RViz MarkerArray ======
    marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("pca_axes", 1, true);

    // ====== 新增：发布 3x3 PCA 轴矩阵 ======
    // ✅ 发布到全局 /calculated_pca_axis
    pca_axis_pub_ = nh_global.advertise<std_msgs::Float64MultiArray>("/calculated_pca_axis", 1, true);

    // ====== 新增：订阅 /desired_food（全局）======
    // ✅ 用 nh_global 确保订阅的是 /desired_food 而不是 /pca_node/desired_food
    desired_food_sub_ = nh_global.subscribe(desired_food_topic_, 1, &PCANode::desiredFoodCb, this);

    ROS_INFO("pca_node listening: %s (target_label=%d)", input_topic_.c_str(), target_label_);
    ROS_INFO("vertical_cos_threshold=%.3f, horizontal_cos_threshold=%.3f",
             vertical_cos_threshold_, horizontal_cos_threshold_);
    ROS_INFO("Subscribing desired_food: %s", desired_food_topic_.c_str());
    ROS_INFO("Publishing /calculated_pca_axis as Float64MultiArray (len=9, row-major 3x3)");
  }

private:
  // =========================
  //  NEW: desired_food 回调
  // =========================
  void desiredFoodCb(const std_msgs::StringConstPtr& msg)
  {
    std::string food = msg->data;

    // 去掉可能的空格/换行
    trim(food);

    auto it = food_to_label_.find(food);
    if (it == food_to_label_.end())
    {
      ROS_WARN("desired_food='%s' not in mapping (milk/cola/sprite). Keep target_label=%d",
               food.c_str(), target_label_);
      return;
    }

    int new_label = it->second;

    if (new_label != target_label_)
    {
      ROS_WARN("desired_food='%s' => target_label changes: %d -> %d",
               food.c_str(), target_label_, new_label);
      target_label_ = new_label;

      // 如果你 run_once=true，但希望切换目标后还能再算一次：
      // ✅ 切换目标时允许再次计算
      done_ = false;
    }
  }

  // =========================
  //  点云回调：PCA + 输出 + 发布矩阵
  // =========================
  void cb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    if (run_once_ && done_) return;

    pcl::PointCloud<pcl::PointXYZL> cloud;
    pcl::fromROSMsg(*msg, cloud);
    if (cloud.empty()) return;

    // 按 label 过滤
    std::vector<pcl::PointXYZL> target_pts;
    target_pts.reserve(cloud.size());
    for (const auto& p : cloud.points)
    {
      if ((int)p.label == target_label_)
        target_pts.push_back(p);
    }

    if (target_pts.empty())
    {
      ROS_WARN("No points with label %d", target_label_);
      return;
    }

    PCAResult r;
    if (!computePCAFromLabeledPoints(target_pts, r))
    {
      ROS_ERROR("PCA failed");
      return;
    }

    r.axis_long.normalize();
    r.axis_mid.normalize();
    r.axis_thin.normalize();

    // ====== 规则化坐标系（你之前的规则）======
    const Eigen::Vector3d up(0, 0, 1);
    const Eigen::Vector3d down(0, 0, -1);
    const Eigen::Vector3d left(0, 1, 0);

    Eigen::Vector3d axis_long_pca = r.axis_long.normalized();
    double cos_to_up = std::fabs(axis_long_pca.dot(up));
    bool is_vertical = (cos_to_up > vertical_cos_threshold_);

    Eigen::Vector3d axis_long, axis_mid, axis_thin;

    if (is_vertical)
    {
      axis_long = down;                 // (0,0,-1)
      axis_mid  = left;                 // (0,1,0)
      axis_thin = axis_long.cross(axis_mid); // (-Z)×(+Y) = +X
      axis_thin.normalize();
    }
    else
    {
      axis_thin = down;                 // (0,0,-1)

      axis_long = axis_long_pca;        // 投影到XY
      axis_long.z() = 0.0;
      if (axis_long.norm() < 1e-6)
        axis_long = Eigen::Vector3d(1, 0, 0);
      axis_long.normalize();

      axis_mid = axis_thin.cross(axis_long); // (-Z)×(long_xy)
      if (axis_mid.norm() < 1e-6)
        axis_mid = Eigen::Vector3d(0, 1, 0);
      axis_mid.normalize();
    }

    r.axis_long = axis_long;
    r.axis_mid  = axis_mid;
    r.axis_thin = axis_thin;

    // 尺寸（投影跨度）
    double len_long = 0, len_mid = 0, len_thin = 0;
    computeAxisLengths(target_pts, r, len_long, len_mid, len_thin);

    // 终端输出
    ROS_INFO("------------------------------------------------------------");
    ROS_INFO("Frame: %s", msg->header.frame_id.c_str());
    ROS_INFO("Target label: %d", target_label_);
    ROS_INFO("Centroid: [x=%.3f, y=%.3f, z=%.3f]", r.centroid.x(), r.centroid.y(), r.centroid.z());

    ROS_INFO("Axis LONG (red)  : [x=%.3f, y=%.3f, z=%.3f]", r.axis_long.x(), r.axis_long.y(), r.axis_long.z());
    ROS_INFO("Axis MID  (green): [x=%.3f, y=%.3f, z=%.3f]", r.axis_mid.x(),  r.axis_mid.y(),  r.axis_mid.z());
    ROS_INFO("Axis THIN (blue) : [x=%.3f, y=%.3f, z=%.3f]", r.axis_thin.x(), r.axis_thin.y(), r.axis_thin.z());

    ROS_INFO("Axis lengths (projection span): long=%.3f, mid=%.3f, thin=%.3f",
             len_long, len_mid, len_thin);

    double cos_long_up_final = std::fabs(r.axis_long.dot(up));
    ROS_INFO("cos(angle(long_final, +Z(up))) = %.3f", cos_long_up_final);
    if (cos_long_up_final < horizontal_cos_threshold_)
      ROS_WARN("[JUDGE] HORIZONTAL (lying on table)");
    else
      ROS_WARN("[JUDGE] VERTICAL (standing)");

    // ====== NEW: 发布 3x3 矩阵到 /calculated_pca_axis ======
    publishPcaMatrix(r);

    // RViz marker
    publishMarkers(r, msg->header.frame_id);

    done_ = true;
  }

  // =========================
  //  NEW: 发布矩阵（Float64MultiArray）
  // =========================
  void publishPcaMatrix(const PCAResult& r)
  {
    std_msgs::Float64MultiArray arr;

    // 你要求的 data 顺序（len=9）：
    //
    // data = [
    //   long.x, long.y, long.z,
    //   mid.x , mid.y , mid.z ,
    //   thin.x, thin.y, thin.z
    // ]
    //
    // ✅ 含义（非常重要）：
    // - 第 1 行（data[0..2]） = LONG 轴方向向量（红轴）
    // - 第 2 行（data[3..5]） = MID  轴方向向量（绿轴）
    // - 第 3 行（data[6..8]） = THIN 轴方向向量（蓝轴）
    //
    // 即：3x3 “行主序 row-major”，每一行是一根轴的 (x,y,z)
    //
    arr.data.resize(9);
    arr.data[0] = r.axis_long.x();
    arr.data[1] = r.axis_long.y();
    arr.data[2] = r.axis_long.z();

    arr.data[3] = r.axis_mid.x();
    arr.data[4] = r.axis_mid.y();
    arr.data[5] = r.axis_mid.z();

    arr.data[6] = r.axis_thin.x();
    arr.data[7] = r.axis_thin.y();
    arr.data[8] = r.axis_thin.z();

    pca_axis_pub_.publish(arr);
  }

  static void computeAxisLengths(const std::vector<pcl::PointXYZL>& pts,
                                 const PCAResult& r,
                                 double& len_long,
                                 double& len_mid,
                                 double& len_thin)
  {
    auto proj_span = [&](const Eigen::Vector3d& axis_unit) -> double {
      double min_p =  std::numeric_limits<double>::infinity();
      double max_p = -std::numeric_limits<double>::infinity();
      for (const auto& p : pts)
      {
        Eigen::Vector3d v(p.x, p.y, p.z);
        double t = v.dot(axis_unit);
        min_p = std::min(min_p, t);
        max_p = std::max(max_p, t);
      }
      return max_p - min_p;
    };

    Eigen::Vector3d aL = r.axis_long.normalized();
    Eigen::Vector3d aM = r.axis_mid.normalized();
    Eigen::Vector3d aT = r.axis_thin.normalized();

    len_long = proj_span(aL);
    len_mid  = proj_span(aM);
    len_thin = proj_span(aT);
  }

  void publishMarkers(const PCAResult& r, const std::string& cloud_frame)
  {
    auto make_arrow =
        [&](int id,
            const Eigen::Vector3d& start,
            const Eigen::Vector3d& dir,
            float cr, float cg, float cb)
    {
      visualization_msgs::Marker m;
      m.header.frame_id = cloud_frame;
      m.header.stamp = ros::Time::now();
      m.ns = "pca_axes";
      m.id = id;
      m.type = visualization_msgs::Marker::ARROW;
      m.action = visualization_msgs::Marker::ADD;

      geometry_msgs::Point p0, p1;
      p0.x = start.x(); p0.y = start.y(); p0.z = start.z();
      Eigen::Vector3d end = start + dir.normalized() * axis_scale_;
      p1.x = end.x();   p1.y = end.y();   p1.z = end.z();

      m.points.push_back(p0);
      m.points.push_back(p1);

      m.scale.x = axis_radius_;
      m.scale.y = axis_radius_ * 2.0;
      m.scale.z = axis_radius_ * 3.0;

      m.color.a = 1.0;
      m.color.r = cr; m.color.g = cg; m.color.b = cb;

      m.lifetime = ros::Duration(0);
      return m;
    };

    visualization_msgs::MarkerArray arr_m;
    Eigen::Vector3d c = r.centroid;

    arr_m.markers.push_back(make_arrow(0, c, r.axis_long, 1.0f, 0.0f, 0.0f));
    arr_m.markers.push_back(make_arrow(1, c, r.axis_mid,  0.0f, 1.0f, 0.0f));
    arr_m.markers.push_back(make_arrow(2, c, r.axis_thin, 0.0f, 0.0f, 1.0f));

    marker_pub_.publish(arr_m);
  }

  // 简单 trim（去掉字符串两边空白）
  static void trim(std::string& s)
  {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  }

private:
  ros::Subscriber sub_;
  ros::Publisher marker_pub_;

  // NEW
  ros::Subscriber desired_food_sub_;
  ros::Publisher pca_axis_pub_;

  std::string input_topic_;
  int target_label_{1};
  bool run_once_{false};
  bool done_{false};

  double axis_scale_{0.25};
  double axis_radius_{0.01};

  double vertical_cos_threshold_{0.8};
  double horizontal_cos_threshold_{0.4};

  // NEW
  std::string desired_food_topic_{"/desired_food"};
  std::unordered_map<std::string, int> food_to_label_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pca_node");

  // 私有参数用 ~
  ros::NodeHandle nh("~");
  // 全局句柄（订阅/发布全局 topic）
  ros::NodeHandle nh_global;

  PCANode node(nh, nh_global);
  ros::spin();
  return 0;
}

