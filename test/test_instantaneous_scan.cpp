#include "preprocess.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {
void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

sensor_msgs::msg::PointCloud2::SharedPtr make_cloud(bool empty = false, bool timed = false)
{
  pcl::PointCloud<velodyne_ros::Point> cloud;
  if (!empty) {
    for (int i = 0; i < 8; ++i) {
      velodyne_ros::Point p{};
      const float angle = -0.1f * i;
      p.x = 2.0f * std::cos(angle);
      p.y = 2.0f * std::sin(angle);
      p.z = 0.0f;
      p.intensity = 1.0f;
      p.ring = 0;
      p.time = timed ? i * 0.001f : 0.0f;
      cloud.push_back(p);
    }
  }
  auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(cloud, *msg);
  msg->header.stamp.sec = 12;
  msg->header.stamp.nanosec = 500000000;
  return msg;
}

Preprocess make_preprocess()
{
  Preprocess pre;
  pre.lidar_type = VELO16;
  pre.time_unit = SEC;
  pre.N_SCANS = 16;
  pre.SCAN_RATE = 10;
  pre.blind = 0.01;
  pre.det_range = 100;
  pre.point_filter_num = 1;
  return pre;
}
}

int main()
{
  auto pre = make_preprocess();
  require(!pre.instantaneous_scan, "default must preserve rotating scans");
  PointCloudXYZI::Ptr out(new PointCloudXYZI);
  pre.instantaneous_scan = true;
  pre.process(make_cloud(), out);
  require(out->size() == 8, "instantaneous normal path must retain every valid point");
  for (const auto & point : *out) require(point.curvature == 0, "instantaneous offset must be zero");
  pre.process(make_cloud(false, true), out);
  for (const auto & point : *out) require(point.curvature == 0, "explicit instantaneous mode overrides input offsets");
  pre.process(make_cloud(true), out);
  require(out->empty(), "empty normal cloud");
  std::deque<PointCloudXYZI::Ptr> cuts;
  std::deque<double> times;
  pre.process_cut_frame_pcl2(make_cloud(), cuts, times, 4, 30);
  require(cuts.size() == 1 && times.size() == 1 && cuts.front()->size() == 8, "instantaneous cut must retain a whole frame");
  require(std::abs(times.front() - 12500.0) < 1e-6, "cut-frame stamp is milliseconds");
  for (const auto & point : *cuts.front()) require(point.curvature == 0, "cut offset must be zero");
  cuts.clear(); times.clear();
  pre.process_cut_frame_pcl2(make_cloud(true), cuts, times, 4, 30);
  require(cuts.empty() && times.empty(), "empty cut cloud must produce no frames");
  pre.instantaneous_scan = false;
  pre.process(make_cloud(), out);
  require(!out->empty() && out->back().curvature > 0, "legacy rotating fallback must remain active");
  pre.process(make_cloud(false, true), out);
  require(out->size() == 8 && std::abs(out->back().curvature - 7.0f) < 1e-4, "timed rotating cloud must retain offsets");
  pre.process_cut_frame_pcl2(make_cloud(), cuts, times, 1, 30);
  require(!cuts.empty() && cuts.front()->back().curvature > 0, "legacy cut rotating fallback must remain active");
  return 0;
}
