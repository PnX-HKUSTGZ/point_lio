#include "ivox/ivox3d.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>

using Point = pcl::PointXYZ;
using Map = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, Point>;
void require(bool ok, const char * message)
{
  if (!ok) {std::cerr << message << '\n'; std::exit(1);}
}
std::size_t count(const Map & map)
{
  std::size_t n = 0;
  for (const auto & entry : map.grids_map_) n += entry.second->second.Size();
  return n;
}

int main()
{
  Map::Options options;
  options.resolution_ = .5f;
  options.point_resolution_ = .1f;
  options.nearby_type_ = Map::NearbyType::NEARBY26;
  Map map(options);
  require(map.StatGridPoints() == std::vector<float>(5, 0), "empty statistics invalid");
  Map::PointVector scan;
  // A dense revisited surface crossing both zero and coarse-cell boundaries.
  for (int x = -10; x < 10; ++x) {
    for (int y = -10; y < 10; ++y) {
      scan.emplace_back(x * .1f + .011f, y * .1f + .013f, .021f);
    }
  }
  for (int pass = 0; pass < 250; ++pass) map.AddPoints(scan);
  require(count(map) == scan.size(), "revisited voxels must not accumulate points");
  require(map.StatGridPoints()[2] <= 25, "coarse-cell density must remain bounded");
  // Prefer the measured point closest to the fine voxel's center; no averaging
  // across an edge, and retain the positive/negative voxel distinction.
  map.AddPoints(Map::PointVector{Point(.049f, .049f, .049f), Point(-.049f,-.049f,.049f)});
  require(count(map) == scan.size(), "replacement changed voxel count");
  Map::PointVector nearest;
  require(map.GetClosestPoint(Point(.05f,.05f,.05f), nearest, 1, .1), "no nearest point");
  require(std::abs(nearest.front().x-.049f)<1e-6, "center representative not replaced");
  require(map.GetClosestPoint(Point(-.05f,-.05f,.05f), nearest, 1, .1), "negative voxel missing");
  require(std::abs(nearest.front().x+.049f)<1e-6, "negative coordinates rounded incorrectly");
  require(!map.GetClosestPoint(Point(100,100,100), nearest), "empty query unexpectedly matched");
  require(nearest.empty(), "failed KNN query retained old neighbors");
  map.AddPoints(Map::PointVector{Point(std::numeric_limits<float>::quiet_NaN(),0,0)});
  require(count(map) == scan.size(), "nonfinite point inserted");

  // Compare exact KNN distances to an exhaustive oracle over all retained
  // points in the queried neighborhood. Filtering must not corrupt indices.
  const Point query(.027f, .039f, .046f);
  require(map.GetClosestPoint(query, nearest, 5, .2), "KNN failed");
  std::vector<float> expected, actual;
  for (const auto & entry : map.grids_map_) {
    for (size_t i=0; i<entry.second->second.Size(); ++i) {
      float d = faster_lio::distance2(query, entry.second->second.GetPoint(i));
      if (d < .2f*.2f) expected.push_back(d);
    }
  }
  for (const auto & point : nearest) actual.push_back(faster_lio::distance2(query, point));
  std::sort(expected.begin(), expected.end()); std::sort(actual.begin(), actual.end());
  require(actual.size()==5, "not enough nearest neighbors");
  for (size_t i=0; i<5; ++i) require(std::abs(actual[i]-expected[i])<1e-6, "KNN differs from oracle");

  options.capacity_ = 3;
  Map evicted(options);
  evicted.AddPoints(Map::PointVector{Point(.01f,0,0), Point(2,0,0), Point(4,0,0)});
  evicted.AddPoints(Map::PointVector{Point(.01f,0,0)});
  require(evicted.GetClosestPoint(Point(.01f,0,0), nearest), "evicted voxel could not be reinserted");

  options.capacity_ = 1000000;
  options.point_resolution_ = 0;
  Map legacy(options);
  for (int pass=0; pass<250; ++pass) legacy.AddPoints(scan);
  require(count(legacy)==250*scan.size(), "disabled filtering changed legacy insertion");
  auto benchmark = [&](Map & tested) {
    auto start=std::chrono::steady_clock::now();
    for (int i=0; i<2000; ++i) tested.GetClosestPoint(query, nearest);
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  };
  std::cout << "revisit points legacy=" << count(legacy) << " bounded=" << count(map)
            << " query_ms legacy=" << benchmark(legacy) << " bounded=" << benchmark(map) << '\n';
}
