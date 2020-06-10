#include <gtest/gtest.h>
#include <kd_trees/kd_tree_container.h>

#define _USE_MATH_DEFINES
#include <cmath>

#define EPSILON std::pow(10, -6)

using namespace kd_trees;

TEST(pushEntry, overflowCapacity)
{
  Eigen::MatrixXd space(2, 2);
  space << -1.0, 2.0, 0.0, 4.0;
  KdTreeContainer<double> c(space, 2);
  std::vector<Eigen::Vector2d> points = { Eigen::Vector2d(0, 0), Eigen::Vector2d(0.5, 0.0), Eigen::Vector2d(1.5, 2.0) };
  std::vector<double> values = { 1.0, 2.3, 5.7 };
  for (size_t point_id = 0; point_id < points.size(); point_id++)
  {
    c.pushEntry(points[point_id], values[point_id]);
  }
  EXPECT_EQ(2, c.size());
}

TEST(nearestNeighbor, simple)
{
  Eigen::MatrixXd space(2, 2);
  space << -1.0, 2.0, 0.0, 4.0;
  KdTreeContainer<double> c(space, 100);
  std::vector<std::pair<Eigen::Vector2d, double>> input_data = { { Eigen::Vector2d(0.0, 0), 1.0 },
                                                                 { Eigen::Vector2d(0.5, 0.0), 2.3 },
                                                                 { Eigen::Vector2d(1.5, 2.0), 5.7 } };
  std::vector<std::pair<Eigen::Vector2d, double>> test_data = { { Eigen::Vector2d(0.2, 0), 1.0 },
                                                                { Eigen::Vector2d(0.5, 0.0), 2.3 },
                                                                { Eigen::Vector2d(1.2, 1.8), 5.7 } };
  for (const auto& entry : input_data)
    c.pushEntry(entry.first, entry.second);
  for (size_t idx = 0; idx < test_data.size(); idx++)
  {
    double data = c.getNearestNeighborData(test_data[idx].first);
    EXPECT_EQ(test_data[idx].second, data);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
