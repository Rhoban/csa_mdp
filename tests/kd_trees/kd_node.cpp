#include <gtest/gtest.h>
#include <kd_trees/kd_node.h>

#define _USE_MATH_DEFINES
#include <cmath>

#define EPSILON std::pow(10, -6)

using namespace kd_trees;

TEST(constructor, empty)
{
  KdNode n;
  EXPECT_EQ(nullptr, n.getLowerChild());
  EXPECT_EQ(nullptr, n.getUpperChild());
  EXPECT_EQ(-1, n.getSplitDim());
  EXPECT_TRUE(n.isLeaf());
}

TEST(getLeaf, depth1)
{
  KdNode n;
  n.split(2, 3.2);
  EXPECT_EQ(n.getLowerChild(), n.getLeaf(Eigen::Vector3d(4.0, 1.2, 2.0)));
  EXPECT_EQ(n.getUpperChild(), n.getLeaf(Eigen::Vector3d(1.0, 1.2, 4.0)));
  /// Testing border case
  EXPECT_EQ(n.getLowerChild(), n.getLeaf(Eigen::Vector3d(6.0, 1.2, 3.2)));
}

TEST(push, no_param)
{
  KdNode n;
  Eigen::Vector2d p1(1.0, -1.0);
  Eigen::Vector2d p2(1.2, -9.3);
  n.push(p1);
  EXPECT_EQ((size_t)1, n.getPoints().size());
  n.push(p2);
  EXPECT_EQ((size_t)2, n.getPoints().size());
}

TEST(push, duplicated_point)
{
  KdNode n;
  Eigen::Vector2d p(1.0, -1.0);
  n.push(p);
  try
  {
    n.push(p);
    FAIL() << "Adding twice the same point to a node should lead to an error" << std::endl;
  }
  catch (const std::logic_error& exc)
  {
  }
}

TEST(pop_back, basic)
{
  KdNode n;
  Eigen::Vector2d p1(1.0, -1.0);
  Eigen::Vector2d p2(1.2, -9.3);
  n.push(p1);
  EXPECT_EQ((size_t)1, n.getPoints().size());
  n.push(p2);
  EXPECT_EQ((size_t)2, n.getPoints().size());
  n.pop_back();
  EXPECT_EQ((size_t)1, n.getPoints().size());
  Eigen::Vector2d remaining_point = n.getPoints()[0];
  for (int dim = 0; dim < 2; dim++)
  {
    EXPECT_EQ(p1(dim), remaining_point(dim));
  }
}

TEST(pushAutoSplit, simple)
{
  KdNode n;
  Eigen::Vector2d p1(1.0, -1.0);
  Eigen::Vector2d p2(1.2, -9.3);
  n.pushAutoSplit(p1, 1);
  n.pushAutoSplit(p2, 1);
  // Node should be split -> no more points inside
  EXPECT_EQ((size_t)0, n.getPoints().size());
  EXPECT_EQ(0, n.getSplitDim());
  EXPECT_FLOAT_EQ(1.1, n.getSplitVal());
}

TEST(pushAutoSplit, duplicated_val)
{
  KdNode n;
  Eigen::Vector2d p1(1.2, -1.0);
  Eigen::Vector2d p2(1.2, -2.0);
  n.pushAutoSplit(p1, 1);
  n.pushAutoSplit(p2, 1);
  // Node should be split -> no more points inside
  EXPECT_EQ((size_t)0, n.getPoints().size());
  EXPECT_EQ(1, n.getSplitDim());
  EXPECT_FLOAT_EQ(-1.5, n.getSplitVal());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
