#pragma once

#include "kd_trees/kd_node.h"

namespace kd_trees
{
class KdTree
{
private:
  KdNode root;
  Eigen::MatrixXd space;

public:
  KdTree(const Eigen::MatrixXd& space);

  int dim() const;
  const KdNode* getRoot() const;

  /// Return all the leaves inside the tree
  std::vector<KdNode*> getLeaves();

  KdNode* getLeaf(const Eigen::VectorXd& point);
  const KdNode* getLeaf(const Eigen::VectorXd& point) const;

  /// Simply push the point to the appropriate leaf
  void push(const Eigen::VectorXd& point, size_t point_id = -1);

  /// Push the point in the KdTree on the appropriate leaf:
  /// - If number of points after addition is above max_points_by_leaf, split the node on dimension `depth mod dim()`
  void pushAutoSplit(const Eigen::VectorXd& point, int max_points_by_leaf, size_t point_id = -1);

  const Eigen::MatrixXd& getSpace() const;
  Eigen::MatrixXd getSpace(const Eigen::VectorXd& point) const;

  /// Remove the given point from the KdTree:
  /// - If the point is not in the tree, throws an out_of_range exception
  void erasePoint(const Eigen::VectorXd& point);

  Eigen::VectorXd getNearestNeighbor(const Eigen::VectorXd& p, int* id = nullptr) const;
};

}  // namespace kd_trees
