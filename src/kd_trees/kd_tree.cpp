#include "kd_trees/kd_tree.h"

#include <rhoban_utils/util.h>

namespace kd_trees
{
KdTree::KdTree(const Eigen::MatrixXd& tree_space) : space(tree_space)
{
}

const KdNode* KdTree::getRoot() const
{
  return &root;
}

std::vector<KdNode*> KdTree::getLeaves()
{
  std::vector<KdNode*> leaves;
  root.addLeaves(leaves);
  return leaves;
}

int KdTree::dim() const
{
  return space.rows();
}

KdNode* KdTree::getLeaf(const Eigen::VectorXd& point)
{
  return root.getLeaf(point);
}

const KdNode* KdTree::getLeaf(const Eigen::VectorXd& point) const
{
  return root.getLeaf(point);
}

void KdTree::push(const Eigen::VectorXd& point, size_t point_id)
{
  getLeaf(point)->push(point, point_id);
}

void KdTree::pushAutoSplit(const Eigen::VectorXd& point, int max_points_by_leaf, size_t point_id)
{
  root.pushAutoSplit(point, max_points_by_leaf, 0, point_id);
}

const Eigen::MatrixXd& KdTree::getSpace() const
{
  return space;
}

Eigen::MatrixXd KdTree::getSpace(const Eigen::VectorXd& point) const
{
  Eigen::MatrixXd leaf_space = space;
  root.leafSpace(leaf_space, point);
  return leaf_space;
}

void KdTree::erasePoint(const Eigen::VectorXd& point)
{
  root.getLeaf(point)->erasePoint(point);
}

Eigen::VectorXd KdTree::getNearestNeighbor(const Eigen::VectorXd& p, int* id) const
{
  int unused;
  if (id == nullptr)
    id = &unused;
  double lowest_dist = std::numeric_limits<double>::max();
  Eigen::VectorXd nearest_neighbor;
  root.getNearestNeighbor(p, &nearest_neighbor, &lowest_dist, id);
  if (lowest_dist == std::numeric_limits<double>::max())
    throw std::logic_error(DEBUG_INFO + "requesting nearest neighbor in an empty KdTree");
  return nearest_neighbor;
}

}  // namespace kd_trees
