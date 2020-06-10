#include "kd_trees/kd_node.h"

#include <iostream>

#include <rhoban_utils/util.h>

namespace kd_trees
{
KdNode::KdNode() : lChild(NULL), uChild(NULL), splitDim(-1), splitValue(0.0)
{
}

bool KdNode::isLeaf() const
{
  return lChild == NULL;
}

void KdNode::addLeaves(std::vector<KdNode*>& leaves)
{
  if (isLeaf())
  {
    leaves.push_back(this);
    return;
  }
  lChild->addLeaves(leaves);
  uChild->addLeaves(leaves);
}

KdNode* KdNode::getLeaf(const Eigen::VectorXd& point)
{
  if (isLeaf())
  {
    return this;
  }
  if (point(splitDim) > splitValue)
  {
    return uChild->getLeaf(point);
  }
  return lChild->getLeaf(point);
}

const KdNode* KdNode::getLeaf(const Eigen::VectorXd& point) const
{
  if (isLeaf())
  {
    return this;
  }
  if (point(splitDim) > splitValue)
  {
    return uChild;
  }
  return lChild;
}

const KdNode* KdNode::getLowerChild() const
{
  return lChild;
}

const KdNode* KdNode::getUpperChild() const
{
  return uChild;
}

int KdNode::getSplitDim() const
{
  return splitDim;
}

double KdNode::getSplitVal() const
{
  return splitValue;
}

void KdNode::push(const Eigen::VectorXd& point, size_t point_id)
{
  for (const Eigen::VectorXd& p : points)
    if (p == point)
      throw std::logic_error(DEBUG_INFO + "trying to push a duplicated point");
  points.push_back(point);
  points_ids.push_back(point_id);
}

void KdNode::pushAutoSplit(const Eigen::VectorXd& p, int max_points_by_leaf, int depth, size_t point_id)
{
  if (splitDim == -1)
  {
    push(p, point_id);
    if ((int)points.size() > max_points_by_leaf)
    {
      autoSplit(depth % p.rows());
    }
  }
  else if (p(splitDim) > splitValue)
    uChild->pushAutoSplit(p, max_points_by_leaf, depth + 1, point_id);
  else
    lChild->pushAutoSplit(p, max_points_by_leaf, depth + 1, point_id);
}

void KdNode::pop_back()
{
  points.pop_back();
}

void KdNode::split(int dim, double value)
{
  if (!isLeaf())
  {
    throw std::runtime_error("KdNode: Cannot split a non-leaf node");
  }
  splitDim = dim;
  splitValue = value;
  lChild = new KdNode();
  uChild = new KdNode();
  for (size_t idx = 0; idx < points.size(); idx++)
  {
    const Eigen::VectorXd& p = points[idx];
    size_t id = points_ids[idx];
    if (p(splitDim) > splitValue)
    {
      uChild->push(p, id);
    }
    else
    {
      lChild->push(p, id);
    }
  }
  points.clear();
  points_ids.clear();
}

void KdNode::leafSpace(Eigen::MatrixXd& space, const Eigen::VectorXd& point) const
{
  if (isLeaf())
  {
    return;
  }
  if (point(splitDim) > splitValue)
  {
    space(splitDim, 0) = splitValue;
    uChild->leafSpace(space, point);
  }
  else
  {
    space(splitDim, 1) = splitValue;
    lChild->leafSpace(space, point);
  }
}

const std::vector<Eigen::VectorXd>& KdNode::getPoints() const
{
  return points;
}
void KdNode::erasePoint(const Eigen::VectorXd& point)
{
  int point_idx = -1;
  for (size_t i = 0; i < points.size(); i++)
  {
    if (points[i] == point)
    {
      point_idx = i;
      break;
    }
  }
  if (point_idx == -1)
  {
    std::ostringstream oss;
    oss << "Point " << point.transpose() << " is not in Node";
    throw std::out_of_range(DEBUG_INFO + oss.str());
  }
  points.erase(points.begin() + point_idx);
}

void KdNode::autoSplit(int splitDim)
{
  if (points.size() < 2)
    throw std::logic_error(DEBUG_INFO + "autoSplit requires at least 2 nodes");
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  for (const Eigen::VectorXd& p : points)
  {
    double val = p(splitDim);
    if (val < min)
      min = val;
    if (val > max)
      max = val;
  }
  // If all values are the same along splitDim, rather split on another dimension
  // Impossible to end stuck in an endless loop because two equivalent points cannot be in the same node
  // There is at least one point, cf above
  double splitVal = (max + min) / 2;
  if (min == splitVal)
    autoSplit((splitDim + 1) % points[0].rows());
  else
    split(splitDim, splitVal);
}

void KdNode::getNearestNeighbor(const Eigen::VectorXd& p, Eigen::VectorXd* best_point, double* lowest_dist,
                                int* best_point_id) const
{
  // Leaf case: inspect all points in node
  if (splitDim == -1)
  {
    for (size_t idx = 0; idx < points.size(); idx++)
    {
      double dist = (p - points[idx]).norm();
      if (dist < *lowest_dist)
      {
        *lowest_dist = dist;
        *best_point = points[idx];
        if (best_point_id)
          *best_point_id = points_ids[idx];
      }
    }
    return;
  }
  // Look first on space containing the point
  if (p(splitDim) > splitValue)
    uChild->getNearestNeighbor(p, best_point, lowest_dist, best_point_id);
  else
    lChild->getNearestNeighbor(p, best_point, lowest_dist, best_point_id);
  // If distance_to_split is lower than distance to nearest neighbor met, look other side of the tree
  double dist_to_split = std::fabs(p(splitDim) - splitValue);
  if (dist_to_split < *lowest_dist)
  {
    if (p(splitDim) > splitValue)
      lChild->getNearestNeighbor(p, best_point, lowest_dist, best_point_id);
    else
      uChild->getNearestNeighbor(p, best_point, lowest_dist, best_point_id);
  }
}

}  // namespace kd_trees
