#pragma once

#include <Eigen/Core>

#include <vector>

namespace kd_trees
{
class KdNode
{
private:
  KdNode* lChild;  // point(splitDim) <= splitValue
  KdNode* uChild;  // point(splitDim)  > splitValue
  /// Dimension on which split is applied, -1 if the node is a leaf
  int splitDim;
  /// Value on which the node is splitted, point `p` is in:
  /// - lowerChild if p[splitDim] <= splitValue
  /// - upperChild if p[splitDim] > splitValue
  double splitValue;
  std::vector<Eigen::VectorXd> points;
  /// Store the identifier of each of the points belonging to the node
  std::vector<size_t> points_ids;

public:
  KdNode();

  bool isLeaf() const;

  /// Add the leaves of the given node to the provided vector
  void addLeaves(std::vector<KdNode*>& leaves);

  // Get the leaf corresponding to the given point
  KdNode* getLeaf(const Eigen::VectorXd& point);
  const KdNode* getLeaf(const Eigen::VectorXd& point) const;

  const KdNode* getLowerChild() const;
  const KdNode* getUpperChild() const;
  int getSplitDim() const;
  double getSplitVal() const;

  // Add the point to the current node if it is not already inside
  void push(const Eigen::VectorXd& point, size_t point_id = -1);

  // Remove the last point pushed into this node
  void pop_back();

  // Split node and add separate points to his child
  void split(int splitDim, double splitValue);

  // Update the given space to match the space of the leaf concerning the
  // provided point space is a N by 2 matrix where space(d,0) is the min and
  // space(d,1) is the max
  void leafSpace(Eigen::MatrixXd& space, const Eigen::VectorXd& point) const;

  /// Returns the points directly belonging to the node
  const std::vector<Eigen::VectorXd>& getPoints() const;

  /// Remove the given point from the node:
  /// - If the point is not in the node, throws an out_of_range exception
  void erasePoint(const Eigen::VectorXd& point);

  /// Returns the nearestNeighbor
  void getNearestNeighbor(const Eigen::VectorXd& p, Eigen::VectorXd* best_point, double* lowest_dist,
                          int* best_point_id = nullptr) const;

  /// Navigate to the leaf corresponding to 'p' then pushes the point inside the leaf
  /// - If number of points after addition is above max_points_by_leaf, split the node on dimension `depth mod dim()`
  void pushAutoSplit(const Eigen::VectorXd& p, int max_points_by_leaf, int depth = 0, size_t point_id = -1);

private:
  /// Chooses automatically the split dimension along splitDim
  void autoSplit(int splitDim);
};

}  // namespace kd_trees
