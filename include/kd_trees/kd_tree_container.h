#pragma once

#include <kd_trees/kd_tree.h>
#include <map>

namespace kd_trees
{
/// Stores k-dimensional points in a KdTree with associated values
template <class T>
class KdTreeContainer
{
public:
  KdTreeContainer(const Eigen::MatrixXd& space, size_t capacity = -1) : tree(space), capacity(capacity), next_id(0)
  {
  }

  /// Add an entry with the associated data to the container.
  void pushEntry(const Eigen::VectorXd& point, const T& data)
  {
    // If maximal capacity has been reached, remove oldest entry
    if (size() == capacity)
    {
      tree.erasePoint(points_by_id.at(next_id));
      points_by_id.erase(next_id);
      data_by_id.erase(next_id);
    }
    tree.pushAutoSplit(point, 1, next_id);
    points_by_id[next_id] = point;
    data_by_id[next_id] = data;
    next_id++;
    if (next_id == capacity)
      next_id = 0;
  }

  size_t size() const
  {
    return points_by_id.size();
  }

  /// Return the value of the closest object in the container
  /// throws an error if container is empty
  const T& getNearestNeighborData(const Eigen::VectorXd& point) const
  {
    int id = -1;
    tree.getNearestNeighbor(point, &id);
    return data_by_id.at(id);
  }

private:
  /// The tree allowing to retrieve the position of the closest element
  KdTree tree;
  /// Access from point_id to point
  std::map<size_t, Eigen::VectorXd> points_by_id;
  /// Relationship between point id and data
  std::map<size_t, T> data_by_id;
  /// The maximal number of entries at any time
  size_t capacity;
  /// Id of the next point to be added
  size_t next_id;
};

}  // namespace kd_trees
