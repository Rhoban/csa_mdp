#include "kd_trees/kd_tree.h"

namespace Math {
  namespace KdTrees {

    KdTree::KdTree(const Eigen::MatrixXd & tree_space)
      : space(tree_space)
    {
    }

    int KdTree::dim() const
    {
      return space.rows();
    }

    KdNode * KdTree::getLeaf(const Eigen::VectorXd& point)
    {
      return root.getLeaf(point);
    }

    void KdTree::push(const Eigen::VectorXd& point)
    {
      getLeaf(point)->push(point);
    }

    Eigen::MatrixXd KdTree::getSpace(const Eigen::VectorXd& point) const
    {
      Eigen::MatrixXd leaf_space = space;
      root.leafSpace(leaf_space, point);
      return leaf_space;
    }
  }
}
