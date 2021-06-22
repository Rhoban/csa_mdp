#include "rhoban_csa_mdp/knownness/knownness_tree.h"

#include "rhoban_regression_forests/approximations/pwc_approximation.h"
#include "rhoban_random/tools.h"
#include "rhoban_regression_forests/tools/statistics.h"

using regression_forests::Approximation;
using namespace regression_forests::Statistics;

namespace csa_mdp
{
KnownnessTree::Config::Config() : max_points(10), type(Type::Random)
{
}

std::string KnownnessTree::Config::getClassName() const
{
  return "KnownnessTreeConfig";
}

Json::Value KnownnessTree::Config::toJson() const
{
  Json::Value v;
  v["max_points"] = max_points;
  v["type"] = to_string(type);
  return v;
}

void KnownnessTree::Config::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  std::string type_str;
  rhoban_utils::tryRead(v, "max_points", &max_points);
  rhoban_utils::tryRead(v, "type", &type_str);
  if (type_str != "")
  {
    type = loadType(type_str);
  }
}

KnownnessTree::KnownnessTree(const Eigen::MatrixXd& space, const Config& conf_)
  : tree(space), conf(conf_), nb_points(0), next_split_dim(0)
{
  random_engine = rhoban_random::getRandomEngine();
}

void KnownnessTree::push(const Eigen::VectorXd& point)
{
  // Checking if the point is in the tree space
  const Eigen::MatrixXd& tree_space = tree.getSpace();
  for (int dim = 0; dim < point.rows(); dim++)
  {
    if (point(dim) < tree_space(dim, 0) || point(dim) > tree_space(dim, 1))
    {
      std::ostringstream oss;
      oss << "Point is outside of space!" << std::endl
          << "P: " << point.transpose() << std::endl
          << "Space:" << std::endl
          << tree_space << std::endl;
      throw std::runtime_error(oss.str());
    }
  }
  // Pushing point
  kd_trees::KdNode* leafNode = tree.getLeaf(point);
  Eigen::MatrixXd leaf_space = tree.getSpace(point);
  leafNode->push(point);
  int leafCount = leafNode->getPoints().size();
  if (leafCount > conf.max_points)
  {
    int split_dim = -1;
    double split_val = 0;
    switch (conf.type)
    {
      case Type::MRE:
        split_dim = next_split_dim;
        split_val = (leaf_space(split_dim, 0) + leaf_space(split_dim, 1)) / 2;
        break;
      case Type::Random:  // TODO refactor (move to a function and split)
      {
        double best_dim_score = 0;
        // For every dimension, try a random split, score it and keep it if necessary
        for (int dim = 0; dim < tree_space.rows(); dim++)
        {
          // choose a random split ensuring there is at least one point on each side
          double s_val_max = std::numeric_limits<double>::lowest();
          double s_val_min = std::numeric_limits<double>::max();
          // Finding min and max points along this dimension
          for (const auto& p : leafNode->getPoints())
          {
            double val = p(dim);
            if (val < s_val_min)
              s_val_min = val;
            if (val > s_val_max)
              s_val_max = val;
          }
          // Choose another dimension if all the points along this dimension are similars
          if (s_val_min == s_val_max)
            continue;
          // Generate random value
          std::uniform_real_distribution<double> val_distrib(s_val_min, s_val_max);
          double curr_split_val = val_distrib(random_engine);
          // This can really happen (even if it not supposed to)
          if (curr_split_val == s_val_max)
            continue;
          // Gathering points
          std::vector<double> values, lower_values, upper_values;
          for (const auto& p : leafNode->getPoints())
          {
            double val = p(dim);
            values.push_back(val);
            if (val <= curr_split_val)
            {
              lower_values.push_back(val);
            }
            else
            {
              upper_values.push_back(val);
            }
          }
          // Size of points set
          int global_size = values.size();
          int lower_size = lower_values.size();
          int upper_size = upper_values.size();
          // variance score [0, 1], 1 is the best
          double global_var = variance(values);
          double lower_var = variance(lower_values);
          double upper_var = variance(upper_values);
          double var_score = (lower_var * lower_size + upper_var * upper_size) / (global_size * global_var);
          var_score = std::max(0.0, 1 - var_score);  // Normalization
          // size score [0,1], 1 is the best
          double dim_size = leaf_space(dim, 1) - leaf_space(dim, 0);
          double lower_ratio = (curr_split_val - leaf_space(dim, 0)) / dim_size;
          double upper_ratio = (leaf_space(dim, 1) - curr_split_val) / dim_size;
          double size_score = 1 - (lower_size * lower_ratio + upper_size * upper_ratio) / (global_size);
          // dim_score [0,1], 1 is the best
          double dim_score = size_score / 2 + var_score / 2;
          if (dim_score > best_dim_score)
          {
            split_dim = dim;
            best_dim_score = dim_score;
            split_val = curr_split_val;
          }
        }
        if (split_dim < 0)
        {
          leafNode->pop_back();
          return;
          // std::ostringstream oss;
          // oss << "No split candidate found: Points:" << std::endl;
          // for (const auto & p : leafNode->getPoints())
          //{
          //  oss << "\t" << p.transpose() << std::endl;
          //}
          // throw std::runtime_error(oss.str());
        }
      }
    }
    // Apply split
    leafNode->split(split_dim, split_val);
    next_split_dim++;
    if (next_split_dim == leaf_space.rows())
    {
      next_split_dim = 0;
    }
  }
  nb_points++;
}

double KnownnessTree::getMu() const
{
  int k = tree.dim();
  return 1.0 / floor(std::pow(nb_points * k / conf.max_points, 1.0 / k));
}

double KnownnessTree::getValue(const Eigen::VectorXd& point) const
{
  const kd_trees::KdNode* leaf = tree.getLeaf(point);
  int leaf_count = leaf->getPoints().size();
  Eigen::MatrixXd leaf_space = tree.getSpace(point);
  return getValue(leaf_space, leaf_count);
}

double KnownnessTree::getValue(const Eigen::MatrixXd& space, int local_points) const
{
  double max_size = 0;
  const Eigen::MatrixXd& tree_space = tree.getSpace();
  switch (conf.type)
  {
    case Type::MRE:
      for (int dim = 0; dim < space.rows(); dim++)
      {
        double local_size = space(dim, 1) - space(dim, 0);
        double tree_size = tree_space(dim, 1) - tree_space(dim, 0);
        // Since forall s, norm_inf(s) <= 1, then the length of a dimension is maximum 2
        double size = 2 * local_size / tree_size;
        if (size > max_size)
        {
          max_size = size;
        }
      }
      return std::min(1.0, (double)local_points / conf.max_points * getMu() / max_size);
    case Type::Random: {
      double local_size = 1.0;
      double global_size = 1.0;
      for (int dim = 0; dim < space.rows(); dim++)
      {
        local_size *= space(dim, 1) - space(dim, 0);
        global_size *= tree_space(dim, 1) - tree_space(dim, 0);
      }
      double local_density = local_points / local_size;
      double global_density = nb_points / global_size;
      double density_ratio = local_density / global_density;
      double raw_value = density_ratio;
      // Test:
      // - Required density is reduced when the number of points grows
      raw_value = density_ratio * log(nb_points);
      double value = std::min(1.0, raw_value);
      return value;
    }
  }
  throw std::runtime_error("Unhandled type for knownness tree");
}

regression_forests::Node* KnownnessTree::convertToRegNode(const kd_trees::KdNode* node, Eigen::MatrixXd& space) const
{
  if (node == NULL)
    return NULL;
  regression_forests::Node* new_node = new regression_forests::Node();
  // Leaf case
  if (node->isLeaf())
  {
    int nb_points = node->getPoints().size();
    double value = getValue(space, nb_points);
    new_node->a = std::unique_ptr<Approximation>(new regression_forests::PWCApproximation(value));
    return new_node;
  }
  // Node case
  int split_dim = node->getSplitDim();
  double split_val = node->getSplitVal();
  double old_min = space(split_dim, 0);
  double old_max = space(split_dim, 1);
  // Update split
  new_node->s.dim = node->getSplitDim();
  new_node->s.val = node->getSplitVal();
  // Update lower child
  space(split_dim, 1) = split_val;
  new_node->lowerChild = convertToRegNode(node->getLowerChild(), space);
  space(split_dim, 1) = old_max;
  // Update upper child
  space(split_dim, 0) = split_val;
  new_node->upperChild = convertToRegNode(node->getUpperChild(), space);
  space(split_dim, 0) = old_min;
  return new_node;
}

std::unique_ptr<regression_forests::Tree> KnownnessTree::convertToRegTree() const
{
  std::unique_ptr<regression_forests::Tree> reg_tree(new regression_forests::Tree);
  Eigen::MatrixXd space = tree.getSpace();
  reg_tree->root = convertToRegNode(tree.getRoot(), space);
  return reg_tree;
}

void KnownnessTree::checkConsistency()
{
  std::vector<kd_trees::KdNode*> leaves = tree.getLeaves();
  int leaf_points = 0;
  for (kd_trees::KdNode* leaf : leaves)
  {
    leaf_points += leaf->getPoints().size();
  }
  if (leaf_points != nb_points)
  {
    std::ostringstream oss;
    oss << "KnownnessTree::checkConsistency: not consistent! expecting " << nb_points << " points and found "
        << leaf_points << " points";
    throw std::logic_error(oss.str());
  }
}

std::string to_string(KnownnessTree::Type type)
{
  switch (type)
  {
    case KnownnessTree::Type::MRE:
      return "MRE";
    case KnownnessTree::Type::Random:
      return "Random";
  }
  throw std::runtime_error("Unknown type in to_string(Type)");
}

KnownnessTree::Type loadType(const std::string& type)
{
  if (type == "MRE")
  {
    return KnownnessTree::Type::MRE;
  }
  if (type == "Random")
  {
    return KnownnessTree::Type::Random;
  }
  throw std::runtime_error("Unknown KnownnessTree Type: '" + type + "'");
}

}  // namespace csa_mdp
