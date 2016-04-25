#include "rosban_csa_mdp/core/policy.h"

namespace csa_mdp
{

Policy::Policy()
{
  init();
}

void Policy::init(){}

void Policy::setActionLimits(const Eigen::MatrixXd &limits)
{
  action_limits = limits;
}

Eigen::VectorXd Policy::boundAction(const Eigen::VectorXd & raw_action)
{
  if (raw_action.rows() != action_limits.rows())
  {
    std::ostringstream oss;
    oss << "Policy::boundAction: Number of rows does not match" << std::endl
        << "\traw_action   : " << raw_action.rows()    << std::endl
        << "\taction_limits: " << action_limits.rows() << std::endl;
    throw std::runtime_error(oss.str());
  }
  Eigen::VectorXd action(raw_action.rows());
  for (int dim = 0; dim < raw_action.rows(); dim++)
  {
    action(dim) = std::min(action_limits(dim,1), std::max(action_limits(dim,0), raw_action(dim)));
  }
  return action;
}

Eigen::VectorXd Policy::getAction(const Eigen::VectorXd &state)
{
  return boundAction(getRawAction(state));
}

}