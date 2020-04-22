#include "rhoban_csa_mdp/core/problem.h"

#include <chrono>

namespace csa_mdp
{
Problem::Problem()
{
}

Problem::~Problem()
{
}

void Problem::checkActionId(int action_id) const
{
  int max_index = getNbActions() - 1;
  if (action_id < 0 || action_id > max_index)
  {
    std::ostringstream oss;
    oss << "Problem::checkActionId: action_id = " << action_id << " is out of bounds [0," << max_index << "]";
    throw std::runtime_error(oss.str());
  }
}

Problem::ResultFunction Problem::getResultFunction() const
{
  return [this](const Eigen::VectorXd& state, const Eigen::VectorXd& action, std::default_random_engine* engine) {
    return this->getSuccessor(state, action, engine);
  };
}

int Problem::stateDims() const
{
  return state_limits.rows();
}

int Problem::getNbActions() const
{
  return actions_limits.size();
}

int Problem::actionDims(int action_id) const
{
  checkActionId(action_id);
  return actions_limits[action_id].rows();
}

const Eigen::MatrixXd& Problem::getStateLimits() const
{
  return state_limits;
}

const std::vector<Eigen::MatrixXd>& Problem::getActionsLimits() const
{
  return actions_limits;
}

const Eigen::MatrixXd& Problem::getActionLimits(int action_id) const
{
  checkActionId(action_id);
  return actions_limits[action_id];
}

void Problem::setStateLimits(const Eigen::MatrixXd& new_limits)
{
  state_limits = new_limits;
  resetStateNames();
}

void Problem::setActionLimits(const std::vector<Eigen::MatrixXd>& new_limits)
{
  actions_limits = new_limits;
  resetActionsNames();
}

void Problem::resetStateNames()
{
  std::vector<std::string> names;
  std::string prefix = "state_";
  for (int i = 0; i < state_limits.rows(); i++)
  {
    std::ostringstream oss;
    oss << prefix << i;
    names.push_back(oss.str());
  }
  setStateNames(names);
}

void Problem::resetActionsNames()
{
  std::vector<std::vector<std::string>> names;
  std::string prefix = "action";
  for (int action_id = 0; action_id < getNbActions(); action_id++)
  {
    std::vector<std::string> action_names;
    for (int dim = 0; dim < actionDims(action_id); dim++)
    {
      std::ostringstream oss;
      oss << prefix << "_" << action_id << "_" << dim;
      action_names.push_back(oss.str());
    }
    names.push_back(action_names);
  }
  setActionsNames(names);
}

void Problem::setStateNames(const std::vector<std::string>& names)
{
  if ((int)names.size() != state_limits.rows())
  {
    std::ostringstream oss;
    oss << "Problem::setStateNames: names.size() != state_limits.rows(), " << names.size()
        << " != " << state_limits.rows();
    throw std::runtime_error(oss.str());
  }
  state_names = names;
}

void Problem::setActionsNames(const std::vector<std::vector<std::string>>& names)
{
  if ((int)names.size() != getNbActions())
  {
    std::ostringstream oss;
    oss << "Problem::setActionsNames: names.size() != getNbActions(), " << names.size() << " != " << getNbActions();
    throw std::runtime_error(oss.str());
  }
  actions_names.clear();
  actions_names.resize(getNbActions());
  for (int action_id = 0; action_id < getNbActions(); action_id++)
  {
    setActionNames(action_id, names[action_id]);
  }
}

void Problem::setActionNames(int action_id, const std::vector<std::string>& names)
{
  checkActionId(action_id);
  // Consistency of actions_names and actions_limits is always ensured
  actions_names[action_id] = names;
}

const std::vector<std::string>& Problem::getStateNames() const
{
  return state_names;
}

const std::vector<std::vector<std::string>>& Problem::getActionsNames() const
{
  return actions_names;
}

const std::vector<std::string> Problem::getActionNames(int action_id) const
{
  checkActionId(action_id);
  return actions_names[action_id];
}

std::vector<int> Problem::getLearningDimensions() const
{
  std::vector<int> result;
  result.reserve(stateDims());
  for (int i = 0; i < stateDims(); i++)
  {
    result.push_back(i);
  }
  return result;
}

Eigen::VectorXd Problem::getLearningState(const Eigen::VectorXd& state) const
{
  std::vector<int> learning_dimensions = getLearningDimensions();
  Eigen::VectorXd learning_state(learning_dimensions.size());
  for (size_t dim = 0; dim < learning_dimensions.size(); dim++)
  {
    learning_state(dim) = state(learning_dimensions[dim]);
  }
  return learning_state;
}

Eigen::MatrixXd Problem::getLearningStateLimits() const
{
  std::vector<int> learning_dimensions = getLearningDimensions();
  Eigen::MatrixXd state_limits = getStateLimits();
  Eigen::MatrixXd learning_limits(learning_dimensions.size(), 2);
  for (size_t dim = 0; dim < learning_dimensions.size(); dim++)
  {
    learning_limits.row(dim) = state_limits.row(learning_dimensions[dim]);
  }
  return learning_limits;
}

double Problem::sampleRolloutReward(const Eigen::VectorXd& initial_state, const csa_mdp::Policy& policy,
                                    int max_horizon, double discount, std::default_random_engine* engine) const
{
  double coeff = 1;
  double reward = 0;
  Eigen::VectorXd state = initial_state;
  bool is_terminated = false;
  // Compute the reward over the next 'nb_steps'
  for (int i = 0; i < max_horizon; i++)
  {
    Eigen::VectorXd action = policy.getAction(state, engine);
    Problem::Result result = getSuccessor(state, action, engine);
    reward += coeff * result.reward;
    state = result.successor;
    coeff *= discount;
    is_terminated = result.terminal;
    // Stop predicting steps if a terminal state has been reached
    if (is_terminated)
      break;
  }
  return reward;
}

}  // namespace csa_mdp
