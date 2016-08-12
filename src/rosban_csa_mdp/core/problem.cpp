#include "rosban_csa_mdp/core/problem.h"

#include <chrono>

namespace csa_mdp
{

Problem::Problem()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  random_engine = std::default_random_engine(seed);
}

Problem::~Problem()
{
}

Problem::RewardFunction Problem::getRewardFunction() const
{
  return [this](const Eigen::VectorXd &state,
                const Eigen::VectorXd &action,
                const Eigen::VectorXd &next_state)
  {
    return this->getReward(state, action, next_state);
  };
}

Problem::TransitionFunction Problem::getTransitionFunction() const
{
  return [this](const Eigen::VectorXd &state,
                const Eigen::VectorXd &action,
                std::default_random_engine * engine)
  {
    return this->getSuccessor(state, action, engine);
  };
}

Problem::TerminalFunction Problem::getTerminalFunction() const
{
  return [this](const Eigen::VectorXd &state)
  {
    return this->isTerminal(state);
  };
}

int Problem::stateDims() const
{
  return state_limits.rows();
}

int Problem::actionDims() const
{
  return action_limits.rows();
}

const Eigen::MatrixXd & Problem::getStateLimits() const
{
  return state_limits;
}

const Eigen::MatrixXd & Problem::getActionLimits() const
{
  return action_limits;
}

void Problem::setStateLimits(const Eigen::MatrixXd & new_limits)
{
  state_limits = new_limits;
  state_distribution.clear();
  for (int row = 0; row < new_limits.rows(); row++)
  {
    double min = new_limits(row, 0);
    double max = new_limits(row, 1);
    state_distribution.push_back(std::uniform_real_distribution<double>(min, max));
  }
  resetStateNames();
}

void Problem::setActionLimits(const Eigen::MatrixXd & new_limits)
{
  action_limits = new_limits;
  action_distribution.clear();
  for (int row = 0; row < new_limits.rows(); row++)
  {
    double min = new_limits(row, 0);
    double max = new_limits(row, 1);
    action_distribution.push_back(std::uniform_real_distribution<double>(min, max));
  }
  resetActionNames();
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

void Problem::resetActionNames()
{
  std::vector<std::string> names;
  std::string prefix = "action";
  for (int i = 0; i < action_limits.rows(); i++)
  {
    std::ostringstream oss;
    oss << prefix << i;
    names.push_back(oss.str());
  }
  setActionNames(names);
}

void Problem::setStateNames(const std::vector<std::string> &names)
{
  if ((int)names.size() != state_limits.rows())
  {
    std::ostringstream oss;
    oss << "Problem::setStateNames: names.size() != state_limits.rows(), "
        << names.size() << " != " << state_limits.rows();
    throw std::runtime_error(oss.str());
  }
  state_names = names;
}

void Problem::setActionNames(const std::vector<std::string> &names)
{
  if ((int)names.size() != action_limits.rows())
  {
    std::ostringstream oss;
    oss << "Problem::setActionNames: names.size() != action_limits.rows(), "
        << names.size() << " != " << action_limits.rows();
    throw std::runtime_error(oss.str());
  }
  action_names = names;
}

const std::vector<std::string> & Problem::getStateNames() const
{
  return state_names;
}

const std::vector<std::string> & Problem::getActionNames() const
{
  return action_names;
}

std::vector<int> Problem::getLearningDimensions() const
{
  std::vector<int> result;
  result.reserve(stateDims());
  for (int i = 0; i < stateDims(); i++) {
    result.push_back(i);
  }
  return result;
}

Eigen::VectorXd Problem::getSuccessor(const Eigen::VectorXd & state,
                                      const Eigen::VectorXd & action)
{
  return getSuccessor(state, action, &random_engine);
}

Eigen::VectorXd Problem::getRandomAction()
{
  Eigen::VectorXd action(actionDims());
  for (int i = 0; i < actionDims(); i++)
  {
    action(i) = action_distribution[i](random_engine);
  }
  return action;
}

Sample Problem::getRandomSample(const Eigen::VectorXd & state)
{
  Eigen::VectorXd action = getRandomAction();
  Eigen::VectorXd result = getSuccessor(state, action);
  double reward = getReward(state, action, result);
  return Sample(state, action, result, reward);
}

std::vector<Sample> Problem::getRandomTrajectory(const Eigen::VectorXd & initial_state,
                                                 int max_length)
{
  std::vector<Sample> result;
  Eigen::VectorXd state = initial_state;
  while(result.size() < (size_t)max_length)
  {
    Sample new_sample = getRandomSample(state);
    result.push_back(new_sample);
    if (isTerminal(new_sample.next_state))
      break;
    state = new_sample.next_state;
  }
  return result;
}

std::vector<Sample> Problem::getRandomBatch(const Eigen::VectorXd & initial_state,
                                            int max_length,
                                            int nb_trajectories)
{
  std::vector<Sample> result;
  for (int i = 0; i < nb_trajectories; i++)
  {
    for (const Sample & s : getRandomTrajectory(initial_state, max_length))
    {
      result.push_back(s);
    }
  }
  return result;
}

Sample Problem::getSample(const Eigen::VectorXd &state,
                          const Eigen::VectorXd &action)
{
  Eigen::VectorXd result = getSuccessor(state, action);
  double reward = getReward(state, action, result);
  return Sample(state, action, result, reward);
}

std::vector<Sample> Problem::simulateTrajectory(const Eigen::VectorXd &initial_state,
                                                int max_length,
                                                Problem::Policy p)
{
  std::vector<Sample> result;
  Eigen::VectorXd state = initial_state;
  while(result.size() < (size_t)max_length)
  {
    Sample new_sample = getSample(state, p(state));
    result.push_back(new_sample);
    if (isTerminal(new_sample.next_state))
      break;
    state = new_sample.next_state;
  }
  return result;
}

}
