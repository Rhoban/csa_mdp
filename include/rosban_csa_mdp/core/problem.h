#pragma once

#include "rosban_csa_mdp/core/sample.h"

#include <Eigen/Core>

#include <functional>

namespace csa_mdp
{

class Problem
{
public:
  std::function<Eigen::VectorXd(const Eigen::VectorXd &state)> Policy;

private:
  Eigen::MatrixXd state_limits;
  Eigen::MatrixXd action_limits;

  std::vector<std::uniform_real_distribution<double>> state_distribution;
  std::vector<std::uniform_real_distribution<double>> action_distribution;

protected:
  std::default_random_engine random_engine;

public:
  Problem();

  int stateDims() const;
  int actionDims() const;

  void setStateLimits(const Eigen::MatrixXd & new_limits);
  void setActionLimits(const Eigen::MatrixXd & new_limits);
  
  virtual bool isTerminal(const Eigen::VectorXd &) = 0;
  /// This function is allowed to be stochastic
  virtual double getReward(const Eigen::VectorXd & state,
                           const Eigen::VectorXd & action,
                           const Eigen::VectorXd & dst) = 0;
  /// This function is allowed to be stochastic
  virtual Eigen::VectorXd getSuccessor(const Eigen::VectorXd & state,
                                       const Eigen::VectorXd & action) = 0;

  /// Provide a random state in state_limits (uniformous distribution)
  Eigen::VectorXd getRandomState();
  /// Provide a random action in action_limits (uniformous distribution)
  Eigen::VectorXd getRandomAction();
  /// Provide a sample starting from state and using a random action
  Sample getRandomSample(const Eigen::VectorXd & state);
  /// Compute a random trajectory starting at the given state
  std::vector<Sample> getRandomTrajectory(const Eigen::VectorXd & initial_state,
                                          int max_length);
  /// Compute several random trajectories starting from the same state
  std::vector<Sample> getRandomBatch(const Eigen::VectorXd & initial_state,
                                     int max_length,
                                     int nb_trajectories);
};

}
