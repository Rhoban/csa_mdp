#pragma once

#include "rhoban_csa_mdp/core/policy.h"
#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_bbo/optimizer.h"

namespace csa_mdp
{
/// In the MonteCarloPolicy, an optimization is performed on the first step
/// to be taken, then several other steps are performed using the
/// default_policy (provided by the user). This procedure allows to perform
/// a local search while still benefiting of offline procedures for the
/// long-term behaviors.
///
/// TODO: accept a multiple step optimization
///
/// Note: This class is now a specific instance of a PlannerBasedPolicy
class MonteCarloPolicy : public Policy
{
public:
  MonteCarloPolicy();

  virtual void init() override;

  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state) override;
  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const override;

  // Use the provided parameters for first action and then perform a rollout
  double sampleReward(const Eigen::VectorXd& initial_state, const Eigen::VectorXd& action,
                      std::default_random_engine* engine) const;

  /// Uses several rollouts to estimate
  double averageReward(const Eigen::VectorXd& initial_state, const Eigen::VectorXd& action, int rollouts,
                       std::default_random_engine* engine) const;

  std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// Engine used when none is provided
  std::default_random_engine internal_engine;

  /// Definition of the problem is required to simulate the behavior
  std::unique_ptr<Problem> problem;

  /// The policy which will be used once the optimization step has been performed
  std::unique_ptr<Policy> default_policy;

  /// Number of rollouts used to average the reward function inside the eval function
  int nb_rollouts;

  /// Maximal number of calls to the eval function
  /// Total number of simulations is: nb_rollouts * max_eval
  int max_evals;

  /// Number of rollouts used to estimate the value of the function for
  /// each type of action available, once parameters have been optimized
  int validation_rollouts;

  /// How many steps are taken in total
  int simulation_depth;

  /// Verbosity level
  /// 0 -> no output
  /// 1 -> Display estimated gain for using MCP
  /// 2 -> Display estimated value and action for each discrete action
  int debug_level;

  /// The optimizer used for local search
  std::unique_ptr<rhoban_bbo::Optimizer> optimizer;
};

}  // namespace csa_mdp
