#pragma once
#include "rhoban_csa_mdp/online_planners/open_loop_planner.h"
#include "rhoban_csa_mdp/core/agent_selector.h"

namespace csa_mdp
{
class MultiOpenLoopPlanner : public OpenLoopPlanner
{
public:
  MultiOpenLoopPlanner();
  ~MultiOpenLoopPlanner();
  // Check that the OpenLoopPlanner is properly configured for the given problem
  void checkConsistency(const AgentSelector& as) const;

  /// Configure the optimizer for the given problem
  void prepareOptimizer(const AgentSelector& as);

  /// Uses the provided policy to obtain an initial guess for the 'look_ahead' coming actions
  Eigen::VectorXd getInitialGuess(const Problem& p, const AgentSelector& as, const int main_agent,
                                  const Eigen::VectorXd& initial_state, const Policy& policy,
                                  std::default_random_engine* engine) const;

  /// Sample the reward received by applying all the 'next_actions' from
  /// 'initial_state' with problem 'p'. The last state of the rollout is stored
  /// in 'last_state' and if a terminal status is received, 'is_terminated' is
  /// set to true.
  double sampleLookAheadReward(const Problem& p, const AgentSelector& as, const Policy& policy, const int main_agent,
                               const Eigen::VectorXd& initial_state, const Eigen::VectorXd& next_actions,
                               Eigen::VectorXd* last_state, bool* is_terminated,
                               std::default_random_engine* engine) const;

  /// Optimize the next action for the given problem 'p' starting at 'state'
  /// according to inner parameters and provided policy and value_function
  ///
  /// It is mandatory to have called 'prepareOptimizer' with the same problem before
  Eigen::VectorXd planNextAction(const Problem& p, const AgentSelector& as, const Eigen::VectorXd& state,
                                 const Policy& policy, const rhoban_fa::FunctionApproximator& value_function,
                                 std::default_random_engine* engine) const;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  virtual std::string getClassName() const override;

private:
  /// Optimizer used for open loop planning
  std::unique_ptr<rhoban_bbo::Optimizer> optimizer;

  /// Number of steps of look ahead
  int look_ahead;

  /// Number of steps for a trial when using a default policy after step
  int trial_length;

  /// Number of rollouts used to average the reward at each sample
  int rollouts_per_sample;

  /// Discount value used for optimization
  double discount;

  /// Is policy used to guess initial candidate
  bool guess_initial_candidate;

  EvaluationPolicy evaluation_policy;
};
}  // namespace csa_mdp
