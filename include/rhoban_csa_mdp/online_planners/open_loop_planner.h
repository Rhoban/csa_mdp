#pragma once

#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_utils/serialization/json_serializable.h"
#include "rhoban_fa/function_approximator.h"
#include "rhoban_bbo/optimizer.h"

namespace csa_mdp
{
class OpenLoopPlanner : public rhoban_utils::JsonSerializable
{
public:
  OpenLoopPlanner();

  /// Check that the OpenLoopPlanner is properly configured for the given problem
  void checkConsistency(const Problem& p) const;

  /// Configure the optimizer for the given problem
  void prepareOptimizer(const Problem& p);

  /// Uses the provided policy to obtain an initial guess for the 'look_ahead' coming actions
  Eigen::VectorXd getInitialGuess(const Problem& p, const Eigen::VectorXd& initial_state, const Policy& policy,
                                  std::default_random_engine* engine) const;

  /// Sample the reward received by applying all the 'next_actions' from
  /// 'initial_state' with problem 'p'. The last state of the rollout is stored
  /// in 'last_state' and if a terminal status is received, 'is_terminated' is
  /// set to true.
  double sampleLookAheadReward(const Problem& p, const Eigen::VectorXd& initial_state,
                               const Eigen::VectorXd& next_actions, Eigen::VectorXd* last_state, bool* is_terminated,
                               std::default_random_engine* engine) const;

  /// Optimize the next action for the given problem 'p' starting at 'state'
  /// according to inner parameters and provided policy and value_function
  ///
  /// It is mandatory to have called 'prepareOptimizer' with the same problem before
  Eigen::VectorXd planNextAction(const Problem& p, const Eigen::VectorXd& state, const Policy& policy,
                                 const rhoban_fa::FunctionApproximator& value_function,
                                 std::default_random_engine* engine) const;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  /// When the open-loop planner reaches look-ahead without encountering a terminal state, EvaluationPolicy defines how
  /// the expected reward is computed
  enum EvaluationPolicy
  {
    ValueBased,
    PolicyBased
  };

  static enum EvaluationPolicy evaluationPolicyFromString(const std::string& str);
  static std::string evaluationPolicyToString(enum EvaluationPolicy evaluation_policy);

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
