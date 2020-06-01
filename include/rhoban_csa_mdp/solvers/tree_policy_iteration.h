#pragma once

#include "rhoban_csa_mdp/solvers/black_box_learner.h"

namespace csa_mdp
{
class TreePolicyIteration : public BlackBoxLearner
{
public:
  TreePolicyIteration();
  virtual ~TreePolicyIteration();

  virtual void init(std::default_random_engine* engine) override;
  virtual void update(std::default_random_engine* engine) override;

  virtual void setNbThreads(int nb_threads) override;

  /// Update value function based on current policy
  void updateValue(std::default_random_engine* engine);

  /// Return a new policy based on current value function
  std::unique_ptr<Policy> updatePolicy(std::default_random_engine* engine);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

protected:
  /// Best expected score reached by a policy at the moment
  double best_score;

  /// Current approximation of the value function
  std::unique_ptr<rhoban_fa::FunctionApproximator> value;

  /// The approximator used to update the value function
  std::unique_ptr<ValueApproximator> value_approximator;

  /// The optimizer used to train policies
  std::unique_ptr<rhoban_fa::OptimizerTrainer> policy_trainer;

  /// If enabled, then memory of the policy_trainer is emptied before each
  /// training process
  bool memoryless_policy_trainer;

  /// Does the learner use an approximation of the value
  bool use_value_approximator;
};

}  // namespace csa_mdp
