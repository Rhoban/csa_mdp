#pragma once

#include "rhoban_csa_mdp/core/policy.h"

#include "rhoban_fa/function_approximator.h"

#include <memory>
#include <random>

namespace csa_mdp
{
/// This class implements a policy using a function approximator
class FAPolicy : public Policy
{
public:
  FAPolicy();
  FAPolicy(std::unique_ptr<rhoban_fa::FunctionApproximator> fa);

  void setRandomness(bool apply_noise);

  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state) override;
  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const override;

  std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  void saveFA(const std::string& filename) const;
  std::unique_ptr<rhoban_fa::FATree> extractFATree() const override;

private:
  /// The policies
  std::unique_ptr<rhoban_fa::FunctionApproximator> fa;

  /// Is the noise applied when requesting raw action?
  bool apply_noise;

  /// Random generator used to select noisy actions
  std::default_random_engine engine;
};

}  // namespace csa_mdp
