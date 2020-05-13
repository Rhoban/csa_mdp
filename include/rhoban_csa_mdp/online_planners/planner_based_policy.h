#pragma once

#include "rhoban_csa_mdp/core/policy.h"
#include "rhoban_csa_mdp/core/problem.h"

#include "rhoban_csa_mdp/online_planners/open_loop_planner.h"

namespace csa_mdp
{
/// A PlannerBasedPolicy uses a planner to explore short term options while relying on a fallback policy and a fallback
/// value function to estimate long term results
class PlannerBasedPolicy : public Policy
{
public:
  PlannerBasedPolicy();

  void init() override;
  void setActionLimits(const std::vector<Eigen::MatrixXd>& limits) override;
  Eigen::VectorXd getRawAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const override;

  std::string getClassName() const override;
  Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  OpenLoopPlanner planner;
  std::unique_ptr<Problem> problem;
  std::unique_ptr<Policy> fallback_policy;
  std::unique_ptr<rhoban_fa::FunctionApproximator> fallback_value;
};
}  // namespace csa_mdp
