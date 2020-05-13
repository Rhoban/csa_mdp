#include "rhoban_csa_mdp/online_planners/planner_based_policy.h"

#include "rhoban_fa/function_approximator_factory.h"
#include "rhoban_csa_mdp/core/policy_factory.h"
#include "rhoban_csa_mdp/core/problem_factory.h"

namespace csa_mdp
{
PlannerBasedPolicy::PlannerBasedPolicy()
{
}

void PlannerBasedPolicy::init()
{
  planner.prepareOptimizer(*problem);
}

void PlannerBasedPolicy::setActionLimits(const std::vector<Eigen::MatrixXd>& limits)
{
  Policy::setActionLimits(limits);
  fallback_policy->setActionLimits(limits);
}

Eigen::VectorXd PlannerBasedPolicy::getRawAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const
{
  // TODO check state or learningState
  return planner.planNextAction(*problem, state, *fallback_policy, *fallback_value, engine);
}

std::string PlannerBasedPolicy::getClassName() const
{
  return "PlannerBasedPolicy";
}

Json::Value PlannerBasedPolicy::toJson() const
{
  throw std::logic_error("Unimplemented serialization");
  // TODO: to implement serialization an automatic generation of binaries should be available.
  // Json::Value v = Policy::toJson();
  // v["planner"] = planner.toJson();
  // v["problem"] = problem->toJson();
  // v["fallback_policy"] = fallback_policy->toJson();
  // v["fallback_value"] = fallback_value->toJson();// Serialization of FA to json is problematic due to space use
  // return v;
}

void PlannerBasedPolicy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  planner.read(v, "planner", dir_name);
  problem = ProblemFactory().read(v, "problem", dir_name);
  PolicyFactory().tryRead(v, "fallback_policy", dir_name, &fallback_policy);
  std::string value_path;
  rhoban_utils::tryRead(v, "fallback_value", &value_path);
  rhoban_fa::FunctionApproximatorFactory().loadFromFile(value_path, fallback_value);
  init();
}

}  // namespace csa_mdp
