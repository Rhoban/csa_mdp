#include "rhoban_csa_mdp/core/policy_factory.h"

#include "rhoban_csa_mdp/core/fa_policy.h"
#include "rhoban_csa_mdp/core/forests_policy.h"
#include "rhoban_csa_mdp/core/monte_carlo_policy.h"
#include "rhoban_csa_mdp/core/opportunist_policy.h"
#include "rhoban_csa_mdp/core/random_policy.h"
#include "rhoban_csa_mdp/online_planners/planner_based_policy.h"

namespace csa_mdp
{
std::map<std::string, PolicyFactory::Builder> PolicyFactory::extra_builders;

PolicyFactory::PolicyFactory()
{
  registerBuilder("FAPolicy", []() { return std::unique_ptr<Policy>(new FAPolicy); });
  registerBuilder("ForestsPolicy", []() { return std::unique_ptr<Policy>(new ForestsPolicy); });
  registerBuilder("MonteCarloPolicy", []() { return std::unique_ptr<Policy>(new MonteCarloPolicy); });
  registerBuilder("OpportunistPolicy", []() { return std::unique_ptr<Policy>(new OpportunistPolicy); });
  registerBuilder("RandomPolicy", []() { return std::unique_ptr<Policy>(new RandomPolicy); });
  registerBuilder("PlannerBasedPolicy", []() { return std::unique_ptr<Policy>(new PlannerBasedPolicy); });
  for (const auto& entry : extra_builders)
  {
    registerBuilder(entry.first, entry.second);
  }
}

void PolicyFactory::registerExtraBuilder(const std::string& name, Builder b)
{
  extra_builders[name] = b;
}

}  // namespace csa_mdp
