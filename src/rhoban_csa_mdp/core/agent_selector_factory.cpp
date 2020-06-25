#include "rhoban_csa_mdp/core/agent_selector_factory.h"

using csa_mdp::AgentSelector;

namespace csa_mdp
{
std::map<std::string, AgentSelectorFactory::JsonBuilder> AgentSelectorFactory::extra_builders;

AgentSelectorFactory::AgentSelectorFactory()
{
  // This factory does not contain default problems
  for (const auto& entry : extra_builders)
  {
    registerBuilder(entry.first, entry.second);
  }
}

void AgentSelectorFactory::registerExtraBuilder(const std::string& name, Builder b, bool parse_json)
{
  registerExtraBuilder(name, toJsonBuilder(b, parse_json));
}

void AgentSelectorFactory::registerExtraBuilder(const std::string& name, JsonBuilder b)
{
  extra_builders[name] = b;
}

}  // namespace csa_mdp
