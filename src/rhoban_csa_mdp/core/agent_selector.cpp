#include <rhoban_csa_mdp/core/agent_selector.h>

namespace csa_mdp
{
AgentSelector::AgentSelector()
{
}

AgentSelector::~AgentSelector()
{
}

double AgentSelector::getDist(Eigen::VectorXd agent_1, Eigen::VectorXd agent_2)
{
  if (agent_1.size() != agent_2.size())
  {
    std::ostringstream oss;
    oss << "AgentSelector::getDist: agent_1.size() != agent_2.size(), " << agent_1.size() << " != " << agent_2.size();
    throw std::runtime_error(oss.str());
  }
  else
  {
    if (agent_1.size() == 1)
    {
      return fabs(agent_1(0) - agent_2(0));
    }
    else
    {
      double dx = agent_1(0) - agent_2(0);
      double dy = agent_1(1) - agent_2(1);
      return sqrt(dx * dx + dy * dy);
    }
  }
}

Eigen::MatrixXd AgentSelector::getRevelantAgent(const Eigen::VectorXd world, const Eigen::MatrixXd agents,
                                                const int main_agent, int nb_selected_agents)
{
  return agents.topRows(nb_selected_agents);
}

Eigen::MatrixXd AgentSelector::removeMainAgent(Eigen::MatrixXd agents, int main_agent)
{
  unsigned int numRows = agents.rows() - 1;
  unsigned int numCols = agents.cols();
  agents.block(main_agent, 0, numRows - main_agent, numCols) =
      agents.block(main_agent + 1, 0, numRows - main_agent, numCols);
  agents.conservativeResize(numRows, numCols);
  return agents;
}

}  // namespace csa_mdp
