#include <rhoban_csa_mdp/core/agent_selector.h>
#include "rhoban_csa_mdp/core/problem_factory.h"

namespace csa_mdp
{
AgentSelector::AgentSelector() : nb_selected_agents(5)
{
}

AgentSelector::~AgentSelector()
{
}

bool sortByScore(const std::pair<double, Eigen::VectorXd>& a, const std::pair<double, Eigen::VectorXd>& b)
{
  return (a.first < b.first);
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
      // one dimension agent
      return fabs(agent_1(0) - agent_2(0));
    }
    else
    {
      // two dimensions
      double dx = agent_1(0) - agent_2(0);
      double dy = agent_1(1) - agent_2(1);
      return sqrt(dx * dx + dy * dy);
    }
  }
}

Eigen::VectorXd AgentSelector::getRevelantAgents(const Eigen::VectorXd world, const Eigen::MatrixXd agents,
                                                 const int main_agent)
{
  Eigen::MatrixXd agents_to_score = removeMainAgent(agents, main_agent);
  std::vector<std::pair<double, Eigen::VectorXd>> score_agents;
  for (int r = 0; r < agents_to_score.rows(); r++)
  {
    // calculate distance between main agent and other robots
    double score = getDist(agents.row(main_agent), agents_to_score.row(r));
    score_agents.push_back(std::make_pair(score, agents.row(r)));
  }

  // select the closests agents
  std::sort(score_agents.begin(), score_agents.end(), sortByScore);
  Eigen::VectorXd agent_state(nb_selected_agents * agents.cols());
  for (int i = 0; i < nb_selected_agents; i++)
    agent_state << score_agents.at(i).second;
  return agent_state;
}

Eigen::VectorXd AgentSelector::getRelevantState(const Eigen::VectorXd state, const int main_agent)
{
  // split world and state
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> split_state = this->pb->splitMultiAgentState(state);
  // get relevant agents
  Eigen::VectorXd relevant_agents = getRevelantAgents(split_state.first, split_state.second, main_agent);
  // merge world and relevant agent
  Eigen::VectorXd relevant_state(split_state.first.size() + nb_selected_agents);
  relevant_state << split_state.first, relevant_agents;
  return relevant_state;
}

const Eigen::MatrixXd& AgentSelector::getStateLimits() const
{
  // get limits of world and n agents
  int nb_dimensions = this->pb->getNbStaticElements();
  nb_dimensions += nb_selected_agents;

  Eigen::MatrixXd relevant_state_limits(nb_dimensions, 2);
  Eigen::MatrixXd state_limits = this->pb->getStateLimits();
  for (int i = 0; i < nb_dimensions; i++)
  {
    relevant_state_limits << state_limits.row(i);
  }
  return relevant_state_limits;
}

int AgentSelector::getNbActions() const
{
  // one because we are dealing with one agent at the time
  return 1;
}

const Eigen::MatrixXd& AgentSelector::getActionsLimits() const
{
  // get limits for one action
  std::vector<Eigen::MatrixXd> action_limits = this->pb->getActionsLimits();
  return action_limits.front();
}

Eigen::VectorXd AgentSelector::getAction(Eigen::VectorXd actions, int agent)
{
  int nb_agents = this->pb->getNbAgents();
  int action_dimension = actions.size() / nb_agents;
  return actions.segment(agent * action_dimension, action_dimension),
}

const Eigen::VectorXd& AgentSelector::mergeActions(std::vector<Eigen::VectorXd> actions) const
{
  int nb_actions = actions.size();
  Eigen::VectorXd merged_actions(nb_actions * actions.front().size());
  for (std::vector<Eigen::VectorXd>::iterator it = actions.begin(); it != actions.end(); ++it)
  {
    merged_actions << *it;
  }
  return merged_actions;
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

void AgentSelector::fromJson(const Json::Value& v, const std::string& dir_name)
{
  pb = ProblemFactory().read(v, "model", dir_name);
  rhoban_utils::tryRead(v, "nb_selected_agents", &nb_selected_agents);
}
}  // namespace csa_mdp
