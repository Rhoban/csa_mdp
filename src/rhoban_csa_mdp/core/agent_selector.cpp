#include <rhoban_csa_mdp/core/agent_selector.h>
#include "rhoban_csa_mdp/core/problem_factory.h"

namespace csa_mdp
{
AgentSelector::AgentSelector() : nb_selected_agents(2)
{
}

AgentSelector::~AgentSelector()
{
}

bool sortByScore(const std::pair<double, Eigen::VectorXd>& a, const std::pair<double, Eigen::VectorXd>& b)
{
  return (a.first < b.first);
}

int AgentSelector::getNbAgents() const
{
  return this->pb->getNbAgents();
}

double AgentSelector::getDist(Eigen::VectorXd agent_1, Eigen::VectorXd agent_2) const
{
  if (agent_1.size() != agent_2.size())
  {
    std::ostringstream oss;
    oss << "AgentSelector::getDist: agent_1.size() != agent_2.size(), " << agent_1.size() << " != " << agent_2.size();
    throw std::runtime_error(oss.str());
  }
  else
  {
    return (agent_1 - agent_2).norm();
  }
}

Eigen::VectorXd AgentSelector::getRevelantAgents(const Eigen::VectorXd& world, const Eigen::MatrixXd& agents,
                                                 int main_agent) const
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
  Eigen::VectorXd agent_state(nb_selected_agents + 1 * agents.cols());
  agent_state(0) = agents(main_agent, 0);
  for (int i = 0; i < nb_selected_agents; i++)
    agent_state(i + 1) = score_agents.at(i).second(0);
  return agent_state;
}

Eigen::VectorXd AgentSelector::getRelevantState(const Eigen::VectorXd& state, int main_agent) const
{
  // split world and state
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> split_state = this->pb->splitMultiAgentState(state);
  // get relevant agents
  Eigen::VectorXd relevant_agents = getRevelantAgents(split_state.first, split_state.second, main_agent);
  // merge world and relevant agent
  Eigen::VectorXd relevant_state(split_state.first.size() + nb_selected_agents + 1);
  relevant_state << split_state.first, relevant_agents;
  return relevant_state;
}

const Eigen::MatrixXd AgentSelector::getStateLimits() const
{
  // get limits of world and n agents
  int nb_dimensions = this->pb->getNbStaticElements() + 1;
  nb_dimensions += nb_selected_agents;

  Eigen::MatrixXd relevant_state_limits(nb_dimensions, 2);
  Eigen::MatrixXd state_limits = this->pb->getStateLimits();
  for (int i = 0; i < nb_dimensions; i++)
  {
    relevant_state_limits.row(i) = state_limits.row(i);
  }

  return relevant_state_limits;
}

int AgentSelector::getNbActions() const
{
  return this->pb->getNbActions();
}

const std::vector<Eigen::MatrixXd> AgentSelector::getActionsLimits() const
{
  // get limits for one action

  int nb_agents = this->pb->getNbAgents();
  std::vector<Eigen::MatrixXd> action_limits = this->pb->getActionsLimits();
  int action_dimension = action_limits.front().rows() / nb_agents;

  std::vector<Eigen::MatrixXd> relevant_action_limits;
  for (std::vector<Eigen::MatrixXd>::iterator it = action_limits.begin(); it != action_limits.end(); ++it)
  {
    Eigen::MatrixXd full_action = *it;
    relevant_action_limits.push_back(full_action.block(0, 0, action_dimension, full_action.cols()));
  }

  return relevant_action_limits;
}

Eigen::VectorXd AgentSelector::getAction(Eigen::VectorXd actions, int agent)
{
  int nb_agents = this->pb->getNbAgents();
  int action_dimension = actions.size() / nb_agents;
  return actions.segment(agent * action_dimension, action_dimension);
}

const Eigen::VectorXd AgentSelector::mergeActions(std::vector<Eigen::VectorXd> actions) const
{
  int nb_actions = actions.size();
  Eigen::VectorXd merged_actions(nb_actions * actions.front().size() + 1);
  merged_actions << 0;

  for (int i = 0; i < actions.size(); i++)
  {
    Eigen::VectorXd a = actions.at(i);
    merged_actions(i + 1) = a(0);
  }
  return merged_actions;
}

Eigen::MatrixXd AgentSelector::removeMainAgent(Eigen::MatrixXd agents, int main_agent) const
{
  unsigned int num_rows = agents.rows() - 1;
  unsigned int num_cols = agents.cols();
  agents.block(main_agent, 0, num_rows - main_agent, num_cols) =
      agents.block(main_agent + 1, 0, num_rows - main_agent, num_cols);
  agents.conservativeResize(num_rows, num_cols);
  return agents;
}

void AgentSelector::fromJson(const Json::Value& v, const std::string& dir_name)
{
  pb = ProblemFactory().read(v, "model", dir_name);
  rhoban_utils::tryRead(v, "nb_selected_agents", &nb_selected_agents);
}

Json::Value AgentSelector::toJson() const
{
  throw std::runtime_error("AgentSelector::toJson: Not implemented");
}

std::string AgentSelector::getClassName() const
{
  return "AgentSelector";
}
}  // namespace csa_mdp
