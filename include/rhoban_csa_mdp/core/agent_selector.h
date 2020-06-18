#pragma once
#include "rhoban_utils/serialization/json_serializable.h"
#include <Eigen/Core>

#include <functional>
#include <random>

namespace csa_mdp
{
class AgentSelector : public rhoban_utils::JsonSerializable
{
public:
  AgentSelector();

  ~AgentSelector();

  virtual double getDist(Eigen::VectorXd agent_1, Eigen::VectorXd agent_2);
  virtual Eigen::MatrixXd getRevelantAgent(const Eigen::VectorXd world, const Eigen::MatrixXd agents,
                                           const int main_agent, int nb_selected_agents);

  Eigen::MatrixXd removeMainAgent(Eigen::MatrixXd agents, int main_agent);
};

}  // namespace csa_mdp
