#pragma once
#include "rhoban_utils/serialization/json_serializable.h"
#include "rhoban_csa_mdp/core/problem.h"

#include <Eigen/Core>

#include <functional>
#include <random>
#include <memory>
namespace csa_mdp
{
class AgentSelector : public rhoban_utils::JsonSerializable
{
public:
  AgentSelector();

  ~AgentSelector();

  int getNbAgents() const;

  /// global method to calculate distance between 2 agents
  virtual double getDist(Eigen::VectorXd agent_1, Eigen::VectorXd agent_2) const;

  /// return the selected agents, global methhod pick the closest of main agent
  virtual Eigen::VectorXd getRevelantAgents(const Eigen::VectorXd& world, const Eigen::MatrixXd& agents,
                                            int main_agent) const;

  /// return state with world + relevant agents
  Eigen::VectorXd getRelevantState(const Eigen::VectorXd& state, int main_agent) const;

  /// return agent without agent in position main_agent
  Eigen::MatrixXd removeMainAgent(Eigen::MatrixXd agents, int main_agent) const;

  /// return limit for relevant state
  const Eigen::MatrixXd getStateLimits() const;

  int getNbActions() const;
  /// return limit for relevant action
  const std::vector<Eigen::MatrixXd> getActionsLimits() const;

  Eigen::VectorXd getAction(Eigen::VectorXd actions, int agent);

  const Eigen::VectorXd mergeActions(std::vector<Eigen::VectorXd> actions) const;

  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;
  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;

protected:
  std::shared_ptr<const Problem> pb;

  int nb_selected_agents;
};

}  // namespace csa_mdp
