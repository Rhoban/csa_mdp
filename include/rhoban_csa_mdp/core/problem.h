#pragma once

#include "rhoban_csa_mdp/core/policy.h"
#include "rhoban_csa_mdp/core/sample.h"

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <functional>
#include <random>

namespace csa_mdp
{
class Problem : public rhoban_utils::JsonSerializable
{
public:
  /// This inner structure represents the result of a transition
  struct Result
  {
    Eigen::VectorXd successor;
    double reward;
    bool terminal;
  };

  /// Return true if state is terminal
  typedef std::function<bool(const Eigen::VectorXd& state)> TerminalFunction;
  /// Return a value associate to the state
  typedef std::function<double(const Eigen::VectorXd& state)> ValueFunction;
  /// Return action for given state
  typedef std::function<Eigen::VectorXd(const Eigen::VectorXd& state)> Policy;
  /// Sample successor state from a couple (state, action) using provided random engine
  typedef std::function<Eigen::VectorXd(const Eigen::VectorXd& state, const Eigen::VectorXd& action,
                                        std::default_random_engine* engine)>
      TransitionFunction;
  /// Return Reward for the given triplet (state, action, next_state)
  typedef std::function<double(const Eigen::VectorXd& state, const Eigen::VectorXd& action,
                               const Eigen::VectorXd& next_state)>
      RewardFunction;
  /// Return successor, reward and terminal status in a structure
  typedef std::function<Result(const Eigen::VectorXd& state, const Eigen::VectorXd& action,
                               std::default_random_engine* engine)>
      ResultFunction;

private:
  /// What are the state limits of the problem
  Eigen::MatrixXd state_limits;
  /// Each action has its own limits
  std::vector<Eigen::MatrixXd> actions_limits;

  /// Names used for states
  std::vector<std::string> state_names;
  /// action_names[i]: names of the action dimensions for the i-th action
  std::vector<std::vector<std::string>> actions_names;

public:
  Problem();
  virtual ~Problem();

  ResultFunction getResultFunction() const;

  /// Throw an explicit runtime_error if action_id is outside of acceptable range
  void checkActionId(int action_id) const;

  int stateDims() const;
  int getNbActions() const;
  int actionDims(int action_id) const;

  const Eigen::MatrixXd& getStateLimits() const;
  const std::vector<Eigen::MatrixXd>& getActionsLimits() const;
  const Eigen::MatrixXd& getActionLimits(int action_id) const;

  /// Also reset state names
  void setStateLimits(const Eigen::MatrixXd& new_limits);

  /// Also reset action names
  void setActionLimits(const std::vector<Eigen::MatrixXd>& new_limits);

  /// Set the names of the states to "state_0, state_1, ..."
  void resetStateNames();

  /// Set the names of the states to a default value
  void resetActionsNames();

  /// To call after setting properly the limits
  /// throw a runtime_error if names size is not appropriate
  void setStateNames(const std::vector<std::string>& names);

  /// To call after setting properly the limits
  /// throw a runtime_error if names size is not appropriate
  void setActionsNames(const std::vector<std::vector<std::string>>& names);

  /// To call after setting properly the limits
  void setActionNames(int action_id, const std::vector<std::string>& names);

  const std::vector<std::string>& getStateNames() const;
  const std::vector<std::vector<std::string>>& getActionsNames() const;
  /// Return the names of the dimensions for the specified action
  const std::vector<std::string> getActionNames(int action_id) const;

  /// Which state dimensions are used as input for learning (default is all)
  virtual std::vector<int> getLearningDimensions() const;

  /// Filters an exhaustive state to use only states relevant for learning
  Eigen::VectorXd getLearningState(const Eigen::VectorXd& exhaustive_state) const;

  /// Filters an exhaustive state to use only states relevant for learning
  Eigen::MatrixXd getLearningStateLimits() const;

  /// Uses an external random engine and generate successor, reward and terminal check
  virtual Result getSuccessor(const Eigen::VectorXd& state, const Eigen::VectorXd& action,
                              std::default_random_engine* engine) const = 0;

  double sampleRolloutReward(const Eigen::VectorXd& initial_state, const csa_mdp::Policy& policy, int max_horizon,
                             double discount, std::default_random_engine* engine) const;
};

}  // namespace csa_mdp
