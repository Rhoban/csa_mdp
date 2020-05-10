#include "rhoban_csa_mdp/online_planners/open_loop_planner.h"

#include "rhoban_bbo/optimizer_factory.h"

namespace csa_mdp
{
OpenLoopPlanner::OpenLoopPlanner()
  : look_ahead(0), rollouts_per_sample(1), discount(1), evaluation_policy(EvaluationPolicy::ValueBased)
{
}

void OpenLoopPlanner::checkConsistency(const Problem& p) const
{
  if (p.getNbActions() != 1)
  {
    throw std::runtime_error("OpenLoopPlanner::checkConsistency: no support for hybrid action spaces");
  }
  if (look_ahead <= 0)
  {
    throw std::runtime_error("OpenLoopPlanner::checkConsistency: invalid value for look_ahead: '" +
                             std::to_string(look_ahead) + "'");
  }
}

void OpenLoopPlanner::prepareOptimizer(const Problem& p)
{
  checkConsistency(p);
  Eigen::MatrixXd action_limits = p.getActionLimits(0);
  int action_dims = action_limits.rows();
  Eigen::MatrixXd optimizer_limits(action_dims * look_ahead, 2);
  for (int action = 0; action < look_ahead; action++)
  {
    optimizer_limits.block(action * action_dims, 0, action_dims, 2) = action_limits;
  }
  optimizer->setLimits(optimizer_limits);
}

Eigen::VectorXd OpenLoopPlanner::getInitialGuess(const Problem& p, const Eigen::VectorXd& initial_state,
                                                 const Policy& policy, std::default_random_engine* engine) const
{
  // Note: using multiple trials and averaging the actions might have an interest but it also carries out the risk of
  // averaging actions which are pretty different one from another
  int action_dims = p.actionDims(0);
  Eigen::VectorXd initial_guess(look_ahead * action_dims);
  Eigen::VectorXd state = initial_state;
  for (int step = 0; step < look_ahead; step++)
  {
    Eigen::VectorXd action = policy.getAction(p.getLearningState(state), engine);
    Problem::Result result = p.getSuccessor(state, action, engine);
    state = result.successor;
    initial_guess.segment(step * action_dims, action_dims) = action.segment(1, action_dims);
  }
  return initial_guess;
}

double OpenLoopPlanner::sampleLookAheadReward(const Problem& p, const Eigen::VectorXd& initial_state,
                                              const Eigen::VectorXd& next_actions, Eigen::VectorXd* last_state,
                                              bool* is_terminated, std::default_random_engine* engine) const
{
  int action_dims = p.actionDims(0);
  double gain = 1.0;
  double rollout_reward = 0;
  Eigen::VectorXd curr_state = initial_state;
  for (int step = 0; step < look_ahead; step++)
  {
    Eigen::VectorXd action(action_dims + 1);
    action(0) = 0;
    action.segment(1, action_dims) = next_actions.segment(action_dims * step, action_dims);
    Problem::Result result = p.getSuccessor(curr_state, action, engine);
    rollout_reward += gain * result.reward;
    curr_state = result.successor;
    gain *= discount;
    // Stop predicting steps if a terminal state has been reached
    if (result.terminal)
    {
      *is_terminated = true;
      *last_state = curr_state;
      break;
    }
  }
  *last_state = curr_state;
  return rollout_reward;
}

Eigen::VectorXd OpenLoopPlanner::planNextAction(const Problem& p, const Eigen::VectorXd& state, const Policy& policy,
                                                const rhoban_fa::FunctionApproximator& value_function,
                                                std::default_random_engine* engine) const
{
  // Building reward function
  rhoban_bbo::Optimizer::RewardFunc reward_function = [&p, &policy, &value_function, this,
                                                       state](const Eigen::VectorXd& next_actions,
                                                              std::default_random_engine* engine) {
    double total_reward = 0;
    for (int rollout = 0; rollout < this->rollouts_per_sample; rollout++)
    {
      bool trial_terminated = false;
      Eigen::VectorXd final_state;
      double rollout_reward =
          this->sampleLookAheadReward(p, state, next_actions, &final_state, &trial_terminated, engine);
      // If rollout has not ended with a terminal status, use policy to end
      // the trial
      if (!trial_terminated)
      {
        double disc = pow(this->discount, look_ahead);
        double future_reward;
        switch (evaluation_policy)
        {
          case PolicyBased:
            future_reward =
                p.sampleRolloutReward(final_state, policy, trial_length - look_ahead, this->discount, engine);
            break;
          case ValueBased:
            future_reward = value_function.predict(p.getLearningState(final_state), 0);
            break;
          default:
            throw std::logic_error(DEBUG_INFO + "Invalid enum");
        }
        rollout_reward += disc * future_reward;
      }
      total_reward += rollout_reward;
    }
    double avg_reward = total_reward / rollouts_per_sample;
    return avg_reward;
  };
  // Optimizing next actions
  Eigen::VectorXd next_actions;
  if (guess_initial_candidate)
  {
    Eigen::VectorXd initial_guess = getInitialGuess(p, state, policy, engine);
    next_actions = optimizer->train(reward_function, initial_guess, engine);
  }
  else
  {
    next_actions = optimizer->train(reward_function, engine);
  }
  // Only return next action, with a prefix
  int action_dims = p.actionDims(0);
  Eigen::VectorXd prefixed_action(1 + action_dims);
  prefixed_action(0) = 0;
  prefixed_action.segment(1, action_dims) = next_actions;
  return prefixed_action;
}

std::string OpenLoopPlanner::getClassName() const
{
  return "OpenLoopPlanner";
}

Json::Value OpenLoopPlanner::toJson() const
{
  Json::Value v;
  v["optimizer"] = optimizer->toFactoryJson();
  v["look_ahead"] = look_ahead;
  v["trial_length"] = trial_length;
  v["rollouts_per_sample"] = rollouts_per_sample;
  v["discount"] = discount;
  v["guess_initial_candidate"] = guess_initial_candidate;
  v["evaluation_policy"] = evaluationPolicyToString(evaluation_policy);
  return v;
}

void OpenLoopPlanner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  rhoban_bbo::OptimizerFactory().tryRead(v, "optimizer", dir_name, &optimizer);
  rhoban_utils::tryRead(v, "look_ahead", &look_ahead);
  rhoban_utils::tryRead(v, "trial_length", &trial_length);
  rhoban_utils::tryRead(v, "rollouts_per_sample", &rollouts_per_sample);
  rhoban_utils::tryRead(v, "discount", &discount);
  rhoban_utils::tryRead(v, "guess_initial_candidate", &guess_initial_candidate);
  std::string evaluation_policy_str;
  rhoban_utils::tryRead(v, "evaluation_policy", &evaluation_policy_str);
  if (evaluation_policy_str != "")
    evaluation_policy = evaluationPolicyFromString(evaluation_policy_str);
}

enum OpenLoopPlanner::EvaluationPolicy OpenLoopPlanner::evaluationPolicyFromString(const std::string& str)
{
  if (str == "ValueBased")
    return EvaluationPolicy::ValueBased;
  if (str == "PolicyBased")
    return EvaluationPolicy::PolicyBased;
  throw std::runtime_error(DEBUG_INFO + "Unexpected string '" + str + "'");
}

std::string OpenLoopPlanner::evaluationPolicyToString(enum EvaluationPolicy evaluation_policy)
{
  switch (evaluation_policy)
  {
    case EvaluationPolicy::ValueBased:
      return "ValueBased";
    case EvaluationPolicy::PolicyBased:
      return "PolicyBased";
    default:
      throw std::logic_error(DEBUG_INFO + "Invalid evaluation policy");
  }
}

}  // namespace csa_mdp
