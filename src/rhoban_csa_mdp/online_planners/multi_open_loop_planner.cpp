#include "rhoban_csa_mdp/online_planners/multi_open_loop_planner.h"

namespace csa_mdp
{
MultiOpenLoopPlanner::MultiOpenLoopPlanner()
{
}
MultiOpenLoopPlanner::~MultiOpenLoopPlanner()
{
}

void MultiOpenLoopPlanner::checkConsistency(const AgentSelector& as) const
{
  if (as.getNbActions() != 1)  // as
  {
    throw std::runtime_error("MultiOpenLoopPlanner::checkConsistency: no support for hybrid action spaces");
  }
  if (look_ahead <= 0)
  {
    throw std::runtime_error("MultiOpenLoopPlanner::checkConsistency: invalid value for look_ahead: '" +
                             std::to_string(look_ahead) + "'");
  }
}
void MultiOpenLoopPlanner::prepareOptimizer(const Problem& pb, const AgentSelector& as)
{
  //  checkConsistency(p);
  Eigen::MatrixXd action_limits = as.getActionsLimits().front();
  int action_dims = action_limits.rows();
  Eigen::MatrixXd optimizer_limits(action_dims * look_ahead, 2);
  for (int action = 0; action < look_ahead; action++)
  {
    optimizer_limits.block(action * action_dims, 0, action_dims, 2) = action_limits;
  }
  optimizer->setLimits(optimizer_limits);
}

Eigen::VectorXd MultiOpenLoopPlanner::getInitialGuess(const Problem& p, const AgentSelector& as, const int main_agent,
                                                      const Eigen::VectorXd& initial_state, const Policy& policy,
                                                      std::default_random_engine* engine) const
{
  // Note: using multiple trials and averaging the actions might have an interest but it also carries out the risk of
  // averaging actions which are pretty different one from another
  int action_dims = as.getActionsLimits().size();
  Eigen::VectorXd initial_guess(look_ahead * action_dims);
  Eigen::VectorXd state = initial_state;
  for (int step = 0; step < look_ahead; step++)
  {
    Eigen::VectorXd action = policy.getAction(as.getRelevantState(state, main_agent), engine);
    Problem::Result result = p.getSuccessor(state, action, engine);
    state = result.successor;
    initial_guess.segment(step * action_dims, action_dims) = action.segment(1, action_dims);
  }
  return initial_guess;
}

double MultiOpenLoopPlanner::sampleLookAheadReward(const Problem& p, const AgentSelector& as, const Policy& policy,
                                                   const int main_agent, const Eigen::VectorXd& initial_state,
                                                   const Eigen::VectorXd& next_actions, Eigen::VectorXd* last_state,
                                                   bool* is_terminated, std::default_random_engine* engine) const
{
  // to do add action from policy for other robots
  int nb_action = as.getActionsLimits().size();
  int action_dim = nb_action / as.getNbActions();

  double gain = 1.0;
  double rollout_reward = 0;
  Eigen::VectorXd curr_state = initial_state;
  for (int step = 0; step < look_ahead; step++)
  {
    Eigen::VectorXd action(nb_action + 1);
    action(0) = 0;
    action.segment(1 + main_agent * action_dim, action_dim) = next_actions.segment(action_dim * step, action_dim);
    for (int i = 0; i < as.getNbActions(); i++)
    {
      if (i != main_agent)
      {
        action.segment(1 + i * action_dim, action_dim) =
            policy.getAction(as.getRelevantState(initial_state, i), engine);
      }
    }

    Problem::Result result = p.getSuccessor(curr_state, action, engine);  // ok
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

Eigen::VectorXd MultiOpenLoopPlanner::planNextAction(const Problem& p, const AgentSelector& as,
                                                     const Eigen::VectorXd& state, const Policy& policy,
                                                     const rhoban_fa::FunctionApproximator& value_function,
                                                     std::default_random_engine* engine) const
{
  // Building reward function
  rhoban_bbo::Optimizer::RewardFunc reward_function = [&p, &as, &policy, &value_function, this,
                                                       state](const Eigen::VectorXd& next_actions, int main_agent,
                                                              std::default_random_engine* engine) {
    double total_reward = 0;
    for (int rollout = 0; rollout < this->rollouts_per_sample; rollout++)
    {
      bool trial_terminated = false;
      Eigen::VectorXd final_state;
      double rollout_reward = this->sampleLookAheadReward(p, as, policy, main_agent, state, next_actions, &final_state,
                                                          &trial_terminated, engine);
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
            future_reward = value_function.predict(p.getLearningState(final_state), 0);  // ok
            break;
          default:
            throw std::logic_error("Invalid enum");
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
  for (int i = 0; i < as.getNbAgents(); i++)
  {
    if (guess_initial_candidate)  // for each candidate
    {
      Eigen::VectorXd initial_guess = getInitialGuess(p, as, i, state, policy, engine);
      next_actions = optimizer->train(reward_function, initial_guess, engine, main_agent);
    }
    else
    {
      next_actions = optimizer->train(reward_function, engine, main_agent);
    }
  }

  // Only return next action, with a prefix
  int action_dims = p.actionDims(0);
  Eigen::VectorXd prefixed_action(1 + action_dims);
  prefixed_action(0) = 0;
  prefixed_action.segment(1, action_dims) = next_actions.segment(0, action_dims);
  return prefixed_action;
}  // namespace csa_mdp

std::string MultiOpenLoopPlanner::getClassName() const
{
  return "MultiOpenLoopPlanner";
}

}  // namespace csa_mdp
