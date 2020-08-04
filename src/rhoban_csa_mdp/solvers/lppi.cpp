#include "rhoban_csa_mdp/solvers/lppi.h"

#include "rhoban_csa_mdp/core/random_policy.h"
#include "rhoban_csa_mdp/core/policy_factory.h"

#include "rhoban_fa/constant_approximator.h"
#include "rhoban_fa/function_approximator.h"
#include "rhoban_fa/function_approximator_factory.h"
#include "rhoban_fa/trainer_factory.h"
#include "rhoban_csa_mdp/core/agent_selector_factory.h"

#include "rhoban_random/tools.h"

#include "rhoban_utils/threading/multi_core.h"
#include "rhoban_utils/timing/time_stamp.h"

#include <limits>
#include <mutex>

using namespace rhoban_fa;
using rhoban_utils::TimeStamp;

namespace csa_mdp
{
LPPI::LPPI()
  : min_rollout_length(-1)
  , max_rollout_length(-1)
  , nb_entries(-1)
  , entries_increasement(0)
  , recall_ratio(0.0)
  , best_reward(std::numeric_limits<double>::lowest())
  , use_policy(false)
  , dump_dataset(false)
{
}

LPPI::~LPPI()
{
}

void LPPI::performRollout(Eigen::MatrixXd* states, Eigen::MatrixXd* actions, Eigen::VectorXd* values,
                          std::default_random_engine* engine)
{
  int state_dims = problem->getLearningDimensions().size();
  int action_dims = problem->actionDims(0);
  if (agent_selector)
  {
    action_dims = action_dims / agent_selector->getNbAgents();
    state_dims = state_dims - agent_selector->getNbAgents() + agent_selector->getNbSelectedAgents() + 1;
  }

  // First, run the rollout storing visited states
  std::vector<Eigen::VectorXd> rollout_states, rollout_actions;
  std::vector<double> rollout_rewards;
  Eigen::VectorXd state = problem->getStartingState(engine);  // Exhaustive state
  bool end_with_terminal = false;
  for (int step = 0; step < max_rollout_length; step++)
  {
    // Local optimization of the action
    Eigen::VectorXd action;
    if (agent_selector)
    {
      action = multi_planner.planNextAction(*problem, *agent_selector, state, *policy, *value, engine);
    }
    else
      action = planner.planNextAction(*problem, state, *policy, *value, engine);
    // Applying action, storing results and updating current state
    Problem::Result res = problem->getSuccessor(state, action, engine);
    if (agent_selector)
    {
      for (int i = 0; i < agent_selector->getNbAgents(); i++)
      {
        rollout_states.push_back(agent_selector->getRelevantState(state, i));
        Eigen::VectorXd tmp_action = Eigen::VectorXd::Zero(action_dims + 1);
        tmp_action.segment(1, action_dims) = action.row(1 + i);
        rollout_actions.push_back(tmp_action);
        rollout_rewards.push_back(res.reward);
      }
    }

    else
    {
      rollout_states.push_back(problem->getLearningState(state));
      rollout_actions.push_back(action);
      rollout_rewards.push_back(res.reward);
    }

    // Stop if we obtained a terminal status, otherwise, update current state
    state = res.successor;
    if (res.terminal)
    {
      end_with_terminal = true;
      break;
    }
  }
  // Now, fill states, actions and values by going back
  int rollout_length = rollout_states.size();
  int last_idx_used = rollout_length - 1;
  double value = 0;
  // If end of rollout was the result of an horizon end, ensure that
  // states, actions and values returned
  if (!end_with_terminal)
  {
    last_idx_used = rollout_length - min_rollout_length;
    for (int idx = rollout_length - 1; idx > last_idx_used; idx--)
    {
      value = value * discount + rollout_rewards[idx];
    }
  }
  // Initalize and fill results

  int rollout_entries = last_idx_used + 1;

  (*states) = Eigen::MatrixXd(state_dims, rollout_entries);
  (*actions) = Eigen::MatrixXd(1 + action_dims, rollout_entries);
  (*values) = Eigen::VectorXd(rollout_entries);
  for (int idx = last_idx_used; idx >= 0; idx--)
  {
    value = value * discount + rollout_rewards[idx];
    // TODO: If min_rollout_length is not set, a segfault occurs here:
    // - Investigate and eventually enforce min_rollout_length
    states->col(idx) = rollout_states[idx];
    actions->col(idx) = rollout_actions[idx];
    (*values)(idx) = value;
  }
  if (verbosity > 1)
    std::cout << "-----" << std::endl
              << "rollout_length : " << rollout_length << std::endl
              << "final_state : " << state.transpose() << std::endl
              << "value : " << value << std::endl;
}

void LPPI::performRollouts(Eigen::MatrixXd* states, Eigen::MatrixXd* actions, Eigen::VectorXd* values,
                           std::default_random_engine* engine)
{
  int state_dims = problem->getLearningDimensions().size();
  int action_dims = problem->actionDims(0);
  if (agent_selector)
  {
    action_dims = action_dims / agent_selector->getNbAgents();
    state_dims = state_dims - agent_selector->getNbAgents() + agent_selector->getNbSelectedAgents() + 1;
  }

  int entry_count = 0;
  (*states) = Eigen::MatrixXd(state_dims, nb_entries);
  (*actions) = Eigen::MatrixXd(1 + action_dims, nb_entries);
  (*values) = Eigen::VectorXd(nb_entries);
  // If samples are remembered from previous iterations, start by including them
  if (recall_states.cols() > 0)
  {
    states->block(0, 0, state_dims, recall_states.cols()) = recall_states;
    actions->block(0, 0, 1 + action_dims, recall_actions.cols()) = recall_actions;
    values->segment(0, recall_values.rows()) = recall_values;
    entry_count += recall_states.cols();
  }

  std::mutex mutex;  // Ensures only one thread modifies common properties at the same time
  // TODO: add another StochasticTask which does not depend on start_idx and end_idx eventually
  rhoban_utils::MultiCore::StochasticTask thread_task = [this, &mutex, &state_dims, &action_dims, states, actions,
                                                         values, &entry_count](int start_idx, int end_idx,
                                                                               std::default_random_engine* engine) {
    (void)start_idx;
    (void)end_idx;
    while (entry_count < this->nb_entries)
    {
      Eigen::MatrixXd rollout_states, rollout_actions;
      Eigen::VectorXd rollout_values;
      this->performRollout(&rollout_states, &rollout_actions, &rollout_values, engine);
      int nb_new_entries = rollout_states.cols();
      // Updating content
      mutex.lock();
      if (entry_count >= this->nb_entries)
      {
        mutex.unlock();
        return;
      }

      // If there is too much new entries, remove some to have exactly the
      // requested number
      if (entry_count + nb_new_entries > this->nb_entries)
      {
        nb_new_entries = this->nb_entries - entry_count;
        rollout_states = rollout_states.block(0, 0, state_dims, nb_new_entries);
        rollout_actions = rollout_actions.block(0, 0, 1 + action_dims, nb_new_entries);
        rollout_values = rollout_values.segment(0, nb_new_entries);
      }
      states->block(0, entry_count, state_dims, nb_new_entries) = rollout_states;
      actions->block(0, entry_count, 1 + action_dims, nb_new_entries) = rollout_actions;
      values->segment(entry_count, nb_new_entries) = rollout_values;
      entry_count += nb_new_entries;
      if (verbosity > 0)
        std::cout << "Entry count: " << entry_count << std::endl;
      mutex.unlock();
    }
  };
  std::vector<std::default_random_engine> engines;
  int local_nb_threads = std::min(nb_threads, nb_entries);
  engines = rhoban_random::getRandomEngines(local_nb_threads, engine);
  rhoban_utils::MultiCore::runParallelStochasticTask(thread_task, local_nb_threads, &engines);

  if (dump_dataset)
  {
    std::cout << "Writing dataset.json" << std::endl;
    Json::StyledWriter writer;
    Json::Value content;
    content["states"] = rhoban_utils::matrix2Json(*states);
    content["actions"] = rhoban_utils::matrix2Json(*actions);
    content["values"] = rhoban_utils::vector2Json(*values);
    // Prepare output stream
    // TODO: error treatment
    std::ofstream output("dataset.json");
    output << writer.write(content);
  }
}
void LPPI::init(std::default_random_engine* engine)
{
  if (problem->getNbActions() != 1)
  {
    throw std::runtime_error("LPPI::performRollouts: no support for hybrid action spaces");
  }
  (void)engine;
  if (!policy)
  {
    policy = std::unique_ptr<Policy>(new RandomPolicy);
    if (agent_selector)
      policy->setActionLimits(agent_selector->getActionsLimits());
    else
      policy->setActionLimits(problem->getActionsLimits());
  }
  if (!value)
  {
    Eigen::VectorXd default_value(1);
    default_value(0) = 0;
    value = std::unique_ptr<FunctionApproximator>(new ConstantApproximator(default_value));
  }
  if (!policy_trainer)
  {
    throw std::runtime_error("LPPI::init: no policy trainer");
  }
  if (!value_trainer)
  {
    throw std::runtime_error("LPPI::init: no policy trainer");
  }
  if (entries_increasement != 0)
  {
    increase_last_iteration = true;
  }
  best_reward = std::numeric_limits<double>::lowest();
}

void LPPI::update(std::default_random_engine* engine)
{
  // Acquiring entries by performing rollouts with online planner
  Eigen::MatrixXd states, actions;
  Eigen::VectorXd values;
  TimeStamp start = TimeStamp::now();
  if (verbosity > 0)
    std::cout << "performing rollouts" << std::endl;
  performRollouts(&states, &actions, &values, engine);
  TimeStamp mid1 = TimeStamp::now();
  writeTime("performRollouts", diffSec(start, mid1));
  // Updating both policy and value based on actions
  if (verbosity > 0)
    std::cout << "Training value" << std::endl;
  updateValues(states, values);
  TimeStamp mid2 = TimeStamp::now();
  writeTime("updateValue", diffSec(mid1, mid2));
  if (verbosity > 0)
    std::cout << "Training policy" << std::endl;
  std::unique_ptr<rhoban_fa::FunctionApproximator> new_policy_fa = updatePolicy(states, actions);
  TimeStamp mid3 = TimeStamp::now();
  writeTime("updatePolicy", diffSec(mid2, mid3));
  if (verbosity > 0)
    std::cout << "Building policy" << std::endl;
  std::unique_ptr<Policy> new_policy = buildPolicy(*new_policy_fa);
  if (verbosity > 0)
    std::cout << "Evaluating policy" << std::endl;
  if (agent_selector)
    last_score = evaluateMultiPolicy(100, engine);
  else
    last_score = evaluatePolicy(*new_policy, engine);
  TimeStamp end = TimeStamp::now();
  writeTime("evalPolicy", diffSec(mid3, end));
  if (verbosity > 0)
    std::cout << "New reward: " << last_score << std::endl;
  if (last_score > best_reward)
  {
    policy = std::move(new_policy);
    if (agent_selector)
      policy->setActionLimits(agent_selector->getActionsLimits());

    policy_fa = std::move(new_policy_fa);
    value->save("value.bin");
    policy_fa->save("policy_fa.bin");
    best_reward = last_score;
  }
  else if (entries_increasement != 0)
  {
    if (!increase_last_iteration)
    {
      nb_entries = nb_entries * 2;
      if (verbosity > 0)
        std::cout << "double number of entries, new nb :" << nb_entries << std::endl;
      entries_increasement--;
      increase_last_iteration = true;
    }
    else
    {
      increase_last_iteration = false;
    }
  }

  publishIteration();
  updateMemory(states, actions, values, engine);
}

void LPPI::updateValues(const Eigen::MatrixXd& states, const Eigen::VectorXd& values)
{
  Eigen::MatrixXd state_limits;

  if (agent_selector)
    state_limits = agent_selector->getStateLimits();
  else
    state_limits = problem->getLearningStateLimits();
  if (value)
  {
    value = value_trainer->train(states, values, state_limits, *value);
  }
  else
  {
    value = value_trainer->train(states, values, state_limits);
  }
}

void LPPI::updateMemory(const Eigen::MatrixXd& states, const Eigen::MatrixXd& actions, const Eigen::VectorXd& values,
                        std::default_random_engine* engine)
{
  int recall_entries = std::min(states.cols(), (long int)std::floor(recall_ratio * nb_entries));
  if (recall_entries == 0)
    return;
  std::cout << "Recalling " << recall_entries << " entries" << std::endl;
  recall_states = Eigen::MatrixXd(states.rows(), recall_entries);
  recall_actions = Eigen::MatrixXd(actions.rows(), recall_entries);
  recall_values = Eigen::VectorXd(recall_entries);
  std::vector<size_t> recall_indices = rhoban_random::getKDistinctFromN(recall_entries, states.cols(), engine);
  for (int idx = 0; idx < recall_entries; idx++)
  {
    size_t dataset_idx = recall_indices[idx];
    recall_states.col(idx) = states.col(dataset_idx);
    recall_actions.col(idx) = actions.col(dataset_idx);
    recall_values(idx) = values(dataset_idx);
  }
}

std::unique_ptr<FunctionApproximator> LPPI::updatePolicy(const Eigen::MatrixXd& states,
                                                         const Eigen::MatrixXd& actions) const
{
  Eigen::MatrixXd state_limits;

  if (agent_selector)
    state_limits = agent_selector->getStateLimits();
  else
    state_limits = problem->getLearningStateLimits();

  std::unique_ptr<FunctionApproximator> new_policy_fa;
  if (policy_fa)
  {
    new_policy_fa = policy_trainer->train(states, actions.transpose(), state_limits, *policy_fa);
  }
  else
  {
    new_policy_fa = policy_trainer->train(states, actions.transpose(), state_limits);
  }
  return new_policy_fa;
}

double LPPI::evaluateMultiPolicy(int nb_evaluations, std::default_random_engine* engine) const
{
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rhoban_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Rewards + visited_states are computed by different threads and stored in the same vector
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  int action_dims = agent_selector->getActionsLimits().size();

  // The task which has to be performed :
  rhoban_utils::MultiCore::StochasticTask task = [this, &rewards, &action_dims](int start_idx, int end_idx,
                                                                                std::default_random_engine* engine) {
    for (int idx = start_idx; idx < end_idx; idx++)
    {
      Eigen::VectorXd state = problem->getStartingState(engine);
      double gain = 1.0;
      for (int step = 0; step < trial_length; step++)
      {
        Eigen::VectorXd action(1 + action_dims * agent_selector->getNbAgents());
        action(0) = 0;
        for (int i = 0; i < agent_selector->getNbAgents(); i++)
        {
          action.segment(1 + i, action_dims) =
              policy->getAction(agent_selector->getRelevantState(state, i), engine).segment(1, action_dims);
        }
        Problem::Result result = problem->getSuccessor(state, action, engine);
        double step_reward = result.reward;
        state = result.successor;
        rewards(idx) += gain * step_reward;
        gain = gain * discount;
        if (result.terminal)
          break;
      }
    }
  };
  // Running computation
  rhoban_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}  // namespace csa_mdp

void LPPI::setNbThreads(int nb_threads)
{
  BlackBoxLearner::setNbThreads(nb_threads);
  if (value_trainer)
  {
    value_trainer->setNbThreads(nb_threads);
  }
  if (policy_trainer)
  {
    policy_trainer->setNbThreads(nb_threads);
  }
}

std::string LPPI::getClassName() const
{
  return "LPPI";
}

Json::Value LPPI::toJson() const
{
  Json::Value v = BlackBoxLearner::toJson();
  throw std::runtime_error("LPPI::toJson: Not implemented");
}

void LPPI::fromJson(const Json::Value& v, const std::string& dir_name)
{
  BlackBoxLearner::fromJson(v, dir_name);
  // as
  AgentSelectorFactory().tryRead(v, "agent_selector", dir_name, &agent_selector);

  if (agent_selector)
  {
    multi_planner.read(v, "planner", dir_name);
  }
  else
  {
    planner.read(v, "planner", dir_name);
  }
  TrainerFactory().tryRead(v, "value_trainer", dir_name, &value_trainer);
  TrainerFactory().tryRead(v, "policy_trainer", dir_name, &policy_trainer);
  FunctionApproximatorFactory().tryRead(v, "value", dir_name, &value);
  PolicyFactory().tryRead(v, "policy", dir_name, &policy);
  rhoban_utils::tryRead(v, "min_rollout_length", &min_rollout_length);
  rhoban_utils::tryRead(v, "max_rollout_length", &max_rollout_length);
  rhoban_utils::tryRead(v, "nb_entries", &nb_entries);
  rhoban_utils::tryRead(v, "entries_increasement", &entries_increasement);
  rhoban_utils::tryRead(v, "recall_ratio", &recall_ratio);
  rhoban_utils::tryRead(v, "use_policy", &use_policy);
  rhoban_utils::tryRead(v, "dump_dataset", &dump_dataset);
  if (recall_ratio < 0 || recall_ratio > 1.0)
  {
    throw std::logic_error("Invalid value for recall_ration: " + std::to_string(recall_ratio));
  }
  if (agent_selector)
    multi_planner.prepareOptimizer(*agent_selector);
  else
    planner.prepareOptimizer(*problem);

  // Update value_trainer and policy_trainer number of threads
  setNbThreads(nb_threads);
}

std::vector<std::string> LPPI::getMetaColumns() const
{
  std::vector<std::string> result = BlackBoxLearner::getMetaColumns();
  result.push_back("nb_entries");
  result.push_back("entries_increasement");
  return result;
}
std::map<std::string, std::string> LPPI::getMetaData() const
{
  std::map<std::string, std::string> result = BlackBoxLearner::getMetaData();
  result["nb_entries"] = std::to_string(nb_entries);
  result["entries_increasement"] = std::to_string(entries_increasement);
  return result;
}

}  // namespace csa_mdp
