#include "rhoban_csa_mdp/solvers/tree_policy_iteration.h"

#include "rhoban_csa_mdp/core/fa_policy.h"
#include "rhoban_csa_mdp/core/policy_factory.h"
#include "rhoban_csa_mdp/core/random_policy.h"
#include "rhoban_csa_mdp/value_approximators/value_approximator_factory.h"

#include "rhoban_fa/optimizer_trainer_factory.h"

using rhoban_utils::TimeStamp;

namespace csa_mdp
{
TreePolicyIteration::TreePolicyIteration()
  : best_score(std::numeric_limits<double>::lowest()), memoryless_policy_trainer(true), use_value_approximator(true)
{
}

TreePolicyIteration::~TreePolicyIteration()
{
}

void TreePolicyIteration::init(std::default_random_engine* engine)
{
  // Initialize a random policy if no policy has been provided
  if (!policy)
  {
    policy = std::unique_ptr<Policy>(new RandomPolicy());
  }
  // Reset policy if necessary
  policy->init();
  policy->setActionLimits(problem->getActionsLimits());
  last_score = evaluatePolicy(*policy, engine);
  publishIteration();
}

void TreePolicyIteration::update(std::default_random_engine* engine)
{
  // Update function approximatiors
  if (use_value_approximator)
  {
    TimeStamp value_start = TimeStamp::now();
    updateValue(engine);
    TimeStamp value_end = TimeStamp::now();
    writeTime("updateValue", diffSec(value_start, value_end));
  }
  // Always update policy
  TimeStamp policy_start = TimeStamp::now();
  std::unique_ptr<Policy> new_policy = updatePolicy(engine);
  TimeStamp policy_end = TimeStamp::now();
  writeTime("updatePolicy", diffSec(policy_start, policy_end));
  // Evaluate the policy expected score
  last_score = evaluatePolicy(*new_policy, engine);
  TimeStamp evaluation_end = TimeStamp::now();
  writeTime("evaluation", diffSec(policy_end, evaluation_end));
  publishIteration();
  // Save value used to build policy if enabled
  if (use_value_approximator)
  {
    std::ostringstream oss_v;
    oss_v << "value" << iterations << ".bin";
    value->save(oss_v.str());
  }
  // Saving policy used at this iteration
  const FAPolicy& fap = dynamic_cast<const FAPolicy&>(*new_policy);
  std::ostringstream oss_p;
  oss_p << "policy" << iterations << ".bin";
  fap.saveFA(oss_p.str());
  // Replace policy if it had a better score
  if (last_score > best_score)
  {
    best_score = last_score;
    policy = std::move(new_policy);
  }
}

void TreePolicyIteration::updateValue(std::default_random_engine* engine)
{
  value = value_approximator->train(*policy, *problem,
                                    [this](const Eigen::VectorXd& state) {
                                      if (!this->value)
                                        return 0.0;
                                      double mean, var;
                                      this->value->predict(state, mean, var);
                                      return mean;
                                    },
                                    discount, engine);
}

std::unique_ptr<Policy> TreePolicyIteration::updatePolicy(std::default_random_engine* engine)
{
  rhoban_fa::OptimizerTrainer::RewardFunction reward_func =
      [this](const Eigen::VectorXd& parameters, const Eigen::VectorXd& actions, std::default_random_engine* engine) {
        // Computing first step reward and successor
        Eigen::VectorXd state = parameters;
        Problem::Result result = problem->getSuccessor(state, actions, engine);
        // If a value approximator is used, only one step is required
        if (this->use_value_approximator)
        {
          double value, value_var;
          this->value->predict(result.successor, value, value_var);
          return result.reward + this->discount * value;
        }
        // Otherwise, do multiple steps
        double reward = result.reward;
        state = result.successor;
        double gain = this->discount;
        for (int step = 1; step < this->trial_length; step++)
        {
          // Stop iterations if we reached a terminal state
          if (result.terminal)
            break;
          // Computing step
          Eigen::VectorXd action = policy->getAction(state, engine);
          result = problem->getSuccessor(state, action, engine);
          // Accumulating reward
          reward += result.reward * gain;
          // Updating values
          state = result.successor;
          gain *= discount;
        }
        return reward;
      };
  if (memoryless_policy_trainer)
  {
    policy_trainer->reset();
  }

  if (problem->getNbActions() != 1)
  {
    throw std::runtime_error("TreePolicyIteration::updatePolicy: not able to handle multiple actions");
  }

  policy_trainer->setParametersLimits(problem->getStateLimits());
  policy_trainer->setActionsLimits(problem->getActionLimits(0));
  std::unique_ptr<rhoban_fa::FunctionApproximator> fa;
  fa = policy_trainer->train(reward_func, engine);
  std::unique_ptr<Policy> new_policy(new FAPolicy(std::move(fa)));
  new_policy->setActionLimits(problem->getActionsLimits());
  return std::move(new_policy);
}

void TreePolicyIteration::setNbThreads(int nb_threads_)
{
  BlackBoxLearner::setNbThreads(nb_threads_);
  if (value_approximator)
    value_approximator->setNbThreads(nb_threads);
  if (policy_trainer)
    policy_trainer->setNbThreads(nb_threads);
}

std::string TreePolicyIteration::getClassName() const
{
  return "TreePolicyIteration";
}

Json::Value TreePolicyIteration::toJson() const
{
  throw std::logic_error("TreePolicyIteration::toJson: not implemented");
}

void TreePolicyIteration::fromJson(const Json::Value& v, const std::string& dir_name)
{
  // Calling parent implementation
  BlackBoxLearner::fromJson(v, dir_name);
  // Reading simple parameters
  rhoban_utils::tryRead(v, "memoryless_policy_trainer", &memoryless_policy_trainer);
  rhoban_utils::tryRead(v, "use_value_approximator", &use_value_approximator);
  // Read value approximator if necessary
  if (use_value_approximator)
  {
    value_approximator = ValueApproximatorFactory().read(v, "value_approximator", dir_name);
  }
  // Policy trainer is required
  policy_trainer = rhoban_fa::OptimizerTrainerFactory().read(v, "policy_trainer", dir_name);
  // Initial policy might be provided (not necessary)
  PolicyFactory().tryRead(v, "policy", dir_name, &policy);
  // Update number of threads for all
  setNbThreads(nb_threads);
}

}  // namespace csa_mdp
