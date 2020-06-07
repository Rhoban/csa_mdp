#include "rhoban_csa_mdp/solvers/policy_mutation_learner.h"

#include "rhoban_bbo/optimizer_factory.h"

#include "rhoban_csa_mdp/core/fa_policy.h"
#include "rhoban_csa_mdp/core/policy_factory.h"

#include "rhoban_fa/constant_approximator.h"
#include "rhoban_fa/fake_split.h"
#include "rhoban_fa/function_approximator_factory.h"
#include "rhoban_fa/linear_approximator.h"
#include "rhoban_fa/orthogonal_split.h"
#include "rhoban_random/tools.h"
#include "rhoban_utils/timing/time_stamp.h"

using namespace rhoban_fa;
using rhoban_utils::TimeStamp;

namespace csa_mdp
{
PolicyMutationLearner::PolicyMutationLearner()
  : training_evaluations(50), split_probability(0.1), split_margin(0.05), evaluations_ratio(-1), age_basis(1.02)
{
}

PolicyMutationLearner::~PolicyMutationLearner()
{
}

int PolicyMutationLearner::getNbEvaluationTrials() const
{
  return nb_evaluation_trials;
}

int PolicyMutationLearner::getTrainingEvaluations() const
{
  return training_evaluations;
}

int PolicyMutationLearner::getOptimizerMaxCall() const
{
  int evaluations_allowed = evaluations_ratio * getNbEvaluationTrials();
  return (int)(evaluations_allowed / getTrainingEvaluations());
}

void PolicyMutationLearner::init(std::default_random_engine* engine)
{
  // If a policy has been specified, try to extract a FATree from policy
  if (policy)
  {
    policy_tree = policy->extractFATree();
    policy = buildPolicy(*policy_tree);
    std::vector<Eigen::MatrixXd> leaf_spaces;
    policy_tree->addSpaces(problem->getStateLimits(), &leaf_spaces);
    for (const Eigen::MatrixXd& leaf_space : leaf_spaces)
    {
      MutationCandidate candidate;
      candidate.space = leaf_space;
      candidate.mutation_score = 1.0;
      candidate.last_training = 0;
      mutation_candidates.push_back(candidate);
    }
  }
  // If no policy were specified, start a new one with a fake split at the root
  else
  {
    std::unique_ptr<Split> split(new FakeSplit());
    // Default action choice is 0
    int action_id = 0;
    const Eigen::MatrixXd& action_limits = problem->getActionLimits(action_id);
    // Default action is the middle of the space
    Eigen::VectorXd action(action_limits.rows() + 1);
    action(0) = action_id;
    action.segment(1, action_limits.rows()) = (action_limits.col(0) + action_limits.col(1)) / 2;
    std::unique_ptr<FunctionApproximator> default_fa(new ConstantApproximator(action));
    std::vector<std::unique_ptr<FunctionApproximator>> fas;
    fas.push_back(std::move(default_fa));
    policy_tree.reset(new FATree(std::move(split), fas));
    policy = buildPolicy(*policy_tree);
    // Adding mutation candidate
    MutationCandidate candidate;
    candidate.space = problem->getStateLimits();
    candidate.mutation_score = 1.0;
    candidate.last_training = 0;
    mutation_candidates.push_back(candidate);
  }
  policy_tree->save("policy_tree.bin");

  double avg_reward = evalAndGetStates(engine);
  updateMutationsScores();
  std::cout << "Initial Reward: " << avg_reward << std::endl;
}

void PolicyMutationLearner::update(std::default_random_engine* engine)
{
  TimeStamp start = TimeStamp::now();
  int mutation_id = getMutationId(engine);
  mutate(mutation_id, engine);
  TimeStamp post_mutation = TimeStamp::now();
  last_score = evalAndGetStates(engine);
  TimeStamp post_evaluation = TimeStamp::now();
  std::cout << "New reward: " << last_score << std::endl;
  policy_tree->save("policy_tree.bin");
  updateMutationsScores();
  TimeStamp post_misc = TimeStamp::now();
  // Writing data
  writeTime("mutation", diffSec(start, post_mutation));
  writeTime("evaluation", diffSec(post_mutation, post_evaluation));
  writeTime("misc", diffSec(post_evaluation, post_misc));
  publishIteration();
}

Eigen::VectorXd PolicyMutationLearner::optimize(rhoban_bbo::Optimizer::RewardFunc rf, const Eigen::MatrixXd& space,
                                                const Eigen::VectorXd& guess, std::default_random_engine* engine,
                                                double evaluation_mult)
{
  // If activated, set the maximal number of calls to the reward function to optimizer
  if (evaluations_ratio > 0)
  {
    optimizer->setMaxCalls(getOptimizerMaxCall() * evaluation_mult);
  }
  optimizer->setLimits(space);
  return optimizer->train(rf, guess, engine);
}

int PolicyMutationLearner::getMutationId(std::default_random_engine* engine)
{
  // Getting max_score
  double total_score = 0;
  double highest_score = std::numeric_limits<double>::lowest();
  for (const MutationCandidate& c : mutation_candidates)
  {
    total_score += c.mutation_score;
    if (c.mutation_score > highest_score)
    {
      highest_score = c.mutation_score;
    }
  }
  // Getting corresponding element
  double c_score = std::uniform_real_distribution<double>(0, total_score)(*engine);

  std::cout << "getMutationId: (max,total) = (" << highest_score << ", " << total_score << ")" << std::endl;

  double acc = 0;
  for (size_t id = 0; id < mutation_candidates.size(); id++)
  {
    acc += mutation_candidates[id].mutation_score;
    if (acc > c_score)
    {
      std::cout << "<- Mutation candidate has score: " << mutation_candidates[id].mutation_score << std::endl;
      return id;
    }
  }
  // Should never happen except with numerical errors
  std::ostringstream oss;
  oss << "PolicyMutationLearner::getMutationId: c_score >= total_score"
      << " (" << c_score << ">=" << total_score << ")";
  throw std::logic_error(oss.str());
}

void PolicyMutationLearner::setNbThreads(int nb_threads)
{
  BlackBoxLearner::setNbThreads(nb_threads);
  // optimizer->setNbThreads(nb_threads);
}

double PolicyMutationLearner::evalAndGetStates(std::default_random_engine* engine)
{
  std::vector<Eigen::VectorXd> global_visited_states;
  double reward = evaluatePolicy(*policy, getNbEvaluationTrials(), engine, &global_visited_states);
  // Clear all visited states between two iterations
  for (MutationCandidate& c : mutation_candidates)
  {
    c.visited_states.clear();
  }
  // TODO: should really be optimized using the tree like structure, but require
  //       to store the mutations using a tree
  for (const Eigen::VectorXd& state : global_visited_states)
  {
    for (MutationCandidate& c : mutation_candidates)
    {
      bool valid = true;
      for (int dim = 0; dim < problem->stateDims(); dim++)
      {
        if (state(dim) < c.space(dim, 0) || state(dim) >= c.space(dim, 1))
        {
          valid = false;
          break;
        }
      }
      if (valid)
      {
        c.visited_states.push_back(state);
      }
    }
  }
  return reward;
}

void PolicyMutationLearner::updateMutationsScores()
{
  for (MutationCandidate& c : mutation_candidates)
  {
    // Computing age score
    double age = iterations - c.last_training;
    double relative_age = age / mutation_candidates.size();
    double age_score = pow(age_basis, relative_age);
    // Computing density score
    int nb_states = c.visited_states.size();
    double density_score = 1 + nb_states;  // avoid 0 score
    // Updating mutation score
    c.mutation_score = age_score * density_score;
  }
}

void PolicyMutationLearner::mutate(int mutation_id, std::default_random_engine* engine)
{
  if (isMutationAllowed(mutation_candidates[mutation_id]))
  {
    mutateLeaf(mutation_id, engine);
  }
  else
  {
    std::cout << "-> Skipping mutation (forbidden)" << std::endl;
    mutation_candidates[mutation_id].last_training = iterations;
  }
}

void PolicyMutationLearner::mutateLeaf(int mutation_id, std::default_random_engine* engine)
{
  double rand_val = std::uniform_real_distribution<double>(0.0, 1.0)(*engine);
  if (rand_val < split_probability)
  {
    splitMutation(mutation_id, engine);
  }
  else
  {
    refineMutation(mutation_id, engine);
  }
}

void PolicyMutationLearner::refineMutation(int mutation_id, std::default_random_engine* engine)
{
  // Get reference to the appropriate mutation
  MutationCandidate* mutation = &(mutation_candidates[mutation_id]);
  // Get space, and space center
  Eigen::MatrixXd space = mutation->space;
  int input_dim = problem->stateDims();
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  // Getting current action_id
  Eigen::VectorXd current_action = policy_tree->predict(space_center);
  int action_id = (int)current_action(0);
  // Generate a random number in [0, nb_actions-1]
  std::uniform_int_distribution<int> action_id_distrib(0, problem->getNbActions() - 1);
  action_id = action_id_distrib(*engine);
  // Test action of type
  std::cout << "Training for action to " << action_id << std::endl;
  int output_dim = problem->actionDims(action_id);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  // Initial states
  std::vector<Eigen::VectorXd> initial_states = getInitialStates(*mutation, engine);
  std::cout << "-> Nb Initial states: " << initial_states.size() << std::endl;
  // Add the parameters to provide an action_id
  auto parameters_to_full = [input_dim, output_dim, action_id](const Eigen::VectorXd& raw_params) {
    Eigen::VectorXd full_params((input_dim + 1) * (output_dim + 1));
    for (int col = 0; col < input_dim + 1; col++)
    {
      int start = col * (output_dim + 1);
      full_params(start) = col == 0 ? action_id : 0;
      full_params.segment(start + 1, output_dim) = raw_params.segment(col * output_dim, output_dim);
    }
    return full_params;
  };
  // Training function
  // TODO: use other models than PWL
  rhoban_bbo::Optimizer::RewardFunc reward_func = [this, &space, input_dim, output_dim, &space_center, &initial_states,
                                                   action_id, &parameters_to_full](const Eigen::VectorXd& parameters,
                                                                                   std::default_random_engine* engine) {
    // Add the parameters to choose action id
    Eigen::VectorXd full_params = parameters_to_full(parameters);
    // Build the tree
    std::unique_ptr<FunctionApproximator> new_approximator(
        new LinearApproximator(input_dim, output_dim + 1, full_params, space_center));
    std::unique_ptr<FATree> new_tree;
    new_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(new_approximator));
    // TODO: avoid this second copy of the tree which is not necessary
    std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
    // Type of evaluation depends on the fact that initial_states have been created
    return evaluation(*policy, initial_states, engine);
  };
  // Getting parameters_space
  Eigen::MatrixXd parameters_space = getParametersSpaces(action_id);
  Eigen::VectorXd parameters_guess = Eigen::VectorXd::Zero(parameters_space.rows());
  for (int dim = 0; dim < parameters_guess.rows(); dim++)
  {
    double original = parameters_guess(dim);
    double min = parameters_space(dim, 0);
    double max = parameters_space(dim, 1);
    parameters_guess(dim) = std::min(max, std::max(min, original));
  }
  // Debug function
  // auto parameters_to_matrix =
  //  [input_dim, output_dim](const Eigen::VectorXd & parameters)
  //  {
  //    Eigen::MatrixXd result(output_dim + 1, input_dim+1);
  //    for (int col = 0; col < input_dim+1; col++) {
  //      result.col(col) = parameters.segment(col * (output_dim+1), output_dim + 1);
  //    }
  //    return result;
  //  };
  // std::cout << "Parameters_space:" << std::endl
  //          << parameters_space << std::endl;
  // std::cout << "Guess:" << std::endl
  //          << parameters_to_matrix(parameters_to_full(parameters_guess)) << std::endl;

  // Creating refined approximator
  Eigen::VectorXd refined_parameters = optimize(reward_func, parameters_space, parameters_guess, engine);
  refined_parameters = parameters_to_full(refined_parameters);
  // std::cout << "\tRefined parameters:" << std::endl
  //          << parameters_to_matrix(refined_parameters) << std::endl;

  std::unique_ptr<FunctionApproximator> refined_approximator(
      new LinearApproximator(input_dim, output_dim + 1, refined_parameters, space_center));
  // Creating refined policy
  std::unique_ptr<FATree> refined_tree;
  refined_tree = policy_tree->copyAndReplaceLeaf(space_center, std::move(refined_approximator));
  // Submit the new tree
  submitTree(std::move(refined_tree), initial_states, engine);
  // Update mutation properties:
  mutation->last_training = iterations;
}

void PolicyMutationLearner::splitMutation(int mutation_id, std::default_random_engine* engine)
{
  const MutationCandidate& mutation = mutation_candidates[mutation_id];
  Eigen::MatrixXd space = mutation.space;
  Eigen::VectorXd space_center = (space.col(0) + space.col(1)) / 2;
  const FunctionApproximator& leaf_fa = policy_tree->getLeafApproximator(space_center);
  std::vector<Eigen::VectorXd> initial_states = getInitialStates(mutation, engine);
  // Debug message
  std::cout << "-> Applying a split mutation on space" << std::endl << space.transpose() << std::endl;
  std::cout << "-> Nb initial states: " << initial_states.size() << std::endl;
  // Testing all dimensions as split and keeping the best one
  double best_score = std::numeric_limits<double>::lowest();
  std::unique_ptr<FATree> best_tree;
  for (int dim = 0; dim < problem->stateDims(); dim++)
  {
    double score;
    std::unique_ptr<FATree> current_tree = trySplit(mutation_id, dim, initial_states, engine, &score);
    if (score > best_score)
    {
      best_score = score;
      best_tree = std::move(current_tree);
    }
  }
  // Avoiding risk of segfault if something wrong happened before
  if (!best_tree)
  {
    throw std::logic_error("PolicyMutationLearner::splitMutation: Failed to find a best tree");
  }
  // Getting ref to the chosen Split and backing it up in case it is needed after
  const Split& split = best_tree->getPreLeafApproximator(space_center).getSplit();
  std::unique_ptr<Split> split_copy = split.clone();
  std::vector<Eigen::MatrixXd> split_spaces = split.splitSpace(space);
  // Evaluating policy with respect to previous solution
  bool replaced_policy = submitTree(std::move(best_tree), initial_states, engine);
  // If the new approximation does not replace current one, reuse current model
  if (!replaced_policy)
  {
    // Create a new FATree with cloned approximators
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    for (size_t elem = 0; elem < split_spaces.size(); elem++)
    {
      approximators.push_back(leaf_fa.clone());
    }
    std::unique_ptr<FATree> new_leaf_fa(new FATree(std::move(split_copy), approximators));
    // Replace it in policy tree and update policy
    policy_tree->replaceApproximator(space_center, std::move(new_leaf_fa));
    policy = buildPolicy(*policy_tree);
  }
  // Updating mutation_candidates
  std::vector<std::unique_ptr<FunctionApproximator>> approximators;
  for (size_t i = 0; i < split_spaces.size(); i++)
  {
    MutationCandidate new_mutation;
    new_mutation.space = split_spaces[i];
    new_mutation.mutation_score = mutation.mutation_score;
    // We consider as training only refinement which are likely to lead to
    // real improvements
    new_mutation.last_training = mutation.last_training;
    if (i == 0)
    {
      mutation_candidates[mutation_id] = new_mutation;
    }
    else
    {
      mutation_candidates.push_back(new_mutation);
    }
  }
}

std::unique_ptr<rhoban_fa::FATree> PolicyMutationLearner::trySplit(int mutation_id, int split_dim,
                                                                   const std::vector<Eigen::VectorXd>& initial_states,
                                                                   std::default_random_engine* engine, double* score)
{
  // Debug message
  std::cout << "\tTesting split along " << split_dim << std::endl;
  // Getting mutation properties
  MutationCandidate mutation = mutation_candidates[mutation_id];
  Eigen::MatrixXd leaf_space = mutation.space;
  Eigen::MatrixXd leaf_center = (leaf_space.col(0) + leaf_space.col(1)) / 2;
  // Getting current action_id
  Eigen::VectorXd current_action = policy_tree->predict(leaf_center);
  int action_id = (int)current_action(0);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  int action_dims = problem->actionDims(action_id);
  // Function to train has the following parameters:
  // - Position of the split
  // - Action in each part of the split
  auto parameters_to_approximator = [action_dims, split_dim, action_id](const Eigen::VectorXd& parameters) {
    // Importing values from parameters
    double split_val = parameters[0];
    // Adding action_id
    Eigen::VectorXd params0 = Eigen::VectorXd::Zero(action_dims + 1);
    params0(0) = action_id;
    params0.segment(1, action_dims) = parameters.segment(1, action_dims);
    Eigen::VectorXd params1 = Eigen::VectorXd::Zero(action_dims + 1);
    params1(0) = action_id;
    params1.segment(1, action_dims) = parameters.segment(1 + action_dims, action_dims);
    // Building FunctionApproximator
    std::unique_ptr<FunctionApproximator> action0(new ConstantApproximator(params0));
    std::unique_ptr<FunctionApproximator> action1(new ConstantApproximator(params1));
    // Define Function approximator which will replace
    std::unique_ptr<Split> split(new OrthogonalSplit(split_dim, split_val));
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    approximators.push_back(std::move(action0));
    approximators.push_back(std::move(action1));
    return std::unique_ptr<FATree>(new FATree(std::move(split), approximators));
  };
  // Defining evaluation function
  rhoban_bbo::Optimizer::RewardFunc reward_func =
      [this, split_dim, action_dims, leaf_space, leaf_center, parameters_to_approximator,
       &initial_states](const Eigen::VectorXd& parameters, std::default_random_engine* engine) {
        std::unique_ptr<FATree> new_approximator;
        new_approximator = parameters_to_approximator(parameters);
        // Replace approximator
        std::unique_ptr<FATree> new_tree;
        new_tree = policy_tree->copyAndReplaceLeaf(leaf_center, std::move(new_approximator));
        // build policy and evaluate
        // TODO: avoid this second copy of the tree which is not necessary
        std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
        return evaluation(*policy, initial_states, engine);
      };
  // Computing boundaries for split
  double dim_min = leaf_space(split_dim, 0);
  double dim_max = leaf_space(split_dim, 1);
  double delta = (dim_max - dim_min) * split_margin;
  double split_min = dim_min + delta;
  double split_max = dim_max - delta;
  std::cout << "\t->Split value range:  [" << split_min << ", " << split_max << "]" << std::endl;
  // Define parameters_space
  Eigen::MatrixXd parameters_space(1 + 2 * action_dims, 2);
  parameters_space(0, 0) = split_min;
  parameters_space(0, 1) = split_max;
  parameters_space.block(1, 0, action_dims, 2) = action_limits;
  parameters_space.block(1 + action_dims, 0, action_dims, 2) = action_limits;
  // Computing initial parameters using current approximator
  double split_default_val = (split_min + split_max) / 2;
  std::unique_ptr<Split> initial_split(new OrthogonalSplit(split_dim, split_default_val));
  std::vector<Eigen::MatrixXd> split_spaces = initial_split->splitSpace(leaf_space);
  Eigen::VectorXd initial_parameters(1 + 2 * action_dims);
  initial_parameters(0) = split_default_val;
  for (size_t leaf_id = 0; leaf_id < split_spaces.size(); leaf_id++)
  {
    int start = 1 + action_dims * leaf_id;
    const Eigen::MatrixXd& curr_space = split_spaces[leaf_id];
    Eigen::VectorXd center = (curr_space.col(0) + curr_space.col(1)) / 2;
    Eigen::VectorXd action = policy_tree->predict(center);
    initial_parameters.segment(start, action_dims) = action.segment(1, action_dims);
  }
  // Optimize
  double evaluation_ratio = 1.0 / problem->getStateLimits().rows();
  Eigen::VectorXd optimized_parameters =
      optimize(reward_func, parameters_space, initial_parameters, engine, evaluation_ratio);
  // Make a more approximate evaluation
  std::unique_ptr<FunctionApproximator> optimized_approximator;
  optimized_approximator = parameters_to_approximator(optimized_parameters);
  std::unique_ptr<FATree> splitted_tree;
  splitted_tree = policy_tree->copyAndReplaceLeaf(leaf_center, std::move(optimized_approximator));
  std::unique_ptr<Policy> optimized_policy = buildPolicy(*splitted_tree);
  // Evaluate with a larger set
  *score = localEvaluation(*optimized_policy, leaf_space, evaluation_ratio * getNbEvaluationTrials(), engine);
  // Debug
  std::cout << "\t->Optimized params: " << optimized_parameters.transpose() << std::endl;
  std::cout << "\t->Avg reward: " << *score << std::endl;
  // return optimized approximator
  return std::move(splitted_tree);
}

Eigen::MatrixXd PolicyMutationLearner::getParametersSpaces(int action_id) const
{
  // Which space is used for parameters
  Eigen::MatrixXd parameters_space;
  Eigen::MatrixXd space_for_parameters = problem->getStateLimits();
  parameters_space =
      LinearApproximator::getParametersSpace(space_for_parameters, problem->getActionLimits(action_id), false);
  return parameters_space;
}

std::string PolicyMutationLearner::getClassName() const
{
  return "PolicyMutationLearner";
}

Json::Value PolicyMutationLearner::toJson() const
{
  throw std::logic_error("PolicyMutationLearner::toJson: not implemented");
}

void PolicyMutationLearner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  // Calling parent implementation
  BlackBoxLearner::fromJson(v, dir_name);
  // Reading class variables
  rhoban_utils::tryRead(v, "training_evaluations", &training_evaluations);
  rhoban_utils::tryRead(v, "split_probability", &split_probability);
  rhoban_utils::tryRead(v, "split_margin", &split_margin);
  rhoban_utils::tryRead(v, "evaluations_ratio", &evaluations_ratio);
  rhoban_utils::tryRead(v, "age_basis", &age_basis);
  // Optimizer is mandatory
  optimizer = rhoban_bbo::OptimizerFactory().read(v, "optimizer", dir_name);
  // Read Policy if provided (optional)
  PolicyFactory().tryRead(v, "policy", dir_name, &policy);
  // Performing some checks
  if (split_margin < 0 || split_margin >= 0.5)
  {
    throw std::logic_error("PolicyMutationLearner::fromJson: invalid value for split_margin");
  }
  // TODO: add checks on probability
  // Synchronize number of threads
  setNbThreads(nb_threads);
}

std::unique_ptr<Policy> PolicyMutationLearner::buildPolicy(const FATree& tree)
{
  std::unique_ptr<FATree> tree_copy(static_cast<FATree*>(tree.clone().release()));
  std::unique_ptr<Policy> result(new FAPolicy(std::move(tree_copy)));
  result->setActionLimits(problem->getActionsLimits());
  return std::move(result);
}

std::vector<Eigen::VectorXd> PolicyMutationLearner::getInitialStates(const MutationCandidate& mc,
                                                                     std::default_random_engine* engine)
{
  size_t nb_evaluations_allowed = getTrainingEvaluations();
  // If we are lacking samples return them all
  if (nb_evaluations_allowed >= mc.visited_states.size())
  {
    return mc.visited_states;
  }
  // Filter most important samples
  std::vector<Eigen::VectorXd> initial_states;
  std::vector<size_t> indices =
      rhoban_random::getKDistinctFromN(nb_evaluations_allowed, mc.visited_states.size(), engine);
  for (size_t idx : indices)
  {
    initial_states.push_back(mc.visited_states[idx]);
  }
  return initial_states;
}

bool PolicyMutationLearner::isMutationAllowed(const MutationCandidate& mc) const
{
  return mc.visited_states.size() > 10;
}

bool PolicyMutationLearner::submitTree(std::unique_ptr<rhoban_fa::FATree> new_tree,
                                       const std::vector<Eigen::VectorXd>& initial_states,
                                       std::default_random_engine* engine)
{
  std::unique_ptr<Policy> new_policy = buildPolicy(*new_tree);
  // Evaluate old and new policy with updated parameters (on local space)
  double old_local_reward = evaluation(*policy, initial_states, engine);
  double new_local_reward = evaluation(*new_policy, initial_states, engine);
  double old_global_reward = evaluatePolicy(*policy, getNbEvaluationTrials(), engine);
  double new_global_reward = evaluatePolicy(*new_policy, getNbEvaluationTrials(), engine);
  std::cout << "\told local reward: " << old_local_reward << std::endl;
  std::cout << "\tnew local reward: " << new_local_reward << std::endl;
  std::cout << "\told global reward: " << old_global_reward << std::endl;
  std::cout << "\tnew global reward: " << new_global_reward << std::endl;
  // Replace current if improvement has been seen both locally and globally
  if (new_local_reward > old_local_reward && new_global_reward > old_global_reward)
  {
    policy_tree = std::move(new_tree);
    policy = std::move(new_policy);
    return true;
  }
  return false;
}

}  // namespace csa_mdp
