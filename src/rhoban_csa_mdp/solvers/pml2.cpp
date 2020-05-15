#include "rhoban_csa_mdp/solvers/pml2.h"

#include "rhoban_bbo/optimizer_factory.h"

#include "rhoban_csa_mdp/core/fa_policy.h"
#include "rhoban_csa_mdp/core/policy_factory.h"

#include "rhoban_fa/constant_approximator.h"
#include "rhoban_fa/fake_split.h"
#include "rhoban_fa/function_approximator_factory.h"
#include "rhoban_fa/linear_approximator.h"
#include "rhoban_fa/linear_split.h"
#include "rhoban_fa/orthogonal_split.h"
#include "rhoban_random/tools.h"
#include "rhoban_utils/timing/time_stamp.h"

using namespace rhoban_fa;
using rhoban_utils::TimeStamp;

namespace csa_mdp
{
// TODO: those functions should be placed elsewhere, maybe static

/// Build a linear function using the provided information
std::unique_ptr<FunctionApproximator> buildFAFromLinearParams(const Eigen::VectorXd& raw_params,
                                                              const Eigen::VectorXd& space_center, int input_dim,
                                                              int output_dim, int action_id)
{
  Eigen::VectorXd full_params((input_dim + 1) * (output_dim + 1));
  for (int col = 0; col < input_dim + 1; col++)
  {
    int start = col * (output_dim + 1);
    full_params(start) = col == 0 ? action_id : 0;
    full_params.segment(start + 1, output_dim) = raw_params.segment(col * output_dim, output_dim);
  }
  return std::unique_ptr<FunctionApproximator>(
      new LinearApproximator(input_dim, output_dim + 1, full_params, space_center));
}

Eigen::VectorXd computeStatesCenter(const std::vector<Eigen::VectorXd>& states)
{
  if (states.size() == 0)
  {
    throw std::logic_error("PML2::computeStatesCenter: empty vector");
  }
  Eigen::VectorXd sum = states[0];
  for (size_t i = 1; i < states.size(); i++)
  {
    sum = sum + states[i];
  }
  return sum / states.size();
}

Eigen::VectorXd computeStatesDeviation(const std::vector<Eigen::VectorXd>& states)
{
  Eigen::VectorXd center = computeStatesCenter(states);
  Eigen::VectorXd squared_errors = Eigen::VectorXd::Zero(center.rows());
  for (size_t i = 0; i < states.size(); i++)
  {
    squared_errors += (states[i] - center).cwiseAbs2();  // Add squared error for current state
  }
  return (squared_errors / states.size()).cwiseSqrt();
}

PML2::PML2()
  : training_evaluations(50), split_margin(0.05), evaluations_ratio(-1), age_basis(1.02), use_linear_splits(false)
{
}

PML2::~PML2()
{
}

int PML2::getNbEvaluationTrials() const
{
  return nb_evaluation_trials;
}

int PML2::getTrainingEvaluations() const
{
  return training_evaluations;
}

int PML2::getOptimizerMaxCall() const
{
  int evaluations_allowed = evaluations_ratio * getNbEvaluationTrials();
  return (int)(evaluations_allowed / getTrainingEvaluations());
}

void PML2::init(std::default_random_engine* engine)
{
  // If a policy has been specified, try to extract a FATree from policy
  if (policy)
  {
    policy_tree = policy->extractFATree();
    policy_tree->updateNodesCount();
    policy = buildPolicy(*policy_tree);
    std::vector<int> leaves_id = policy_tree->getLeavesId();
    for (int leaf_id : leaves_id)
    {
      mutation_candidates[leaf_id].mutation_score = 1.0;
      mutation_candidates[leaf_id].last_training = 0;
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
    // Getting leaf_id formally
    policy_tree->updateNodesCount();
    Eigen::MatrixXd space_limits = problem->getStateLimits();
    Eigen::VectorXd space_center = (space_limits.col(0) + space_limits.col(1)) / 2;
    int leaf_id = policy_tree->getLeafId(space_center);
    // Adding mutation candidate
    MutationCandidate candidate;
    candidate.mutation_score = 1.0;
    candidate.last_training = 0;
    mutation_candidates[leaf_id] = (candidate);
  }
  policy_tree->save("policy_tree.bin");

  double avg_reward = evalAndGetStates(engine);
  updateMutationsScores();
  std::cout << "Initial Reward: " << avg_reward << std::endl;
}

void PML2::update(std::default_random_engine* engine)
{
  TimeStamp start = TimeStamp::now();
  int mutation_id = getMutationId(engine);
  mutate(mutation_id, engine);
  TimeStamp post_mutation = TimeStamp::now();
  double new_reward = evalAndGetStates(engine);
  TimeStamp post_evaluation = TimeStamp::now();
  std::cout << "New reward: " << new_reward << std::endl;
  policy_tree->save("policy_tree.bin");
  updateMutationsScores();
  TimeStamp post_misc = TimeStamp::now();
  // Writing data
  writeTime("mutation", diffSec(start, post_mutation));
  writeTime("evaluation", diffSec(post_mutation, post_evaluation));
  writeTime("misc", diffSec(post_evaluation, post_misc));
  writeScore(new_reward);
}

Eigen::VectorXd PML2::optimize(rhoban_bbo::Optimizer::RewardFunc rf, const Eigen::MatrixXd& space,
                               const Eigen::VectorXd& guess, std::default_random_engine* engine, double evaluation_mult)
{
  // If activated, set the maximal number of calls to the reward function to optimizer
  if (evaluations_ratio > 0)
  {
    optimizer->setMaxCalls(getOptimizerMaxCall() * evaluation_mult);
  }
  optimizer->setLimits(space);
  return optimizer->train(rf, guess, engine);
}

int PML2::getMutationId(std::default_random_engine* engine)
{
  // Getting total_score
  double total_score = 0;
  double highest_score = std::numeric_limits<double>::lowest();
  for (const auto& e : mutation_candidates)
  {
    total_score += e.second.mutation_score;
    if (e.second.mutation_score > highest_score)
    {
      highest_score = e.second.mutation_score;
    }
  }
  // Getting corresponding element
  double c_score = std::uniform_real_distribution<double>(0, total_score)(*engine);

  std::cout << "getMutationId: (max,total) = (" << highest_score << ", " << total_score << ")" << std::endl;

  double acc = 0;
  for (const auto& e : mutation_candidates)
  {
    acc += e.second.mutation_score;
    if (acc > c_score)
    {
      std::cout << "<- Mutation candidate: '" << e.first << "' score: " << e.second.mutation_score << std::endl;
      return e.first;
    }
  }
  // Should never happen except with numerical errors
  std::ostringstream oss;
  oss << "PML2::getMutationId: c_score >= total_score"
      << " (" << c_score << ">=" << total_score << ")";
  throw std::logic_error(oss.str());
}

void PML2::setNbThreads(int nb_threads)
{
  BlackBoxLearner::setNbThreads(nb_threads);
}

double PML2::evalAndGetStates(std::default_random_engine* engine)
{
  std::vector<Eigen::VectorXd> global_visited_states;
  std::cout << "Evaluating with " << getNbEvaluationTrials() << " trials" << std::endl;
  double reward = evaluatePolicy(*policy, getNbEvaluationTrials(), engine, &global_visited_states);
  // Clear all visited states between two iterations
  for (auto& e : mutation_candidates)
  {
    e.second.visited_states.clear();
  }
  for (const Eigen::VectorXd& state : global_visited_states)
  {
    int node_idx = policy_tree->getLeafId(state);
    // node_idx is expected to be present in mutation_candidates
    mutation_candidates.at(node_idx).visited_states.push_back(state);
  }
  return reward;
}

void PML2::updateMutationsScores()
{
  for (auto& e : mutation_candidates)
  {
    MutationCandidate& c = e.second;
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

void PML2::mutate(int mutation_id, std::default_random_engine* engine)
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

void PML2::mutateLeaf(int mutation_id, std::default_random_engine* engine)
{
  for (int action_id = 0; action_id < problem->getNbActions(); action_id++)
  {
    tryRefine(mutation_id, action_id, engine);
  }
  if (use_linear_splits)
  {
    applyBestLinearSplit(mutation_id, engine);
  }
  else
  {
    applyBestSplit(mutation_id, engine);
  }
}

void PML2::tryRefine(int mutation_id, int action_id, std::default_random_engine* engine)
{
  // Get reference to the appropriate mutation
  MutationCandidate* mutation = &(mutation_candidates[mutation_id]);
  // Initial states
  std::vector<Eigen::VectorXd> initial_states = getInitialStates(*mutation, true, engine);
  std::cout << "-> Nb Initial states: " << initial_states.size() << std::endl;
  // Get space, and space center
  int input_dim = problem->stateDims();
  Eigen::VectorXd samples_center = computeStatesCenter(initial_states);
  // Test action of type
  std::cout << "Training for action to " << action_id << std::endl;
  int output_dim = problem->actionDims(action_id);
  Eigen::MatrixXd action_limits = problem->getActionLimits(action_id);
  // Training function
  rhoban_bbo::Optimizer::RewardFunc reward_func = [this, input_dim, output_dim, &samples_center, &initial_states,
                                                   action_id](const Eigen::VectorXd& parameters,
                                                              std::default_random_engine* engine) {
    // Build the tree
    std::unique_ptr<FunctionApproximator> new_approximator;
    new_approximator = buildFAFromLinearParams(parameters, samples_center, input_dim, output_dim, action_id);
    std::unique_ptr<FATree> new_tree;
    new_tree = policy_tree->copyAndReplaceLeaf(initial_states[0], std::move(new_approximator));
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

  // Optimization phase
  Eigen::VectorXd refined_parameters = optimize(reward_func, parameters_space, parameters_guess, engine);

  // Creating refined approximator
  std::unique_ptr<FunctionApproximator> refined_approximator;
  refined_approximator = buildFAFromLinearParams(refined_parameters, samples_center, input_dim, output_dim, action_id);
  // Creating refined policy
  std::unique_ptr<FATree> refined_tree;
  refined_tree = policy_tree->copyAndReplaceLeaf(initial_states[0], std::move(refined_approximator));
  // Submit the new tree
  initial_states = getInitialStates(*mutation, false, engine);
  submitTree(std::move(refined_tree), initial_states, engine);
  // Update mutation properties:
  mutation->last_training = iterations;
}

void PML2::applyBestSplit(int mutation_id, std::default_random_engine* engine)
{
  const MutationCandidate& mutation = mutation_candidates[mutation_id];
  std::vector<Eigen::VectorXd> training_states = getInitialStates(mutation, true, engine);
  std::vector<Eigen::VectorXd> validation_states = getInitialStates(mutation, false, engine);
  Eigen::VectorXd samples_center = computeStatesCenter(training_states);
  const FunctionApproximator& leaf_fa = policy_tree->getLeafApproximator(training_states[0]);
  // Debug message
  std::cout << "-> Applying a split mutation " << std::endl;
  std::cout << "-> Nb training states: " << training_states.size() << std::endl;
  // Testing all dimensions as split and keeping the best one
  double best_score = std::numeric_limits<double>::lowest();
  std::unique_ptr<FATree> best_tree;
  for (int dim = 0; dim < problem->stateDims(); dim++)
  {
    double score;
    std::unique_ptr<FATree> current_tree = trySplit(dim, training_states, validation_states, engine, &score);
    if (score > best_score)
    {
      best_score = score;
      best_tree = std::move(current_tree);
    }
  }
  // Avoiding risk of segfault if something wrong happened before
  if (!best_tree)
  {
    throw std::logic_error("PML2::splitMutation: Failed to find a best tree");
  }
  // Getting ref to the chosen Split and backing it up in case it is needed after
  const Split& split = best_tree->getPreLeafApproximator(training_states[0]).getSplit();
  std::unique_ptr<Split> split_copy = split.clone();
  // Evaluating policy with respect to previous solution
  int nb_new_nodes = split.getNbElements();
  bool replaced_policy = submitTree(std::move(best_tree), training_states, engine);
  // If the new approximation does not replace current one, reuse current FA for children
  if (!replaced_policy)
  {
    // Create a new FATree with cloned approximators
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    for (int elem = 0; elem < split_copy->getNbElements(); elem++)
    {
      approximators.push_back(leaf_fa.clone());
    }
    std::unique_ptr<FATree> new_leaf_fa(new FATree(std::move(split_copy), approximators));
    // Replace it in policy tree and update policy
    policy_tree->replaceApproximator(training_states[0], std::move(new_leaf_fa));
    policy = buildPolicy(*policy_tree);
  }
  postSplitUpdate(mutation_id, nb_new_nodes);
}

void PML2::applyBestLinearSplit(int mutation_id, std::default_random_engine* engine)
{
  const MutationCandidate& mutation = mutation_candidates[mutation_id];
  std::vector<Eigen::VectorXd> training_states = getInitialStates(mutation, true, engine);
  std::vector<Eigen::VectorXd> validation_states = getInitialStates(mutation, false, engine);
  Eigen::VectorXd samples_center = computeStatesCenter(training_states);
  const FunctionApproximator& leaf_fa = policy_tree->getLeafApproximator(training_states[0]);
  // Debug message
  std::cout << "-> Applying a linear split mutation " << std::endl;
  std::cout << "-> Nb training states: " << training_states.size() << std::endl;
  // Testing all actions as split and keeping the best one
  double best_score = std::numeric_limits<double>::lowest();
  std::unique_ptr<FATree> best_tree;
  for (int action_id = 0; action_id < problem->getNbActions(); action_id++)
  {
    double score;
    std::unique_ptr<FATree> current_tree =
        tryLinearSplit(action_id, training_states, validation_states, engine, &score);
    if (score > best_score)
    {
      best_score = score;
      best_tree = std::move(current_tree);
    }
  }
  // Avoiding risk of segfault if something wrong happened before
  if (!best_tree)
  {
    throw std::logic_error("PML2::applyBestLinearSplit: Failed to find a best tree");
  }
  // TODO: this part is duplicated from applyBestOrthogonalSplit code shall be shared
  // Getting ref to the chosen Split and backing it up in case it is needed after
  const Split& split = best_tree->getPreLeafApproximator(training_states[0]).getSplit();
  std::unique_ptr<Split> split_copy = split.clone();
  // Evaluating policy with respect to previous solution
  int nb_new_nodes = split.getNbElements();
  bool replaced_policy = submitTree(std::move(best_tree), training_states, engine);
  // If the new approximation does not replace current one, reuse current FA for children
  if (!replaced_policy)
  {
    // Create a new FATree with cloned approximators
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    for (int elem = 0; elem < split_copy->getNbElements(); elem++)
    {
      approximators.push_back(leaf_fa.clone());
    }
    std::unique_ptr<FATree> new_leaf_fa(new FATree(std::move(split_copy), approximators));
    // Replace it in policy tree and update policy
    policy_tree->replaceApproximator(training_states[0], std::move(new_leaf_fa));
    policy = buildPolicy(*policy_tree);
  }
  postSplitUpdate(mutation_id, nb_new_nodes);
}

std::unique_ptr<rhoban_fa::FATree> PML2::trySplit(int split_dim, const std::vector<Eigen::VectorXd>& training_states,
                                                  const std::vector<Eigen::VectorXd>& validation_states,
                                                  std::default_random_engine* engine, double* score)
{
  // Debug message
  std::cout << "\tTesting split along " << split_dim << std::endl;
  // Getting center of samples
  Eigen::MatrixXd samples_center = computeStatesCenter(training_states);
  // Getting current action_id
  int action_id = (int)policy_tree->predict(training_states[0])(0);
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
  // Defining evaluation and reward function
  rhoban_bbo::Optimizer::RewardFunc reward_func = [this, &training_states,
                                                   parameters_to_approximator](const Eigen::VectorXd& parameters,
                                                                               std::default_random_engine* engine) {
    return evalMutation(parameters, training_states, parameters_to_approximator, engine);
  };
  // Computing boundaries for split
  double dim_min = std::numeric_limits<double>::max();
  double dim_max = std::numeric_limits<double>::lowest();
  for (const Eigen::VectorXd& state : training_states)
  {
    double v = state(split_dim);
    if (v < dim_min)
      dim_min = v;
    if (v > dim_max)
      dim_max = v;
  }
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
  // Computing training parameters using current approximator
  double split_default_val = (split_min + split_max) / 2;
  std::unique_ptr<Split> training_split(new OrthogonalSplit(split_dim, split_default_val));
  Eigen::VectorXd training_parameters(1 + 2 * action_dims);
  Eigen::VectorXd default_action = policy_tree->predict(samples_center);
  training_parameters(0) = split_default_val;
  for (int leaf_id = 0; leaf_id < training_split->getNbElements(); leaf_id++)
  {
    int start = 1 + action_dims * leaf_id;
    training_parameters.segment(start, action_dims) = default_action.segment(1, action_dims);
  }
  // Optimize
  Eigen::VectorXd optimized_parameters = optimize(reward_func, parameters_space, training_parameters, engine);
  // Evaluating split with a larger set
  *score = evalMutation(optimized_parameters, validation_states, parameters_to_approximator, engine);
  // Debug
  std::cout << "\t->Optimized params: " << optimized_parameters.transpose() << std::endl;
  std::cout << "\t->Avg reward: " << *score << std::endl;
  // Build corresponding tree
  std::unique_ptr<FunctionApproximator> optimized_approximator;
  optimized_approximator = parameters_to_approximator(optimized_parameters);
  std::unique_ptr<FATree> splitted_tree;
  splitted_tree = policy_tree->copyAndReplaceLeaf(training_states[0], std::move(optimized_approximator));
  // return optimized approximator
  return std::move(splitted_tree);
}

/// Optimize a split wit the chosen action id
///
/// Function to optimize has the following parameters
/// 0                  : hyperplane offset factor in [-1 1] (with respect to samples center)
/// 1 to D             : hyperplane coeffs (with D input dimensionality) [-1 1] for each coeff
/// D+1 to (D+1)(A+1)  : action coefficients (cf getParametersSpace) ((D+1)*A coeffs)
/// with: D input dimension and A action space dimension
std::unique_ptr<rhoban_fa::FATree> PML2::tryLinearSplit(int action_id,
                                                        const std::vector<Eigen::VectorXd>& training_states,
                                                        const std::vector<Eigen::VectorXd>& validation_states,
                                                        std::default_random_engine* engine, double* score)
{
  // Debug message
  std::cout << "\tTesting linear split with action: " << action_id << std::endl;
  // Getting dimensions
  int D = problem->stateDims();
  int A = problem->actionDims(action_id);
  // Parameters space
  int param_dims = (A + 1) * (D + 1);
  Eigen::MatrixXd parameters_space(param_dims, 2);
  for (int i = 0; i <= D; i++)
  {
    parameters_space(i, 0) = -1;
    parameters_space(i, 1) = 1;
  }
  parameters_space.block(D + 1, 0, (D + 1) * A, 2) = getParametersSpaces(action_id);
  // Initial guess
  // - Impossible to use a normal vector with only 0 -> arbitrary choice for initial vector
  // - Middle of the spaces for linear coefficients (actions are not necessarily centered in 0)
  Eigen::VectorXd initial_parameters = Eigen::VectorXd::Zero(param_dims);
  initial_parameters(1) = 1;
  initial_parameters.segment(D + 1, (D + 1) * A) =
      (parameters_space.block(D + 1, 0, (D + 1) * A, 1) + parameters_space.block(D + 1, 1, (D + 1) * A, 1)) / 2;
  // Getting mutation properties
  Eigen::VectorXd samples_center = computeStatesCenter(training_states);
  Eigen::VectorXd samples_dev = computeStatesDeviation(training_states);
  const FunctionApproximator& leaf_fa = policy_tree->getLeafApproximator(training_states[0]);
  auto parameters_to_approximator = [action_id, A, D, &samples_center, &samples_dev,
                                     &leaf_fa](const Eigen::VectorXd& parameters) {
    // Importing values from parameters
    double offset_param = parameters(0);
    const Eigen::VectorXd& hyperplane_coeffs_param = parameters.segment(1, D);
    const Eigen::VectorXd& linear_approximator_params = parameters.segment(D + 1, A * (D + 1));
    // Extracting hyperplane
    // - Offset of plan is based on deviation of samples, hyperplane coeffs and params
    //   (by using absolute coeffs, we ensure that there is no possible compensation)
    Eigen::VectorXd hyperplane_coeffs = hyperplane_coeffs_param.normalized();
    double offset = -hyperplane_coeffs.dot(samples_center);
    offset += samples_dev.dot(hyperplane_coeffs.cwiseAbs()) * offset_param;
    // - Problematic because of the signs
    // Extracting new function approximator
    std::unique_ptr<FunctionApproximator> new_approximator;
    new_approximator = buildFAFromLinearParams(linear_approximator_params, samples_center, D, A, action_id);
    // Copying old function approximator
    std::unique_ptr<FunctionApproximator> old_approximator;
    old_approximator = leaf_fa.clone();
    // Define Function approximator which will replace
    std::unique_ptr<Split> split(new LinearSplit(hyperplane_coeffs, offset));
    std::vector<std::unique_ptr<FunctionApproximator>> approximators;
    approximators.push_back(std::move(new_approximator));
    approximators.push_back(std::move(old_approximator));
    return std::unique_ptr<FATree>(new FATree(std::move(split), approximators));
  };
  // Defining evaluation function
  rhoban_bbo::Optimizer::RewardFunc reward_func = [this, parameters_to_approximator,
                                                   &training_states](const Eigen::VectorXd& parameters,
                                                                     std::default_random_engine* engine) {
    return evalMutation(parameters, training_states, parameters_to_approximator, engine);
  };
  Eigen::VectorXd optimized_parameters = optimize(reward_func, parameters_space, initial_parameters, engine);
  // Evaluating data with a larger set
  *score = evalMutation(optimized_parameters, validation_states, parameters_to_approximator, engine);
  // Debug
  std::cout << "\t->Optimized params: " << optimized_parameters.transpose() << std::endl;
  std::cout << "\t->Avg reward: " << *score << std::endl;
  // return splitted tree
  std::unique_ptr<FunctionApproximator> optimized_approximator;
  optimized_approximator = parameters_to_approximator(optimized_parameters);
  std::unique_ptr<FATree> splitted_tree;
  splitted_tree = policy_tree->copyAndReplaceLeaf(training_states[0], std::move(optimized_approximator));
  return std::move(splitted_tree);
}

Eigen::MatrixXd PML2::getParametersSpaces(int action_id) const
{
  // Which space is used for parameters
  Eigen::MatrixXd parameters_space;
  Eigen::MatrixXd space_for_parameters = problem->getStateLimits();
  parameters_space =
      LinearApproximator::getParametersSpace(space_for_parameters, problem->getActionLimits(action_id), false);
  return parameters_space;
}

std::string PML2::getClassName() const
{
  return "PML2";
}

Json::Value PML2::toJson() const
{
  throw std::logic_error("PML2::toJson: not implemented");
}

void PML2::fromJson(const Json::Value& v, const std::string& dir_name)
{
  // Calling parent implementation
  BlackBoxLearner::fromJson(v, dir_name);
  // Reading class variables
  rhoban_utils::tryRead(v, "training_evaluations", &training_evaluations);
  rhoban_utils::tryRead(v, "split_margin", &split_margin);
  rhoban_utils::tryRead(v, "evaluations_ratio", &evaluations_ratio);
  rhoban_utils::tryRead(v, "age_basis", &age_basis);
  rhoban_utils::tryRead(v, "use_linear_splits", &use_linear_splits);
  // Optimizer is mandatory
  optimizer = rhoban_bbo::OptimizerFactory().read(v, "optimizer", dir_name);
  // Read Policy if provided (optional)
  PolicyFactory().tryRead(v, "policy", dir_name, &policy);
  // Performing some checks
  if (split_margin < 0 || split_margin >= 0.5)
  {
    throw std::logic_error("PML2::fromJson: invalid value for split_margin");
  }
  // TODO: add checks on probabilities
  // Synchronize number of threads
  setNbThreads(nb_threads);
}

std::unique_ptr<Policy> PML2::buildPolicy(const FATree& tree)
{
  std::unique_ptr<FATree> tree_copy(static_cast<FATree*>(tree.clone().release()));
  std::unique_ptr<Policy> result(new FAPolicy(std::move(tree_copy)));
  result->setActionLimits(problem->getActionsLimits());
  return std::move(result);
}

std::vector<Eigen::VectorXd> PML2::getInitialStates(const MutationCandidate& mc, bool training,
                                                    std::default_random_engine* engine)
{
  size_t nb_evaluations_allowed = getTrainingEvaluations();
  if (!training)
    nb_evaluations_allowed = getNbEvaluationTrials();
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

bool PML2::isMutationAllowed(const MutationCandidate& mc) const
{
  return mc.visited_states.size() > 10;
}

bool PML2::submitTree(std::unique_ptr<rhoban_fa::FATree> new_tree, const std::vector<Eigen::VectorXd>& initial_states,
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
    std::cout << "\t-> replacing tree" << std::endl;
    policy_tree = std::move(new_tree);
    policy = std::move(new_policy);
    return true;
  }
  std::cout << "\t-> not replacing tree" << std::endl;
  return false;
}

void PML2::postSplitUpdate(int node_id, int nb_nodes_added)
{
  int max_id_before = policy_tree->getNodesCount();
  // Moving existing mutation candidates
  for (int old_idx = max_id_before; old_idx > node_id; old_idx--)
  {
    if (mutation_candidates.count(old_idx) > 0)
    {
      mutation_candidates[old_idx + nb_nodes_added] = mutation_candidates[old_idx];
      mutation_candidates.erase(old_idx);
    }
  }
  // Adding new candidates
  for (int child = 1; child <= node_id + nb_nodes_added; child++)
  {
    mutation_candidates[node_id + child].mutation_score = 0;
    mutation_candidates[node_id + child].last_training = iterations;
  }
  // Removing splitted node mutation candidate
  mutation_candidates.erase(node_id);
  // Updating policy tree nodes count
  policy_tree->updateNodesCount();
}

double
PML2::evalMutation(const Eigen::VectorXd& parameters, const std::vector<Eigen::VectorXd>& initial_states,
                   std::function<std::unique_ptr<rhoban_fa::FATree>(const Eigen::VectorXd&)> parameters_to_approximator,
                   std::default_random_engine* engine)
{
  std::unique_ptr<FATree> new_approximator;
  new_approximator = parameters_to_approximator(parameters);
  // Replace approximator
  std::unique_ptr<FATree> new_tree;
  new_tree = policy_tree->copyAndReplaceLeaf(initial_states[0], std::move(new_approximator));
  // build policy and evaluate
  // TODO: avoid this second copy of the tree which is not necessary
  std::unique_ptr<Policy> policy = buildPolicy(*new_tree);
  return evaluation(*policy, initial_states, engine);
}

}  // namespace csa_mdp
