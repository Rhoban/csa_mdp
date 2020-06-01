#pragma once

#include "rhoban_csa_mdp/solvers/black_box_learner.h"

#include "rhoban_bbo/optimizer.h"

#include "rhoban_fa/fa_tree.h"

namespace csa_mdp
{
/// An algorithm learning policies using mutation of the current policy, the
/// policy is always a tree of linear approximators.
///
/// At each mutation, the process is the following:
/// - Refine current leaf
/// - Try to split current leaf
///
/// The key parameters for optimization are the following:
/// - nb_evaluation_trials (see BlackBoxLearner)
/// - training_evaluations
/// - evaluations_ratio
///
/// Choice of the leaf to mutate is based on following parameters:
/// - age_basis
/// - use_density_score
///
/// Sampling of initial states is controled by:
/// - use_visited_states
class PML2 : public BlackBoxLearner
{
protected:
  /// All the information relative to a mutation candidate are stored in this
  /// structure
  struct MutationCandidate
  {
    //    /// Which  is concerned by the candidate
    //    Eigen::MatrixXd space;
    /// Weight of the mutation in the random selection process
    double mutation_score;
    /// At which iteration was this mutation trained for the last time?
    int last_training;
    /// States met in the current leaf
    std::vector<Eigen::VectorXd> visited_states;
  };

public:
  PML2();
  virtual ~PML2();

  /// Return the number of trials which should be used for evaluation
  int getNbEvaluationTrials() const;

  /// How many evaluations are required inside a single space for evaluation of
  /// the reward during the training phase
  int getTrainingEvaluations() const;

  /// Return the number of calls to the reward function allowed for the optimizer
  int getOptimizerMaxCall() const;

  virtual void init(std::default_random_engine* engine) override;
  virtual void update(std::default_random_engine* engine) override;

  virtual void setNbThreads(int nb_threads) override;

  /// Choose a mutation among the available candidates according to their scores
  int getMutationId(std::default_random_engine* engine);

  /// Evaluate policy and return average reward
  /// If required by internal coniguration, save visited states in
  /// mutation candidates
  /// TODO: complexity should be improved
  double evalAndGetStates(std::default_random_engine* engine);

  /// Update mutation scores according to their properties
  void updateMutationsScores();

  /// Root of the mutation process
  void mutate(int mutation_id, std::default_random_engine* engine);

  /// Mutate a leaf: there is several possibilities:
  /// 1: Refine FunctionApproximator for the leaf
  /// 2: Split the leaf
  void mutateLeaf(int mutation_id, std::default_random_engine* engine);

  /// Try to refine the function approximator for the given mutation id and the given action_id
  void tryRefine(int mutation_id, int action_id, std::default_random_engine* engine);

  /// Try to split along every dimensions and keeps the best split
  void applyBestSplit(int mutation_id, std::default_random_engine* engine);

  /// Try to find the best split based on linear transformations
  void applyBestLinearSplit(int mutation_id, std::default_random_engine* engine);

  /// Try to split along 'split_dim' at the given mutation.
  /// Return the FATree built to replace current approximator and update score
  std::unique_ptr<rhoban_fa::FATree> trySplit(int split_dim, const std::vector<Eigen::VectorXd>& training_states,
                                              const std::vector<Eigen::VectorXd>& validation_states,
                                              std::default_random_engine* engine, double* score);

  /// Try to split using a linear split and a new action 'action_id' at the given mutation.
  /// Return the FATree built to replace current approximator and update score
  std::unique_ptr<rhoban_fa::FATree> tryLinearSplit(int action_id, const std::vector<Eigen::VectorXd>& training_states,
                                                    const std::vector<Eigen::VectorXd>& validation_states,
                                                    std::default_random_engine* engine, double* score);

  /// Return the parameters space for training a linear model given the
  /// refinement type
  Eigen::MatrixXd getParametersSpaces(int action_id) const;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  /// Return the best candidate found
  /// rf: the reward function
  /// space: The space allowed for parameters
  /// guess: The initial candidate
  /// engine: Used to draw random numbers
  /// evaluation_gain: Multiplies the usual number of evaluations allowed to the optimizer
  Eigen::VectorXd optimize(rhoban_bbo::Optimizer::RewardFunc rf, const Eigen::MatrixXd& space,
                           const Eigen::VectorXd& guess, std::default_random_engine* engine,
                           double evaluation_mult = 1);

  /// Clone the given tree and use it to build a policy. Also set the action
  /// limits
  std::unique_ptr<Policy> buildPolicy(const rhoban_fa::FATree& tree);

  /// If 'training':
  /// - returns a maximum of 'trainingEvaluations' states
  /// Else
  /// - returns a maximum of nb evaluation trials
  std::vector<Eigen::VectorXd> getInitialStates(const MutationCandidate& mc, bool training,
                                                std::default_random_engine* engine);

  bool isMutationAllowed(const MutationCandidate& mc) const;

  /// Compare the new_tree with current tree. Replace the current tree if the
  /// new tree is better
  /// Return true if the tree has been replaced, false otherwise
  bool submitTree(std::unique_ptr<rhoban_fa::FATree> new_tree, const std::vector<Eigen::VectorXd>& initial_states,
                  std::default_random_engine* engine);

  /// Update the mutation candidates after a split on given id
  /// - Apply appropriate offset on nodes with higher ids
  /// - Add mutation candidates for the new leaves
  /// - Remove mutation candidate on 'node_id'
  /// - Update policy_tree ids
  void postSplitUpdate(int node_id, int nb_nodes_added);

protected:
  double
  evalMutation(const Eigen::VectorXd& parameters, const std::vector<Eigen::VectorXd>& initial_states,
               std::function<std::unique_ptr<rhoban_fa::FATree>(const Eigen::VectorXd&)> parameters_to_approximator,
               std::default_random_engine* engine);

  /// The list of mutations available
  std::map<int, MutationCandidate> mutation_candidates;

  /// The current version of the tree
  std::unique_ptr<rhoban_fa::FATree> policy_tree;

  /// Optimizer used to change split position or to train models
  /// TODO: later, several optimizers should be provided
  std::unique_ptr<rhoban_bbo::Optimizer> optimizer;

  /// The number of rollouts used to obtain the reward associated to the local
  /// parameters of the policy while optimizing them.
  int training_evaluations;

  /// When operating a split on a dimension, this parameters ensures that the
  /// split value is inside [min + delta, max - delta], with:
  /// - delta: (max-min)*split_margin
  /// split_margin has to be in [0, 0.5[ (otherwise, split space is empty)
  double split_margin;

  /// The total number of rollouts evaluated at each step by the optimizer is
  /// equal to 'evaluations_ratio' times the number of trajectories used to
  /// compute the average reward over the entire space
  double evaluations_ratio;

  /// age_score = age_basis ^ (nb_update_since_update / nb_leafs)
  double age_basis;

  /// If false: uses orthogonal splits
  bool use_linear_splits;
};

}  // namespace csa_mdp
