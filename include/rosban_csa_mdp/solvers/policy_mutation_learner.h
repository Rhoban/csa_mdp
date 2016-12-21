#pragma once

#include "rosban_csa_mdp/solvers/black_box_learner.h"

#include "rosban_bbo/optimizer.h"

namespace csa_mdp
{

class PolicyMutationLearner : public BlackBoxLearner {
protected:

  /// All the information relative to a mutation candidate are stored in this
  /// structure
  struct MutationCandidate {
    /// Which space is concerned by the candidate
    Eigen::MatrixXd space;
    /// What was the score of this candidate after training
    double post_training_score;
    /// Weight of the mutation in the random selection process
    double mutation_score;
    /// At which iteration was this mutation trained for the last time?
    int last_training;
    /// Is the candidate a leaf or a pre_leaf?
    bool is_leaf;
  };

public:
  PolicyMutationLearner();
  virtual ~PolicyMutationLearner();

  virtual void init(std::default_random_engine * engine) override;
  virtual void update(std::default_random_engine * engine) override;

  virtual void setNbThreads(int nb_threads) override;

  void mutate(std::default_random_engine * engine);

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:
  /// The list of mutations available
  std::vector<MutationCandidate> mutation_candidates;

  /// The current version of the tree
  std::unique_ptr<FATree> policy_tree;

  /// Current policy
  std::unique_ptr<Policy> policy;

  /// Optimizer used to change split position or to train models
  /// TODO: later, several optimizers should be provided
  std::unique_ptr<rosban_bbo::Optimizer> optimizer;
};

}