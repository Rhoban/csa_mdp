#pragma once

#include <rhoban_csa_mdp/solvers/black_box_learner.h>
#include <kd_trees/kd_tree_container.h>

#include <rhoban_random/gaussian_mixture_model.h>
#include <rhoban_random/expectation_maximization.h>
#include <rhoban_utils/tables/string_table.h>

namespace csa_mdp
{
/// Absolute Learning Progress Gaussian Mixture Model
/// - Implements the algorithm described:
///   - by: Portelas, R., Colas, C., Hofmann, K., & Oudeyer, P.-Y. (2019).
///   - in: Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments.
///   - Retrieved from: http://arxiv.org/abs/1910.07224
class ALPGMM : public BlackBoxLearner
{
public:
  ALPGMM();
  ~ALPGMM();

  void init(std::default_random_engine* engine) override;

  /// Perform rollouts according to the OpenLoopPlanner
  void update(std::default_random_engine* engine) override;

  /// Run `fitting_rate` experiments with random tasks
  void runRound(std::default_random_engine* engine);

  /// Evaluate the performance of the student on an uniform distribution of tasks in the task-space
  double evaluate(std::default_random_engine* engine) override;

  /// Send the task to the student, runs a learning update and returns the reward
  double applyTask(const Eigen::VectorXd& task, std::default_random_engine* engine);

  void setNbThreads(int nb_threads) override;

  std::string getClassName() const override;
  Json::Value toJson() const override;
  void fromJson(const Json::Value& v, const std::string& dir_name) override;

  std::vector<std::string> getMetaColumns() const override;
  std::map<std::string, std::string> getMetaData() const override;

private:
  std::unique_ptr<BlackBoxLearner> student;

  /// The window containing the couples (parameters, ALP) for the last 'fitting_rate' entries
  std::deque<std::pair<Eigen::VectorXd, double>> alp_window;

  /// The history containing all the couples (parameters, reward) since the beginning
  std::unique_ptr<kd_trees::KdTreeContainer<double>> reward_history;

  /// The number of parameters sampled updating the GaussianMixtureModels
  int fitting_rate;

  /// The probability of selecting a uniform random sample for the task
  double p_rnd;

  /// The number of student updates per task:
  /// This parameter is introduced for students who need to be reinitialized after every task change and are requiring
  /// a significant number of tasks to make real improvements
  /// For original ALPGMM, use 1.
  int updates_per_task;

  /// The expectation maximization used to compute the algorithm
  rhoban_random::ExpectationMaximization em;

  /// Current content of the gaussian
  rhoban_random::GaussianMixtureModel gmm;

  /// Last avg_reward obtained when applying the tasks used for learning (unstable distribution)
  double last_learning_score;

  /// Fit and select the best GMM based on alp_window
  void updateGMM();
};

}  // namespace csa_mdp
