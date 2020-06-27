#include <rhoban_csa_mdp/solvers/alp_gmm.h>

#include <rhoban_csa_mdp/solvers/black_box_learner_factory.h>

#include <rhoban_random/tools.h>
#include <rhoban_utils/timing/time_stamp.h>

using rhoban_utils::TimeStamp;

namespace csa_mdp
{
ALPGMM::ALPGMM() : fitting_rate(250), p_rnd(0.2), updates_per_task(1), last_learning_score(0.0)
{
}

ALPGMM::~ALPGMM()
{
}

void ALPGMM::init(std::default_random_engine* engine)
{
  student->openLogs();
  runRound(engine);
}

void ALPGMM::update(std::default_random_engine* engine)
{
  updateGMM();
  runRound(engine);
}

void ALPGMM::runRound(std::default_random_engine* engine)
{
  TimeStamp start = TimeStamp::now();
  std::uniform_real_distribution<double> random_task_distrib(0.0, 1.0);
  Eigen::VectorXd run_rewards(fitting_rate);
  for (int run = 0; run < fitting_rate; run++)
  {
    Eigen::VectorXd task;
    if (gmm.size() == 0 || random_task_distrib(*engine) < p_rnd)
      task = rhoban_random::getUniformSample(problem->getTaskLimits(), engine);
    else
    {
      std::vector<double> gaussian_alp_weights;
      for (size_t gaussian_idx = 0; gaussian_idx < gmm.size(); gaussian_idx++)
      {
        gaussian_alp_weights.push_back(gmm.getGaussian(gaussian_idx).getMean()(0));
      }
      int gaussian_idx = rhoban_random::sampleWeightedIndices(gaussian_alp_weights, 1, engine)[0];
      task = gmm.getGaussian(gaussian_idx).getSample(engine);
      // Removing the 'alp' part of the entry
      task = task.segment(1, task.rows() - 1);
    }
    double reward = applyTask(task, engine);
    run_rewards(run) = reward;
    if (reward_history->size() != 0)
    {
      double old_reward = reward_history->getNearestNeighborData(task);
      double alp = std::fabs(reward - old_reward);
      alp_window.push_back({ task, alp });
      if ((int)alp_window.size() > fitting_rate)
        alp_window.pop_front();
    }
    reward_history->pushEntry(task, reward);
  }
  last_learning_score = run_rewards.mean();
  TimeStamp post_learning = TimeStamp::now();
  writeTime("learning", diffSec(start, post_learning));
  if (verbosity >= 1)
    std::cout << "Evaluating on uniform task distribution" << std::endl;
  last_score = evaluate(engine);
  TimeStamp post_evaluation = TimeStamp::now();
  writeTime("evaluation", diffSec(post_learning, post_evaluation));
  publishIteration();
}

/// Evaluate the performance of current policy
double ALPGMM::evaluate(std::default_random_engine* engine)
{
  double total_reward = 0;
  for (int eval = 0; eval < nb_evaluation_trials; eval++)
  {
    Eigen::VectorXd task = rhoban_random::getUniformSample(problem->getTaskLimits(), engine);
    student->setTask(task);
    total_reward += student->evaluate(1, engine);
  }
  return total_reward / nb_evaluation_trials;
}

double ALPGMM::applyTask(const Eigen::VectorXd& task, std::default_random_engine* engine)
{
  if (verbosity >= 2)
    std::cout << "Running student with task: " << task.transpose() << std::endl;
  student->setTask(task);
  student->init(engine);
  for (int update = 0; update < updates_per_task; update++)
    student->update(engine);
  return student->getLastScore();
}

void ALPGMM::setNbThreads(int nb_threads)
{
  BlackBoxLearner::setNbThreads(nb_threads);
  student->setNbThreads(nb_threads);
}

std::string ALPGMM::getClassName() const
{
  return "ALPGMM";
}

Json::Value ALPGMM::toJson() const
{
  Json::Value v = BlackBoxLearner::toJson();
  v["student"] = student->toFactoryJson();
  v["fitting_rate"] = fitting_rate;
  v["p_rnd"] = p_rnd;
  v["updates_per_task"] = updates_per_task;
  v["em"] = em.toJson();
  return v;
}

void ALPGMM::fromJson(const Json::Value& v, const std::string& dir_name)
{
  BlackBoxLearner::fromJson(v, dir_name);
  student = BlackBoxLearnerFactory().build(v["student"], dir_name);
  rhoban_utils::tryRead(v, "fitting_rate", &fitting_rate);
  rhoban_utils::tryRead(v, "p_rnd", &p_rnd);
  rhoban_utils::tryRead(v, "updates_per_task", &updates_per_task);
  em.tryRead(v, "em", dir_name);

  reward_history.reset(new kd_trees::KdTreeContainer<double>(problem->getTaskLimits()));
}

std::vector<std::string> ALPGMM::getMetaColumns() const
{
  std::vector<std::string> result = BlackBoxLearner::getMetaColumns();
  result.push_back("learning_score");
  for (const std::string& student_meta_column : student->getMetaColumns())
    result.push_back("student." + student_meta_column);
  return result;
}
std::map<std::string, std::string> ALPGMM::getMetaData() const
{
  std::map<std::string, std::string> result = BlackBoxLearner::getMetaData();
  result["learning_score"] = last_learning_score;
  for (const auto& student_entry : student->getMetaData())
    result["student." + student_entry.first] = student_entry.second;
  return result;
}

void ALPGMM::updateGMM()
{
  if (verbosity >= 1)
    std::cout << DEBUG_INFO << " updating GMM" << std::endl;
  int point_dims = problem->getTaskLimits().rows() + 1;
  Eigen::MatrixXd points(point_dims, alp_window.size());
  int i = 0;
  for (const auto& entry : alp_window)
  {
    Eigen::VectorXd point(point_dims);
    point(0) = entry.second;
    point.segment(1, point_dims - 1) = entry.first;
    points.col(i) = point;
    i++;
  }
  std::cout << "Points size: " << points.rows() << "x" << points.cols() << std::endl;
  em.analyze(points);
  gmm = em.getBestResult(rhoban_random::ExpectationMaximization::ScoreCriterion::AIC).gmm;

  if (verbosity >= 1)
    std::cout << "Best GMM has " << gmm.size() << " gaussians of dimension " << gmm.dimension() << std::endl;
  if (verbosity >= 2)
    std::cout << gmm.toJsonStringHuman() << std::endl;
}

}  // namespace csa_mdp
