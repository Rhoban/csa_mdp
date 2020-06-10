#include "rhoban_csa_mdp/solvers/black_box_learner.h"

#include "rhoban_csa_mdp/core/fa_policy.h"
#include "rhoban_csa_mdp/core/problem_factory.h"

#include "rhoban_random/tools.h"
#include "rhoban_utils/threading/multi_core.h"

using rhoban_utils::StringTable;
using rhoban_utils::TimeStamp;

namespace csa_mdp
{
BlackBoxLearner::BlackBoxLearner()
  : nb_threads(1)
  , time_budget(60)
  , discount(0.98)
  , trial_length(50)
  , nb_evaluation_trials(100)
  , iterations(0)
  , last_score(std::numeric_limits<double>::lowest())
  , verbosity(0)
{
}

BlackBoxLearner::~BlackBoxLearner()
{
}

std::unique_ptr<Policy> BlackBoxLearner::buildPolicy(const rhoban_fa::FunctionApproximator& fa) const
{
  std::unique_ptr<Policy> p(new FAPolicy(fa.clone()));
  p->setActionLimits(problem->getActionsLimits());
  return p;
}

void BlackBoxLearner::run(std::default_random_engine* engine)
{
  openLogs();
  init(engine);
  // Main learning loop
  learning_start = rhoban_utils::TimeStamp::now();
  double elapsed = 0;
  while (elapsed < time_budget)
  {
    iterations++;
    std::cout << "Iteration " << iterations << std::endl;
    update(engine);
    // Stop if time has elapsed
    elapsed = diffSec(learning_start, rhoban_utils::TimeStamp::now());
  }
  std::cout << "Ending after " << elapsed << " time" << std::endl;
}
double BlackBoxLearner::evaluate(std::default_random_engine* engine)
{
  return evaluatePolicy(*policy, engine);
}

double BlackBoxLearner::evaluatePolicy(const Policy& p, std::default_random_engine* engine) const
{
  return evaluatePolicy(p, nb_evaluation_trials, engine);
}

double BlackBoxLearner::evaluatePolicy(const Policy& p, int nb_evaluations, std::default_random_engine* engine,
                                       std::vector<Eigen::VectorXd>* visited_states) const
{
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rhoban_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Rewards + visited_states are computed by different threads and stored in the same vector
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  std::vector<std::vector<Eigen::VectorXd>> visited_states_per_thread(nb_evaluations);
  bool store_visited_states = visited_states != nullptr;
  // The task which has to be performed :
  rhoban_utils::MultiCore::StochasticTask task = [this, &p, &rewards, &visited_states_per_thread, store_visited_states](
                                                     int start_idx, int end_idx, std::default_random_engine* engine) {
    for (int idx = start_idx; idx < end_idx; idx++)
    {
      Eigen::VectorXd state = problem->getStartingState(engine);
      double gain = 1.0;
      for (int step = 0; step < trial_length; step++)
      {
        if (store_visited_states)
        {
          visited_states_per_thread[idx].push_back(state);
        }
        Eigen::VectorXd action = p.getAction(problem->getLearningState(state), engine);
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
  // Fill visited states if required
  if (store_visited_states)
  {
    for (const std::vector<Eigen::VectorXd>& eval_visited_states : visited_states_per_thread)
    {
      for (const Eigen::VectorXd& state : eval_visited_states)
      {
        visited_states->push_back(state);
      }
    }
  }
  // Result
  return rewards.mean();
}

double BlackBoxLearner::localEvaluation(const Policy& p, const Eigen::MatrixXd& space, int nb_evaluations,
                                        std::default_random_engine* engine) const
{
  // Sampling starting states
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  // The task which has to be performed :
  rhoban_utils::MultiCore::StochasticTask task = [this, &p, &rewards, &space](int start_idx, int end_idx,
                                                                              std::default_random_engine* engine) {
    // 1: Generating states
    int thread_evaluations = end_idx - start_idx;
    std::vector<Eigen::VectorXd> starting_states;
    starting_states = rhoban_random::getUniformSamples(space, thread_evaluations, engine);
    // 2: Simulating trajectories
    try
    {
      for (int idx = 0; idx < thread_evaluations; idx++)
      {
        rewards(idx + start_idx) = runEpisode(p, starting_states[idx], engine);
      }
    }
    catch (const std::runtime_error& exc)
    {
      std::ostringstream oss;
      oss << "BlackBoxLearner::localEvaluation:task: " << exc.what() << std::endl;
      std::cerr << oss.str();
      throw exc;
    }
  };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rhoban_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Running computation
  rhoban_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}

double BlackBoxLearner::evaluation(const Policy& p, const std::vector<Eigen::VectorXd>& initial_states,
                                   std::default_random_engine* engine) const
{
  // Initializing reward list
  int nb_evaluations = initial_states.size();
  Eigen::VectorXd rewards = Eigen::VectorXd::Zero(nb_evaluations);
  // The task which has to be performed :
  rhoban_utils::MultiCore::StochasticTask task =
      [this, &p, &rewards, &initial_states](int start_idx, int end_idx, std::default_random_engine* engine) {
        // Simulating trajectories
        for (int idx = start_idx; idx < end_idx; idx++)
        {
          Eigen::VectorXd state = initial_states[idx];
          rewards(idx) = runEpisode(p, initial_states[idx], engine);
        }
      };
  // Preparing random_engines
  std::vector<std::default_random_engine> engines;
  engines = rhoban_random::getRandomEngines(std::min(nb_threads, nb_evaluations), engine);
  // Running computation
  rhoban_utils::MultiCore::runParallelStochasticTask(task, nb_evaluations, &engines);
  // Result
  return rewards.mean();
}

double BlackBoxLearner::runEpisode(const Policy& p, const Eigen::VectorXd& initial_state,
                                   std::default_random_engine* engine, Problem::Episode* episode) const
{
  double gain = 1.0;
  double reward = 0;
  if (episode)
  {
    episode->clear();
    episode->states.push_back(initial_state);
  }
  Eigen::VectorXd state = initial_state;
  for (int step = 0; step < trial_length; step++)
  {
    Eigen::VectorXd action = p.getAction(problem->getLearningState(state), engine);
    Problem::Result result = problem->getSuccessor(state, action, engine);
    state = result.successor;
    reward += gain * result.reward;
    gain = gain * discount;
    if (episode)
      episode->feed(action, result);
    if (result.terminal)
      break;
  }
  return reward;
}

double BlackBoxLearner::runEpisode(const Eigen::VectorXd& initial_state, std::default_random_engine* engine,
                                   Problem::Episode* episode) const
{
  double gain = 1.0;
  double reward = 0;
  if (episode)
  {
    episode->clear();
    episode->states.push_back(initial_state);
  }
  Eigen::VectorXd state = initial_state;
  for (int step = 0; step < trial_length; step++)
  {
    Eigen::VectorXd action = getAction(problem->getLearningState(state), engine);
    Problem::Result result = problem->getSuccessor(state, action, engine);
    state = result.successor;
    reward += gain * result.reward;
    gain = gain * discount;
    if (episode)
      episode->feed(action, result);
    if (result.terminal)
      break;
  }
  return reward;
}

Eigen::VectorXd BlackBoxLearner::getAction(const Eigen::VectorXd& state, std::default_random_engine* engine) const
{
  return policy->getAction(problem->getLearningState(state), engine);
}

void BlackBoxLearner::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
}

Json::Value BlackBoxLearner::toJson() const
{
  throw std::logic_error("BlackBoxLearner::toJson: not implemented");
}

void BlackBoxLearner::fromJson(const Json::Value& v, const std::string& dir_name)
{
  // Reading simple parameters
  rhoban_utils::tryRead(v, "nb_threads", &nb_threads);
  rhoban_utils::tryRead(v, "trial_length", &trial_length);
  rhoban_utils::tryRead(v, "nb_evaluation_trials", &nb_evaluation_trials);
  rhoban_utils::tryRead(v, "verbosity", &verbosity);
  rhoban_utils::tryRead(v, "time_budget", &time_budget);
  rhoban_utils::tryRead(v, "discount", &discount);
  rhoban_utils::tryRead(v, "results_path", &results_path);
  rhoban_utils::tryRead(v, "time_path", &time_path);

  // Getting problem
  std::shared_ptr<Problem> tmp_problem;
  std::string problem_path;
  rhoban_utils::tryRead(v, "problem_path", &problem_path);
  if (problem_path != "")
  {
    tmp_problem = ProblemFactory().buildFromJsonFile(dir_name + problem_path);
  }
  else
  {
    tmp_problem = ProblemFactory().read(v, "problem", dir_name);
  }
  problem = std::dynamic_pointer_cast<BlackBoxProblem>(tmp_problem);
  if (!problem)
  {
    throw std::runtime_error("BlackBoxLearner::fromJson: problem is not a BlackBoxProblem");
  }
}

void BlackBoxLearner::openLogs()
{
  results_log = StringTable(getMetaColumns());
  time_log = StringTable({ "iteration", "phase", "time" });
  if (results_path != "")
    results_log.startStreaming(results_path);
  if (time_path != "")
    time_log.startStreaming(time_path);
}

void BlackBoxLearner::closeLogs()
{
  results_log.endStreaming();
  time_log.endStreaming();
}

void BlackBoxLearner::writeTime(const std::string& name, double time)
{
  std::map<std::string, std::string> entry = { { "iteration", std::to_string(iterations) },
                                               { "phase", name },
                                               { "time", std::to_string(time) } };
  time_log.insertRow(entry);
}

void BlackBoxLearner::publishIteration()
{
  results_log.insertRow(getMetaData());
}

void BlackBoxLearner::setTask(const Eigen::VectorXd& task)
{
  problem->setTask(task);
}
Eigen::VectorXd BlackBoxLearner::getAutomatedTask(double difficulty) const
{
  return problem->getAutomatedTask(difficulty);
}
std::vector<std::string> BlackBoxLearner::getMetaColumns() const
{
  return { "iteration", "score", "elapsed" };
}

std::map<std::string, std::string> BlackBoxLearner::getMetaData() const
{
  double elapsed = diffSec(learning_start, rhoban_utils::TimeStamp::now());
  return { { "iteration", std::to_string(iterations) },
           { "score", std::to_string(last_score) },
           { "elapsed", std::to_string(elapsed) } };
}

double BlackBoxLearner::getLastScore() const
{
  return last_score;
}
}  // namespace csa_mdp
