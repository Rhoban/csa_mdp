#include <rhoban_csa_mdp/solvers/dbcl.h>

#include <rhoban_csa_mdp/solvers/black_box_learner_factory.h>

namespace csa_mdp
{
DBCL::DBCL()
  : nb_successful_steps(0)
  , nb_difficulty_steps(10)
  , difficulty(0.0)
  , output_log({ "iteration", "score", "elapsed", "difficulty" })
{
  output_log.startStreaming("dbcl_results.csv");
}

DBCL::~DBCL()
{
}

void DBCL::init(std::default_random_engine* engine)
{
  student->setTask(student->getAutomatedTask(difficulty));
  student->init(engine);
}
void DBCL::update(std::default_random_engine* engine)
{
  student->setTask(student->getAutomatedTask(difficulty));
  student->update(engine);
  double performance = student->evaluate(engine);
  double elapsed = diffSec(learning_start, rhoban_utils::TimeStamp::now());
  std::map<std::string, std::string> log_row;
  log_row["iteration"] = std::to_string(iterations);
  log_row["score"] = std::to_string(performance);
  log_row["elapsed"] = std::to_string(elapsed);
  log_row["difficulty"] = std::to_string(difficulty);
  output_log.insertRow(log_row);

  if (performance > performance_required && nb_difficulty_steps > 0)
  {
    nb_successful_steps++;
    difficulty = nb_successful_steps / (double)(nb_difficulty_steps);
    std::cout << "increasing difficulty to " << difficulty << std::endl;
    // Resetting some properties of the student. This hack currently works only for LPPI.
    student->init(engine);
  }
}
void DBCL::setNbThreads(int nb_threads)
{
  student->setNbThreads(nb_threads);
}
std::string DBCL::getClassName() const
{
  return "DBCL";
}
Json::Value DBCL::toJson() const
{
  throw std::logic_error(DEBUG_INFO + "Not implemented");
}
void DBCL::fromJson(const Json::Value& v, const std::string& dir_name)
{
  BlackBoxLearner::fromJson(v, dir_name);
  student = BlackBoxLearnerFactory().build(v["student"], dir_name);
  rhoban_utils::tryRead(v, "performance_required", &performance_required);
  rhoban_utils::tryRead(v, "nb_difficulty_steps", &nb_difficulty_steps);
  if (nb_difficulty_steps == 0)
  {
    difficulty = 1;
  }
}
}  // namespace csa_mdp
