#include <rhoban_csa_mdp/solvers/dbcl.h>

#include <rhoban_csa_mdp/solvers/black_box_learner_factory.h>

namespace csa_mdp
{
DBCL::DBCL() : nb_successful_steps(0), nb_difficulty_steps(10), difficulty(0.0)
{
}

DBCL::~DBCL()
{
}

void DBCL::init(std::default_random_engine* engine)
{
  student->openLogs();
  student->setTask(student->getAutomatedTask(difficulty));
  student->init(engine);
}
void DBCL::update(std::default_random_engine* engine)
{
  student->setTask(student->getAutomatedTask(difficulty));
  student->update(engine);
  last_score = student->getLastScore();
  publishIteration();

  if (last_score > performance_required && nb_difficulty_steps > 0)
  {
    nb_successful_steps++;
    difficulty = nb_successful_steps / (double)(nb_difficulty_steps);
    std::cout << "increasing difficulty to " << difficulty << std::endl;
    // Resetting some properties of the student. This hack currently works only for LPPI.
    student->setTask(student->getAutomatedTask(difficulty));
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
std::vector<std::string> DBCL::getMetaColumns() const
{
  std::vector<std::string> result = BlackBoxLearner::getMetaColumns();
  result.push_back("difficulty");
  for (const std::string& student_meta_column : student->getMetaColumns())
    result.push_back("student." + student_meta_column);
  return result;
}
std::map<std::string, std::string> DBCL::getMetaData() const
{
  std::map<std::string, std::string> result = BlackBoxLearner::getMetaData();
  result["difficulty"] = std::to_string(difficulty);
  for (const auto& student_entry : student->getMetaData())
    result["student." + student_entry.first] = student_entry.second;
  return result;
}
}  // namespace csa_mdp
