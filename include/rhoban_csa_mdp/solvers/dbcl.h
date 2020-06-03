#pragma once

#include "rhoban_csa_mdp/solvers/black_box_learner.h"

#include <rhoban_utils/tables/string_table.h>

namespace csa_mdp
{
/// Difficulty Based Curriculum Learner is a quick and dirty experiment in order to test an
/// approach where difficulty of the problem is gradually increased
class DBCL : public BlackBoxLearner
{
public:
  DBCL();
  virtual ~DBCL();

  virtual void init(std::default_random_engine* engine) override;

  /// Perform rollouts according to the OpenLoopPlanner
  virtual void update(std::default_random_engine* engine) override;

  virtual void setNbThreads(int nb_threads) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  std::unique_ptr<BlackBoxLearner> student;

  /// The average reward required to move toward next level
  double performance_required;

  /// The number of difficulty steps which have been properly mastered
  int nb_successful_steps;

  /// The number of intermediary difficulty steps that are used
  int nb_difficulty_steps;

  /// Current difficulty [0,1]
  double difficulty;

  /// Storing evolution of results
  rhoban_utils::StringTable output_log;
};

}  // namespace csa_mdp
