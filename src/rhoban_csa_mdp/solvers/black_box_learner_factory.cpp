#include "rhoban_csa_mdp/solvers/black_box_learner_factory.h"

#include "rhoban_csa_mdp/solvers/alp_gmm.h"
#include "rhoban_csa_mdp/solvers/dbcl.h"
#include "rhoban_csa_mdp/solvers/lppi.h"
#include "rhoban_csa_mdp/solvers/policy_mutation_learner.h"
#include "rhoban_csa_mdp/solvers/pml2.h"
#include "rhoban_csa_mdp/solvers/tree_policy_iteration.h"

namespace csa_mdp
{
BlackBoxLearnerFactory::BlackBoxLearnerFactory()
{
  registerBuilder("TreePolicyIteration",
                  []() { return std::unique_ptr<TreePolicyIteration>(new TreePolicyIteration); });
  registerBuilder("PolicyMutationLearner",
                  []() { return std::unique_ptr<PolicyMutationLearner>(new PolicyMutationLearner); });
  registerBuilder("PML2", []() { return std::unique_ptr<PML2>(new PML2); });
  registerBuilder("LPPI", []() { return std::unique_ptr<LPPI>(new LPPI); });
  registerBuilder("ALP-GMM", []() { return std::unique_ptr<ALPGMM>(new ALPGMM); });
  registerBuilder("DBCL", []() { return std::unique_ptr<DBCL>(new DBCL); });
}

}  // namespace csa_mdp
