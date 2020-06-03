#include "rhoban_csa_mdp/core/fa_policy.h"

#include "rhoban_fa/function_approximator_factory.h"

#include "rhoban_random/multivariate_gaussian.h"
#include "rhoban_random/tools.h"

using namespace rhoban_utils;
using rhoban_fa::FunctionApproximatorFactory;
using rhoban_random::MultivariateGaussian;

namespace csa_mdp
{
FAPolicy::FAPolicy() : apply_noise(false)
{
  engine = rhoban_random::getRandomEngine();
}

FAPolicy::FAPolicy(std::unique_ptr<rhoban_fa::FunctionApproximator> fa_) : fa(std::move(fa_)), apply_noise(false)
{
  engine = rhoban_random::getRandomEngine();
}

void FAPolicy::setRandomness(bool new_apply_noise)
{
  apply_noise = new_apply_noise;
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd& state)
{
  return getRawAction(state, &engine);
}

Eigen::VectorXd FAPolicy::getRawAction(const Eigen::VectorXd& state, std::default_random_engine* external_engine) const
{
  bool delete_engine = false;
  if (apply_noise && external_engine == nullptr)
  {
    delete_engine = true;
    external_engine = rhoban_random::newRandomEngine();
  }
  Eigen::VectorXd mean;
  Eigen::MatrixXd covar;
  fa->predict(state, mean, covar);

  Eigen::VectorXd cmd;
  if (apply_noise)
  {
    cmd = MultivariateGaussian(mean, covar).getSample(external_engine);
  }
  else
  {
    cmd = mean;
  }

  if (delete_engine)
  {
    delete (external_engine);
  }
  return cmd;
}

std::string FAPolicy::getClassName() const
{
  return "FAPolicy";
}

Json::Value FAPolicy::toJson() const
{
  Json::Value v;
  v["noise"] = apply_noise;
  throw std::logic_error("FAPolicy:toJson: not implemented");
}

void FAPolicy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  std::string abs_path, rel_path, path;
  rhoban_utils::tryRead<std::string>(v, "abs path", &abs_path);
  rhoban_utils::tryRead<std::string>(v, "rel path", &rel_path);
  if (abs_path != "" && rel_path != "")
  {
    throw JsonParsingError("FAPolicy::fromJson: both 'abs path' and 'rel path' specified");
  }
  else if (abs_path == "" && rel_path == "")
  {
    throw JsonParsingError("FAPolicy::fromJson: no 'abs path' neither 'rel path' specified");
  }
  else if (abs_path != "")
  {
    path = abs_path;
  }
  else
  {
    path = dir_name + rel_path;
  }
  FunctionApproximatorFactory().loadFromFile(path, fa);
  rhoban_utils::tryRead(v, "noise", &apply_noise);
}

void FAPolicy::saveFA(const std::string& filename) const
{
  if (!fa)
    return;
  fa->save(filename);
}

std::unique_ptr<rhoban_fa::FATree> FAPolicy::extractFATree() const
{
  try
  {
    const rhoban_fa::FATree& tree = dynamic_cast<const rhoban_fa::FATree&>(*fa);
    return std::unique_ptr<rhoban_fa::FATree>((rhoban_fa::FATree*)tree.clone().release());
  }
  catch (const std::bad_cast& exc)
  {
    throw std::runtime_error(DEBUG_INFO + " not implemented for non FATree approximators");
  }
}

}  // namespace csa_mdp
