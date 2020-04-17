#include <deep_q_learning.h>
#include <grid_world.h>
#include <torch_utils.h>

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <iterator>
#include <string>

using drl_in_action::grid_world::GridWorld;

const std::string kModelFile = "dql_model.pt";

int main(int argc, char* argv[]) {
  // TODO: train or test model via command line args
  spdlog::set_level(spdlog::level::info);

  auto model = deep_q_learning::create_model(torch::kCUDA);

  bool model_loaded = true;
  try {
    torch::load(model, kModelFile);
    spdlog::info("The model has been loaded from {}", kModelFile);
  } catch (...) {
    model_loaded = false;
    spdlog::info("The model could not be loaded from {}", kModelFile);
  }

  if (model_loaded) {
    deep_q_learning::test_model<GridWorld>(model);
    return 0;
  }

  deep_q_learning::train_model<GridWorld>(model, deep_q_learning::TrainOptions().epochs(100));
  torch::save(model, kModelFile);
  return 0;
}
