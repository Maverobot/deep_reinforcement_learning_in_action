#include <grid_world.h>
#include <spdlog/spdlog.h>
#include <magic_enum.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using drl_in_action::grid_world::GridWorld;

template <class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& arr) {
  copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

torch::nn::Sequential create_model(torch::Device device) {
  const int kL1 = 64;
  const int kL2 = 150;
  const int kL3 = 100;
  const int kL4 = 4;
  torch::nn::Sequential model(torch::nn::Linear(torch::nn::LinearOptions(kL1, kL2).bias(true)),
                              torch::nn::ELU(torch::nn::ELUOptions().alpha(1).inplace(false)),
                              torch::nn::Linear(torch::nn::LinearOptions(kL2, kL3).bias(true)),
                              torch::nn::ELU(torch::nn::ELUOptions().alpha(1).inplace(false)),
                              torch::nn::Linear(torch::nn::LinearOptions(kL3, kL4).bias(true)));
  model->to(device);
  return model;
}

template <typename value_type>
torch::Tensor flat_tensor(std::vector<std::vector<value_type>> input,
                          torch::Device device = torch::kCUDA) {
  std::vector<torch::Tensor> output_vec;
  std::transform(input.begin(), input.end(), std::back_inserter(output_vec), [&device](auto& arr) {
    return torch::from_blob(arr.data(), {1, static_cast<long>(arr.size())},
                            torch::TensorOptions().dtype<value_type>())
        .clone()
        .to(device);
  });
  return torch::cat(output_vec, 1);
}

void loadParameter(torch::nn::Sequential& model,
                   const torch::OrderedDict<std::string, torch::Tensor>& new_params) {
  torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
  auto params = model->named_parameters(true /*recurse*/);
  auto buffers = model->named_buffers(true /*recurse*/);
  for (auto& val : new_params) {
    auto name = val.key();
    auto* t = params.find(name);
    if (t != nullptr) {
      t->copy_(val.value());
    } else {
      t = buffers.find(name);
      if (t != nullptr) {
        t->copy_(val.value());
      }
    }
  }
  torch::autograd::GradMode::set_enabled(true);
}

std::string toString(const torch::Tensor& tensor) {
  std::ostringstream oss;
  oss << tensor;
  // TODO: optionally remove last line of type information
  return oss.str();
}

template <typename Game>
struct Snapshot {
  torch::Tensor old_state;
  typename Game::Action action;
  typename Game::Reward reward;
  torch::Tensor new_state;
  bool over;
};

template <typename Game>
class ExperienceReplay {
 public:
  explicit ExperienceReplay(size_t buffer_size, size_t batch_size, float gamma)
      : buffer_size_(buffer_size), batch_size_(batch_size), gamma_(gamma) {}
  void addSnapshot(Snapshot<Game> snapshot) {
    snapshots_.push_back(snapshot);
    if (snapshots_.size() > buffer_size_) {
      snapshots_.erase(snapshots_.begin());
    }
    spdlog::debug("current replay size: {}", snapshots_.size());
  }

  void randomBatchBackward(torch::nn::Sequential& model, torch::nn::Sequential& target_model) {
    if (snapshots_.size() < buffer_size_) {
      return;
    }
    spdlog::debug("batch size: {}", batch_size_);

    std::vector<Snapshot<Game>> minibatch;
    std::sample(snapshots_.cbegin(), snapshots_.cend(), std::back_inserter(minibatch), batch_size_,
                std::mt19937{std::random_device{}()});

    torch::Tensor q_values_batch =
        torch::empty({batch_size_, magic_enum::enum_count<typename Game::Action>()});
    torch::Tensor q_values_target_batch =
        torch::empty({batch_size_, magic_enum::enum_count<typename Game::Action>()});

    size_t memory_idx = 0;
    for (const Snapshot<Game>& memory : minibatch) {
      torch::Tensor old_q_val = model->forward(memory.old_state);
      torch::Tensor new_q_val = target_model->forward(memory.new_state);
      float new_q_val_max = new_q_val.max().template item<float>();

      float update;
      if (memory.over) {
        update = memory.reward;
      } else {
        update = memory.reward + gamma_ * new_q_val_max;
      }
      auto old_q_val_target = old_q_val.clone();
      old_q_val_target[0][static_cast<int>(memory.action)] = update;
      q_values_batch[memory_idx] = old_q_val.squeeze();
      q_values_target_batch[memory_idx] = old_q_val_target.squeeze();
      memory_idx++;
    }
    spdlog::debug("q_values_batch = {}", toString(q_values_batch));
    spdlog::debug("q_values_target_batch = {}", toString(q_values_target_batch));
    auto loss = torch::mse_loss(q_values_batch, q_values_target_batch.detach());
    spdlog::debug("loss = {}", toString(loss));
    loss.backward();
  }

 private:
  size_t buffer_size_;
  long batch_size_;
  float gamma_;
  std::vector<Snapshot<Game>> snapshots_;
};

constexpr int kMaxSteps = 50;
constexpr float kLearningRate = 0.001f;
constexpr float kGamma = 0.9f;
constexpr int kEpochs = 5000;
constexpr int kReplayBufferSize = 1000;
constexpr int kBatchSize = 200;
constexpr int kSyncDelay = 500;
const std::string kModelFile = "dql_model.pt";

void test_model(torch::nn::Sequential& model) {
  auto game = drl_in_action::grid_world::GridWorld();
  int step_count = 0;
  auto state = flat_tensor(game.state());
  while (step_count++ < kMaxSteps && !game.over()) {
    game.display();
    std::cout << "Press any key to move with trained model...\n";
    std::cin.ignore();
    system("clear");

    auto q_values = model->forward(state);
    int action = q_values.argmax().item<int>();
    auto [reward, new_state_raw] = game.step(GridWorld::Action(action), true);
    state = flat_tensor(new_state_raw);
  }
  spdlog::info("step used: {}, status: {}", step_count, game.win() ? "win" : "loss");
}

int main(int argc, char* argv[]) {
  // TODO: train or test model via command line args
  spdlog::set_level(spdlog::level::info);

  auto model = create_model(torch::kCUDA);
  auto target_model = create_model(torch::kCUDA);
  loadParameter(target_model, model->named_parameters());
  auto optimizer = torch::optim::Adam(model->parameters(), kLearningRate);

  bool model_loaded = true;
  try {
    torch::load(model, kModelFile);
    spdlog::info("The model has been loaded from {}", kModelFile);
  } catch (...) {
    model_loaded = false;
    spdlog::info("The model could not be loaded from {}", kModelFile);
  }

  if (model_loaded) {
    test_model(model);
    return 0;
  }

  // RNG
  std::random_device rd;
  std::uniform_real_distribution<double> dist_real(0.0, 1.0);
  const auto explore_rand = [&rd, &dist_real] { return dist_real(rd); };
  std::uniform_int_distribution<int> dist_int(0, magic_enum::enum_count<GridWorld::Action>() - 1);
  const auto action_rand = [&rd, &dist_int] { return dist_int(rd); };

  ExperienceReplay<GridWorld> replay(kReplayBufferSize, kBatchSize, kGamma);

  const auto noisify_tensor = [](torch::Tensor& tensor) {
    tensor += torch::rand(tensor.sizes(), tensor.options()) / 100;
  };

  float epsilon = 1.0f;
  int sync_count = 0;
  for (size_t epoch_idx = 0; epoch_idx < kEpochs; epoch_idx++) {
    auto game = GridWorld();
    int step_count = 0;
    int total_reward = 0;
    auto state = flat_tensor(game.state());
    noisify_tensor(state);
    // Stop if the maximum steps are reached in case the game is not solvable.
    while (step_count < kMaxSteps && !game.over()) {
      // Get Q values for the current state
      auto q_values = model->forward(state);

      // Either explore with random action or pick the one with the highest Q value
      int action = 0;
      if (explore_rand() < epsilon) {
        action = action_rand();
      } else {
        action = q_values.argmax().item<int>();
      }
      auto [reward, new_state_raw] = game.step(GridWorld::Action(action));
      auto new_state = flat_tensor(new_state_raw);
      noisify_tensor(new_state);

      optimizer.zero_grad();

      // Experience replay
      replay.addSnapshot({state, GridWorld::Action(action), reward, new_state, game.over()});
      replay.randomBatchBackward(model, target_model);

      optimizer.step();

      // Update current state
      state = new_state;

      // Update the target model
      if (sync_count % kSyncDelay == 0) {
        loadParameter(target_model, model->named_parameters());
      }

      total_reward += reward;
      step_count++;
      sync_count++;
    }

    // TODO: epsilon should depend on the winrate
    if (epsilon > 0.1f) {
      epsilon -= 1.0f / kEpochs;
    }
    spdlog::info("Epoch {:3d} steps: {:3d}, total reward: {:4d}, status: {}", epoch_idx, step_count,
                 total_reward, game.win() ? "win" : "loss");
  }

  torch::save(model, kModelFile);
  return 0;
}
