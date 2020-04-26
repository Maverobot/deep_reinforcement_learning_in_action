#pragma once

#include <torch_utils.h>

#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace deep_q_learning {

struct TORCH_API TrainOptions {
  TORCH_ARG(torch::Device, device) = torch::kCPU;
  TORCH_ARG(std::size_t, max_steps) = 50;
  TORCH_ARG(float_t, learning_rate) = 0.001f;
  TORCH_ARG(float_t, gamma) = 0.9f;
  TORCH_ARG(std::size_t, epochs) = 5000;
  TORCH_ARG(std::size_t, replay_buffer_size) = 1000;
  TORCH_ARG(std::size_t, replay_batch_size) = 200;
  TORCH_ARG(std::size_t, replay_sync_delay) = 500;
};

// Deep Q learning with CNN layers
struct DeepQCNNImpl : torch::nn::Module {
  DeepQCNNImpl(int64_t input_dim, int64_t output_dim, int64_t board_size) : board_size(board_size) {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_dim, 10, kernel_size).padding(1)));
    conv2 = register_module(
        "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, kernel_size).padding(1)));
    register_module("conv2_drop", conv2_drop);
    fc1 = register_module(
        "fc1",
        torch::nn::Linear(
            torch::nn::LinearOptions(board_size * board_size * 20, fc1_output_dim).bias(true)));
    fc2 = register_module(
        "fc2", torch::nn::Linear(torch::nn::LinearOptions(fc1_output_dim, output_dim).bias(true)));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view({1, board_state_size, board_size, board_size});
    x = torch::elu(conv1->forward(x));
    x = torch::elu(conv2_drop->forward(conv2->forward(x)));
    x = x.view({-1, board_size * board_size * 20});
    x = torch::elu(fc1->forward(x));
    x = fc2->forward(x);
    return x;
  }

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::Dropout2d conv2_drop;

  int64_t board_size;
  const int board_state_size = 4;  // player, wall, goal, pit
  const int fc1_output_dim = 50;
  const int kernel_size = 3;
};
TORCH_MODULE(DeepQCNN);

// Deep Q learning with fully connected layers
struct FullyConnectedImpl : torch::nn::Module {
  FullyConnectedImpl(int64_t input_dim, int64_t output_dim) {
    linear1 = register_module(
        "linear1", torch::nn::Linear(torch::nn::LinearOptions(input_dim, kL2).bias(true)));
    elu1 = register_module("elu1", torch::nn::ELU(torch::nn::ELUOptions().alpha(1).inplace(false)));
    linear2 = register_module("linear2",
                              torch::nn::Linear(torch::nn::LinearOptions(kL2, kL3).bias(true)));
    elu2 = register_module("elu2", torch::nn::ELU(torch::nn::ELUOptions().alpha(1).inplace(false)));
    linear3 = register_module(
        "linear3", torch::nn::Linear(torch::nn::LinearOptions(kL3, output_dim).bias(true)));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = elu1(linear1(x));
    x = elu2(linear2(x));
    x = linear3(x);
    return x;
  }

  torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
  torch::nn::ELU elu1{nullptr}, elu2{nullptr};

  const int kL2 = 150;
  const int kL3 = 100;
};
TORCH_MODULE(FullyConnected);

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
  explicit ExperienceReplay(std::size_t buffer_size, std::size_t batch_size, float gamma)
      : buffer_size_(buffer_size), batch_size_(batch_size), gamma_(gamma) {}
  void addSnapshot(Snapshot<Game> snapshot) {
    snapshots_.push_back(snapshot);
    if (snapshots_.size() > buffer_size_) {
      snapshots_.erase(snapshots_.begin());
    }
    spdlog::debug("current replay size: {}", snapshots_.size());
  }

  template <typename Model>
  void randomBatchBackward(Model& model, Model& target_model) {
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

    std::size_t memory_idx = 0;
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
    spdlog::debug("q_values_batch = {}", torch_utils::toString(q_values_batch));
    spdlog::debug("q_values_target_batch = {}", torch_utils::toString(q_values_target_batch));
    auto loss = torch::mse_loss(q_values_batch, q_values_target_batch.detach());
    spdlog::debug("loss = {}", torch_utils::toString(loss));
    loss.backward();
  }

 private:
  std::size_t buffer_size_;
  long batch_size_;
  float gamma_;
  std::vector<Snapshot<Game>> snapshots_;
};

template <typename Game, typename Model>
void testModel(Model& model, std::size_t max_steps = 50, torch::Device device = torch::kCPU) {
  model->eval();
  auto game = Game();
  std::size_t step_count = 0;
  auto state = torch_utils::flatTensor(game.state(), device);
  while (step_count++ < max_steps && !game.over()) {
    game.display();
    std::cout << "Press any key to move with trained model...\n";
    std::cin.ignore();
    system("clear");

    auto q_values = model->forward(state);
    int action = q_values.argmax().template item<int>();
    auto [reward, new_state_raw] = game.step(typename Game::Action(action), true);
    state = torch_utils::flatTensor(new_state_raw, device);
  }
  spdlog::info("step used: {}, status: {}", step_count, game.win() ? "win" : "loss");
}

template <typename Game, typename Model>
void trainModel(Model& model, const TrainOptions& options) {
  Model target_model = Model(model);
  torch_utils::loadParameter(target_model, model->named_parameters());
  auto optimizer = torch::optim::Adam(model->parameters(), options.learning_rate());

  // RNG
  std::random_device rd;
  std::uniform_real_distribution<double> dist_real(0.0, 1.0);
  const auto explore_rand = [&rd, &dist_real] { return dist_real(rd); };
  std::uniform_int_distribution<int> dist_int(0,
                                              magic_enum::enum_count<typename Game::Action>() - 1);
  const auto action_rand = [&rd, &dist_int] { return dist_int(rd); };

  deep_q_learning::ExperienceReplay<Game> replay(options.replay_buffer_size(),
                                                 options.replay_batch_size(), options.gamma());

  const auto noisify_tensor = [](torch::Tensor& tensor) {
    tensor += torch::rand(tensor.sizes(), tensor.options()) / 100;
  };

  float epsilon = 1.0f;
  int sync_count = 0;
  for (std::size_t epoch_idx = 0; epoch_idx < options.epochs(); epoch_idx++) {
    auto game = Game();
    std::size_t step_count = 0;
    int total_reward = 0;
    auto state = torch_utils::flatTensor(game.state(), options.device());
    noisify_tensor(state);
    // Stop if the maximum steps are reached in case the game is not solvable.
    while (step_count < options.max_steps() && !game.over()) {
      // Get Q values for the current state
      auto q_values = model->forward(state);

      // Either explore with random action or pick the one with the highest Q value
      int action = 0;
      if (explore_rand() < epsilon) {
        action = action_rand();
      } else {
        action = q_values.argmax().template item<int>();
      }
      auto [reward, new_state_raw] = game.step(typename Game::Action(action));
      auto new_state = torch_utils::flatTensor(new_state_raw, options.device());
      noisify_tensor(new_state);

      optimizer.zero_grad();

      // Experience replay
      replay.addSnapshot({state, typename Game::Action(action), reward, new_state, game.over()});
      replay.randomBatchBackward(model, target_model);

      optimizer.step();

      // Update current state
      state = new_state;

      // Update the target model
      if (sync_count % options.replay_sync_delay() == 0) {
        torch_utils::loadParameter(target_model, model->named_parameters());
      }

      total_reward += reward;
      step_count++;
      sync_count++;
    }

    // TODO: epsilon should depend on the winrate
    if (epsilon > 0.1f) {
      epsilon -= 1.0f / options.epochs();
    }
    spdlog::info("Epoch {:3d} steps: {:3d}, total reward: {:4d}, status: {}", epoch_idx, step_count,
                 total_reward, game.win() ? "win" : "loss");
  }
}

}  // namespace deep_q_learning
