#include <torch/torch.h>

#include <grid_world.h>
#include <magic_enum.hpp>

#include <cstdio>
#include <iostream>
#include <iterator>
#include <random>
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
  torch::nn::Sequential model(
      torch::nn::Linear(torch::nn::LinearOptions(kL1, kL2).bias(true)),
      torch::nn::Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1),
      torch::nn::Linear(torch::nn::LinearOptions(kL2, kL3).bias(true)),
      torch::nn::Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1),
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

constexpr int kMaxSteps = 10;
constexpr float kLearningRate = 0.001f;
constexpr float kGamma = 0.9f;
constexpr int kEpochs = 100;
const std::string kModelFile = "dql_model.pt";

void test_model(torch::nn::Sequential& model) {
  auto game = drl_in_action::grid_world::GridWorld();
  game.display();
  int step_count = 0;
  auto state = flat_tensor(game.state());
  while (step_count++ < kMaxSteps && !game.over()) {
    // Get Q values for the current state
    auto q_values = model->forward(state);
    int action = q_values.argmax().item<int>();
    auto [reward, new_state_raw] = game.step(GridWorld::Action(action), true);
    state = flat_tensor(new_state_raw);
    game.display();
  }
  printf("step used: %d, status: %s\n", step_count, game.win() ? "win" : "loss");
}

int main(int argc, char* argv[]) {
  auto model = create_model(torch::kCUDA);
  auto optimizer = torch::optim::Adam(model->parameters(), kLearningRate);

  bool model_loaded = true;
  try {
    torch::load(model, kModelFile);
    std::cout << "model loaded\n";
  } catch (...) {
    model_loaded = false;
    std::cout << "There is no model to load\n";
  }

  if (model_loaded) {
    std::cout << model << "\n";
    test_model(model);
    return 0;
  }

  // RNG
  std::random_device rd;
  std::uniform_real_distribution<double> dist_real(0.0, 1.0);
  const auto explore_rand = [&rd, &dist_real] { return dist_real(rd); };
  std::uniform_int_distribution<int> dist_int(0, magic_enum::enum_count<GridWorld::Action>() - 1);
  const auto action_rand = [&rd, &dist_int] { return dist_int(rd); };

  float epsilon = 1.0f;
  for (size_t epoch_idx = 0; epoch_idx < kEpochs; epoch_idx++) {
    auto game = drl_in_action::grid_world::GridWorld();
    int step_count = 0;
    int total_reward = 0;
    auto state = flat_tensor(game.state());
    while (!game.over()) {
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

      // Set target Q value
      torch::Tensor q_values_target = q_values.clone();
      float update;
      if (!game.over()) {
        update = reward + (kGamma * model->forward(new_state).max().item<float>());
      } else {
        update = reward;
      }
      q_values_target[0][action] = update;

      // Loss function
      auto loss = torch::mse_loss(q_values, q_values_target.detach());

      // Step
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      // Update current state
      state = new_state;

      total_reward += reward;
      step_count++;
    }

    // TODO: epsilon should depend on the winrate
    if (epsilon > 0.1f) {
      epsilon -= 1.0f / kEpochs;
    }

    printf("Epoch %lu steps: %d, total reward: %d, status: %s\n", epoch_idx, step_count,
           total_reward, game.win() ? "win" : "loss");
  }

  torch::save(model, kModelFile);
  return 0;
}
