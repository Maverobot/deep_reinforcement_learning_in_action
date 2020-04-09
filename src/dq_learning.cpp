#include <torch/torch.h>

#include <grid_world.h>

#include <iostream>
#include <iterator>
#include <vector>

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

using drl_in_action::grid_world::GridWorld;

int main(int argc, char* argv[]) {
  auto model = create_model(torch::kCUDA);
  auto game = drl_in_action::grid_world::GridWorld();
  game.display();

  bool game_over = false;
  int step_cout = 0;
  while (!game_over && step_cout++ < 50) {
    auto state = flat_tensor(game.state());
    auto q_values = model->forward(state);
    auto action = q_values.argmax().item<int>();
    game.step(GridWorld::Action(action), true);
    std::cout << "step: " << step_cout << "\n";
    game.display();
  }
  return 0;
}
