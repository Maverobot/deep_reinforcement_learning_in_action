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

auto model(torch::Device device) {
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
auto flat_tensor(std::vector<std::vector<value_type>> input) {
  std::cout << input << "\n";
  std::vector<torch::Tensor> output_vec;
  std::transform(input.begin(), input.end(), std::back_inserter(output_vec), [](auto& arr) {
    return torch::from_blob(arr.data(), {1, static_cast<long>(arr.size())},
                            torch::TensorOptions().dtype<value_type>())
        .clone();
  });
  return torch::cat(output_vec, 1);
}

using drl_in_action::grid_world::GridWorld;

int main(int argc, char* argv[]) {
  auto game = drl_in_action::grid_world::GridWorld();
  auto state = flat_tensor(game.state());
  std::cout << state << "\n";
  game.display();
  return 0;
}
