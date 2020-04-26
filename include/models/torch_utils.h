#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <string>
#include <vector>

namespace torch_utils {

template <typename value_type>
torch::Tensor toTensor(std::vector<value_type> input, torch::Device device = torch::kCPU) {
  std::vector<torch::Tensor> output_vec;
  return torch::from_blob(input.data(), {1, static_cast<long>(input.size())},
                          torch::TensorOptions().dtype<value_type>())
      .clone()
      .to(device);
}

template <typename value_type>
torch::Tensor flatTensor(std::vector<std::vector<value_type>> input,
                         torch::Device device = torch::kCPU) {
  std::vector<torch::Tensor> output_vec;
  std::transform(input.begin(), input.end(), std::back_inserter(output_vec), [&device](auto& arr) {
    return torch::from_blob(arr.data(), {1, static_cast<long>(arr.size())},
                            torch::TensorOptions().dtype<value_type>())
        .clone()
        .to(device);
  });
  return torch::cat(output_vec, 1);
}

template <typename Model>
void loadParameter(Model& model, const torch::OrderedDict<std::string, torch::Tensor>& new_params) {
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
}  // namespace torch_utils
