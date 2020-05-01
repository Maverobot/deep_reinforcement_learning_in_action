#include <random>

#include <gym.h>
#include <matplotlibcpp.h>
#include <torch_utils.h>

#include <spdlog/spdlog.h>
#include <torch/torch.h>

namespace plt = matplotlibcpp;

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& vec) {
  copy(vec.cbegin(), vec.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

template <typename T>
std::vector<T> movingAverage(const std::vector<T>& input, size_t window_size) {
  std::vector<T> output;
  output.reserve(input.size());

  constexpr auto getAvg = [](auto iterator_begin, auto iterator_end, auto iterator_current,
                             size_t window_size) {
    auto start = iterator_current - window_size;
    if (start < iterator_begin) {
      start = iterator_begin;
    }
    auto end = iterator_current + window_size;
    if (end > iterator_end) {
      end = iterator_end;
    }
    return std::accumulate(start, end, 0) / (end - start);
  };

  for (auto it = input.begin(); it != input.end(); it++) {
    output.push_back(getAvg(input.begin(), input.end(), it, window_size));
  }
  return output;
}

std::vector<float> pickAction(torch::Tensor pred) {
  std::random_device rd;
  std::uniform_real_distribution<float> dist_real(0.0f, 1.0f);
  const auto rand = [&rd, &dist_real] { return dist_real(rd); };

  float r = rand();
  float* ptr = (float*)pred.data_ptr();
  float sum = 0;
  for (int i = 0; i < pred.numel(); ++i) {
    sum += *ptr++;
    if (r < sum) {
      return {static_cast<float>(i)};
    }
  }
  throw std::runtime_error("The sum of the pred is not 1");
}

// rewards is ordered to start from step 0 and end with last step
inline std::vector<float> discountRewards(std::vector<float> rewards, float gamma = 0.99) {
  std::partial_sum(
      rewards.rbegin(), rewards.rend(), rewards.rbegin(),
      [i = 1, gamma](auto& v1, auto& v2) mutable { return v1 + std::pow(gamma, i++) * v2; });

  // TODO: this normalization seems to have negative effect on the performance
  float r_max = *std::max_element(rewards.cbegin(), rewards.cend());
  std::for_each(rewards.begin(), rewards.end(), [r_max](auto& r) { r = r / r_max; });
  return rewards;
}

inline torch::Tensor lossFn(torch::Tensor preds, torch::Tensor rewards) {
  return -1 * torch::sum(rewards * torch::log(preds));
}

template <typename Model>
inline void test_single_environment(boost::shared_ptr<Gym::Environment> env, Model& model) {
  spdlog::info("Please press any key to test the trained model");
  std::cin.ignore();
  model->eval();
  Gym::State s;
  env->reset(&s);
  for (;;) {
    auto pred = model->forward(torch_utils::toTensor(s.observation));
    std::vector<float> action = pickAction(pred);
    env->step(action, true, &s);
    if (s.done) {
      break;
    }
  }
}

template <typename Model>
inline void train_single_environment(boost::shared_ptr<Gym::Environment> env,
                                     Model& model,
                                     int episodes_to_run) {
  using Observation = std::vector<float>;
  using Action = std::vector<float>;
  using Reward = float;

  const float kLearningRate = 0.001;
  const float kGamma = 0.99;
  auto optimizer = torch::optim::Adam(model->parameters(), kLearningRate);

  // Environment
  boost::shared_ptr<Gym::Space> action_space = env->action_space();
  boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

  // Plot
  std::vector<double> total_steps_vec;
  const int kMovingAverageWindowSize = 50;

  for (int episode_idx = 0; episode_idx < episodes_to_run; ++episode_idx) {
    Gym::State s;
    env->reset(&s);
    int total_steps = 0;

    // Episode
    std::vector<Observation> observations;
    std::vector<Action> actions;
    std::vector<Reward> rewards;
    for (;;) {
      auto pred = model->forward(torch_utils::toTensor(s.observation));
      std::vector<float> action = pickAction(pred);

      observations.push_back(s.observation);
      actions.push_back(action);
      rewards.push_back(s.reward);

      env->step(action, false, &s);

      total_steps += 1;
      if (s.done) {
        break;
      }
    }
    int64_t episode_len = observations.size();
    auto preds = torch::zeros(episode_len);
    auto d_rewards = discountRewards(rewards, kGamma);
    for (int64_t idx = 0; idx < episode_len; idx++) {
      auto obs = observations.at(idx);
      int action = actions.at(idx).at(0);
      auto pred = model->forward(torch_utils::toTensor(obs));
      preds[idx] = pred[0][action];
    }

    auto loss = lossFn(preds, torch_utils::toTensor(d_rewards));
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    spdlog::info("Episode {:4d}/{:4d} finished in {:3d} steps", episode_idx, episodes_to_run,
                 total_steps);

    total_steps_vec.push_back(total_steps);

    if (episode_idx % 50 == 0) {
      std::vector<double> x(total_steps_vec.size());
      std::iota(x.begin(), x.end(), 0);

      plt::clf();

      plt::named_plot("total steps", x, movingAverage(total_steps_vec, kMovingAverageWindowSize));
      plt::xlim(0, episodes_to_run);
      plt::legend();

      plt::pause(0.001);
    }
  }
}

const std::string kModelFile = "policy_gradient.pt";

// Only works with discret actions
int main(int argc, char** argv) {
  boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
  boost::shared_ptr<Gym::Environment> env = client->make("CartPole-v1");

  // Model
  auto model = torch::nn::Sequential(
      torch::nn::Linear(env->observation_space()->sample().size(), 150), torch::nn::ELU(),
      torch::nn::Linear(150, env->action_space()->discreet_n), torch::nn::Softmax(/*dim=*/1));

  bool model_loaded = true;
  try {
    torch::load(model, kModelFile);
    spdlog::info("The model has been loaded from {}", kModelFile);
  } catch (...) {
    model_loaded = false;
    spdlog::info("The model could not be loaded from {}", kModelFile);
  }

  if (model_loaded) {
    test_single_environment(env, model);
    return 0;
  }

  train_single_environment(env, model, 1000);
  torch::save(model, kModelFile);

  plt::show();
  return 0;
}
