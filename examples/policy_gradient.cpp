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

// rewards is ordered to start from past and end with current
inline std::vector<float> discountRewards(std::vector<float> rewards, float gamma = 0.99) {
  std::for_each(rewards.begin(), rewards.end(),
                [i = 0, gamma](float& r) mutable { r = std::pow(gamma, i++) * r; });
  return rewards;
}

inline torch::Tensor lossFn(torch::Tensor preds, torch::Tensor rewards) {
  return -1 * torch::sum(rewards * torch::log(preds));
}

inline void run_single_environment(const boost::shared_ptr<Gym::Client>& client,
                                   const std::string& env_id,
                                   int episodes_to_run) {
  using Observation = std::vector<float>;
  using Action = std::vector<float>;
  using Reward = float;

  // Model
  auto model = torch::nn::Sequential(torch::nn::Linear(4, 150), torch::nn::ELU(),
                                     torch::nn::Linear(150, 2), torch::nn::Softmax(/*dim=*/1));
  const float kLearningRate = 0.001;
  const float kGamma = 0.99;
  auto optimizer = torch::optim::Adam(model->parameters(), kLearningRate);

  // Environment
  boost::shared_ptr<Gym::Environment> env = client->make(env_id);
  boost::shared_ptr<Gym::Space> action_space = env->action_space();
  boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

  // Plot
  std::vector<double> total_steps_vec;
  const int kMovingAverageWindowSize = 50;

  bool render = false;

  for (int episode_idx = 0; episode_idx < episodes_to_run; ++episode_idx) {
    Gym::State s;
    env->reset(&s);
    int total_steps = 0;

    if (episode_idx > episodes_to_run * 0.8) {
      render = true;
    }

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

      env->step(action, render, &s);

      total_steps += 1;
      if (s.done) {
        break;
      }
    }
    int64_t episode_len = observations.size();
    auto preds = torch::zeros(episode_len);
    std::partial_sum(rewards.begin(), rewards.end(), rewards.begin());
    std::reverse(rewards.begin(), rewards.end());
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

    spdlog::info("{} episode {:5d}/{:5d} finished in {:3d} steps", env_id, episode_idx,
                 episodes_to_run, total_steps);

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
  plt::show();
}

int main(int argc, char** argv) {
  try {
    boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
    run_single_environment(client, "CartPole-v0", 500);

  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 1;
  }

  return 0;
}
