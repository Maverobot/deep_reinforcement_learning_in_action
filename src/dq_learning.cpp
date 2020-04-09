#include <torch/torch.h>

#include <grid_world.h>

#include <iterator>

/*
  test_game = Gridworld(mode=mode)
  state_ = test_game.board.render_np().reshape(
                                               1, 64) + np.random.rand(1, 64) / 10.0
  state = Variable(torch.from_numpy(state_).float())
  print("Initial State:")
  print(test_game.display())
  gameover = False
  while not gameover:
  qval = model(state)
    qval_ = qval.data.numpy()
    action_ = np.argmax(qval_)  #take action with highest Q-value
    action = action_set[action_]
    print('Move #: %s; Taking action: %s' % (i, action))
    test_game.makeMove(action)
    state_ = test_game.board.render_np().reshape(1, 64)
    state = Variable(torch.from_numpy(state_).float())
    print(test_game.display())
    reward = test_game.reward()
    print(reward)
    if reward != -1:
    gameover = True
      print("Reward: %s" % (reward, ))
      i += 1
      if (i > 15):
        print("Game lost; too many move_count.")
          break
*/

template <class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& arr) {
  copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

using drl_in_action::grid_world::GridWorld;
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

int main(int argc, char* argv[]) {
  auto game = GridWorld();
  game.display();
  auto state = game.state();
  for (const auto& sub_state : state) {
    std::cout << sub_state << "\n";
  }
  return 0;
}
