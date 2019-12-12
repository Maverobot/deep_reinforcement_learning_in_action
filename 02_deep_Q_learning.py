#!/usr/bin/env python
"""
    1. Setup a for-loop to number of epochs
    2. In the loop, setup while loop (while game is in progress)
    3. Run Q network forward.
    4. We're using an epsilon greedy implementation, so at time t with probability ϵ we will choose a random action. With probability 1−ϵ we will choose the action associated with the highest Q value from our neural network.
    5. Take action a as determined in (4), observe new state s' and reward r t+1
    6. Run the network forward using s′. Store the highest Q value, max Q.
    7. Our target value to train the network is Rt+1+γ∗maxQA (St+1 ) where γ (gamma) is a parameter between 0 and 1.
    8. Given that we have 4 outputs and we only want to update/train the output associated with the action we just took, our target output vector is the same as the output vector from the first run, except we change the one output associated with our action to the result we compute using the Q-learning formula.
    9. Train the model on this 1 sample. Repeat process 2-9
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from Gridworld import *

Sequential = torch.nn.Sequential
Linear = torch.nn.Linear
ELU = torch.nn.ELU
MSELoss = torch.nn.MSELoss
Adam = torch.optim.Adam

# For reproducibility
#torch.manual_seed(0)
#random.seed(0)

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
model_path = "dql_model.pt"


def get_model() -> Sequential:
    l1 = 64  # 3 x 16 for 4x4 grids
    l2 = 150
    l3 = 100
    l4 = 4
    model = Sequential(Linear(l1, l2), ELU(), Linear(l2, l3), ELU(),
                       Linear(l3, l4))
    return model


def test_model(model, mode='static'):
    i = 0
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


def train_model(model, epochs=10, mode='static'):
    loss_fn = MSELoss(size_average=True)
    optimizer = Adam(model.parameters(), lr=0.001)

    gamma = 0.9  # decay factor for future reward
    epsilon = 1.0  # probability to take random action
    max_moves = 50  # maximum move_count that one epoch is allowed to have
    replay = []
    replay_buffer = 500
    batch_size = 100

    step_idx = 0
    for epoch_idx in range(epochs):
        game = Gridworld(size=4, mode=mode)
        s_ = game.board.render_np().reshape(1, 64)
        state = Variable(torch.from_numpy(s_).float())

        move_count = 0
        game_over = False
        total_reward = 0
        while not game_over:
            move_count += 1

            q_val = model(state)

            # Random action or best action
            if random.random() < epsilon:
                action_ = random.randint(0, len(action_set) - 1)
            else:
                action_ = np.argmax(q_val.data.numpy())
            action = action_set[action_]

            # Make the action
            game.makeMove(action)
            s_ = game.board.render_np().reshape(1, 64)

            # Get the new state and reward
            new_state = Variable(torch.from_numpy(s_).float())
            reward = game.reward()

            total_reward += reward

            if len(replay) < replay_buffer:
                replay.append((state, action_, reward, new_state))
            else:
                replay.pop(0)
                replay.append((state, action_, reward, new_state))

                minibatch = random.sample(replay, batch_size)
                X_train = Variable(
                    torch.empty(batch_size, 4, dtype=torch.float))
                y_train = Variable(
                    torch.empty(batch_size, 4, dtype=torch.float))

                memory_idx = 0
                for memory in minibatch:
                    (old_state_m, action_m, reward_m, new_state_m) = memory
                    old_q_val = model(old_state_m)
                    new_q_val = model(new_state_m)
                    new_q_val_max = np.max(new_q_val.data.numpy())
                    # Don't forget deepcopy
                    y = copy.deepcopy(old_q_val.data.numpy())
                    if reward == -1:
                        update = reward + (gamma * new_q_val_max)
                    else:
                        update = reward
                    y[0][action_m] = update
                    X_train[memory_idx] = old_q_val
                    y_train[memory_idx] = Variable(torch.from_numpy(y).float())
                    memory_idx += 1

                # Calculate loss
                loss = loss_fn(X_train, y_train)

                # Visualize loss trend
                plt.scatter(step_idx, np.log10(loss.item()))
                plt.pause(0.01)
                step_idx += 1
                # print("Epoch {}: {}".format(epoch_idx, loss))

                # Clear gradients of all optimized tensors
                optimizer.zero_grad()

                # Compute gradient to minimize loss
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

                state = new_state
            if reward != -1 or move_count > max_moves:
                game_over = True
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs
        print("Epoch {} total reward: {}".format(epoch_idx, total_reward))
    return model


def main():
    plt.figure()

    plt.xlabel("step index")
    plt.ylabel('cost (log10)')

    try:
        model = torch.load(model_path)
        test_model(model, mode='random')
    except FileNotFoundError:
        model = get_model()
        model = train_model(model, epochs=3000, mode='random')
        torch.save(model, model_path)
        plt.show(block=True)


if __name__ == "__main__":
    main()
