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


def get_model() -> Sequential:
    l1 = 64  # 3 x 16 for 4x4 grids
    l2 = 150
    l3 = 100
    l4 = 4
    model = Sequential(Linear(l1, l2), ELU(), Linear(l2, l3), ELU(),
                       Linear(l3, l4))
    return model


model_path = "dql_model.pt"


def main():
    plt.figure()

    plt.xlabel("step index")
    plt.ylabel('cost (log10)')

    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        model = get_model()

    loss_fn = MSELoss(size_average=True)
    optimizer = Adam(model.parameters(), lr=0.001)

    gamma = 0.9  # decay factor for future reward
    epsilon = 1.0  # probability to take random action

    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    epochs = 10
    step_idx = 0
    for epoch_idx in range(epochs):
        game = Gridworld(size=4, mode='static')
        s_ = game.board.render_np().reshape(1, 64)
        state = Variable(torch.from_numpy(s_).float())
        #print(game.display())

        game_over = False
        while not game_over:
            q_val = model(state)

            # Random action or best action
            if random.random() < epsilon:
                action_ = random.randint(0, len(action_set) - 1)
            else:
                action_ = np.argmax(q_val.data.numpy())
            action = action_set[action_]
            #print(action)

            # Make the action
            game.makeMove(action)
            s_ = game.board.render_np().reshape(1, 64)
            #print(game.display())

            # Get the new state and reward
            new_state = Variable(torch.from_numpy(s_).float())
            reward = game.reward()

            # Get the maximum Q value of the new new_state
            new_q_val = model(new_state)
            new_q_val_max = np.max(new_q_val.data.numpy())

            # Set target q value
            y_ = np.copy(q_val.data.numpy())
            if reward == -1:
                update = reward + (gamma * new_q_val_max)
            else:
                update = reward
            y_[0][action_] = update
            y = Variable(torch.from_numpy(y_).float())

            # Calculate loss
            loss = loss_fn(q_val, y)

            # Visualize loss trend
            plt.scatter(step_idx, np.log10(loss.item()))
            plt.pause(0.01)
            step_idx += 1
            #print("Epoch {}: {}".format(epoch_idx, loss))

            # Clear gradients of all optimized tensors
            optimizer.zero_grad()

            # Compute gradient to minimize loss
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            state = new_state
            if reward != -1:
                game_over = True
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs
    torch.save(model, model_path)
    plt.show(block=True)


if __name__ == "__main__":
    main()
    pass
