#!/usr/bin/env python
import random

import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class MultiArmBandit:
    def __init__(self, n):
        self.probs = [random.random() for _ in range(n)]
        print(self.probs)

    def step(self, action):
        reward = 1 if random.random() < self.probs[action] else 0
        return reward

    def actions(self):
        return [i for i in range(len(self.probs))]


class Agent:
    def __init__(
            self,
            actions,
    ):
        self.actions = actions
        self.reward_history = [[] for _ in range(len(self.actions))]

    def step(self, env, strategy='greedy'):
        if strategy == 'greedy':
            action = self.greedy_selection(0.9)
        elif strategy == 'softmax':
            action = self.softmax_selection()
        else:
            raise Exception("strategy {} is not allowed.".format(strategy))

        print("action: ", action)

        reward = env.step(action)
        self.reward_history[action].append(reward)

    def greedy_selection(self, espilon=0.9):
        if random.random() < espilon:
            action = np.argmax(self.action_reward_average())
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    def softmax_selection(self):
        probs = softmax(self.action_reward_average())
        probs_cs = np.cumsum(probs)
        r = random.random()
        for action in range(len(probs_cs)):
            if r <= probs_cs[action]:
                return action
        raise Exception("This shall not happen.")

    def action_reward_average(self):
        return [sum(h) / len(h) if h else 0 for h in self.reward_history]

    def reward_average(self):
        reward_sum = sum([sum(h) for h in self.reward_history])
        reward_num = sum([len(h) for h in self.reward_history])
        return reward_sum / reward_num


if __name__ == '__main__':
    env = MultiArmBandit(10)
    agent = Agent(env.actions())
    plt.figure()

    generations = 500

    for i in range(generations):
        agent.step(env, 'softmax')
        plt.scatter(i, agent.reward_average())
        plt.pause(0.001)

    for i in range(generations):
        agent.step(env, 'greedy')
        plt.scatter(i + generations, agent.reward_average())
        plt.pause(0.001)

    plt.show()
