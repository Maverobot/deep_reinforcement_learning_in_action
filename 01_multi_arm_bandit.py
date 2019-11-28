#!/usr/bin/env python
import random
import time

import matplotlib.pyplot as plt
import numpy as np


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
    def __init__(self, actions):
        self.actions = actions
        self.reward_history = [[] for _ in range(len(self.actions))]

    def step(self, env, greedy_espilon=0.9):
        # Choose an action by accident
        if random.random() < greedy_espilon:
            action = np.argmax(self.action_reward_average())
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        reward = env.step(action)
        self.reward_history[action].append(reward)

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

    generations = 250

    for i in range(generations):
        agent.step(env, 0.8)
        plt.scatter(i, agent.reward_average())
        plt.pause(0.001)

    for i in range(generations):
        agent.step(env, 1)
        plt.scatter(i + generations, agent.reward_average())
        plt.pause(0.001)
    plt.show()
