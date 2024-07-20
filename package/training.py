# type: ignore

from collections import deque
from typing import Deque
from params import env
from agent import agent
from typing import Any
import numpy as np
import torch


def Training():
    number_episodes = 2000
    maximum_number_timesteps_per_episodes = 1000
    epsilon_starting_value = 1.0
    epsilon_ending_value = 0.1
    epsilon_decay_value = 0.995
    epsilon = epsilon_starting_value
    scores_on_100_episodes: Deque[float] = deque(maxlen=100)

    for episode in range(1, number_episodes + 1):
        state, _ = env.reset()
        score: Any = 0

        for t in range(maximum_number_timesteps_per_episodes):
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        mean: Any = np.mean(scores_on_100_episodes)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(episode, mean),
            end="",
        )

        if episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(episode, mean))

        if np.mean(scores_on_100_episodes) >= 200.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    episode - 100, mean
                )
            )
            torch.save(agent.local_qnetwork.state_dict(), "checkpoint.pth")
            break

Training()