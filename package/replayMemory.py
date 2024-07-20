import torch
import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity):
        # check for gpu or run using cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        # Stacking state in to device - e[0] state is the first element
        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        # action is the second element e[1]
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None]))
            .long()
            .to(self.device)
        )
        # rewards is the third element in the experience
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        # uint8 is use for boolean values
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return states, next_states, actions, rewards, dones
