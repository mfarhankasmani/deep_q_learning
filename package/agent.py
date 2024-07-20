# type: ignore

import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import params as args
from network import Network
from replayMemory import ReplayMemory


class Agent:
    def __init__(self, state_size: int, action_size: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        # create the instance of network class and move it to the device
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=args.learning_rate
        )
        self.memory = ReplayMemory(args.replay_buffer_size)
        self.time_step = 0

    def step(
        self,
        state,
        action,
        reward,
        next_step,
        done,
    ):
        self.memory.push((state, action, reward, next_step, done))
        # resetting after every 4 steps
        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0:
            # learn every 100 experiences
            if len(self.memory.memory) > args.minbatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, args.discount_factor)

    # 0. makes type as float
    def act(self, state, epsilon=0.0):
        # add extra dimention to state corresponding to batch
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(
        self,
        experiences,
        discount_factor: float,
    ):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = (
            self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        )
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(
            self.local_qnetwork, self.target_qnetwork, args.interpolation_param
        )

    def soft_update(
        self, local_model: Network, target_model: Network, interpolation_param: float
    ):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                interpolation_param * local_param.data
                + (1.0 - interpolation_param) * target_param
            )


agent = Agent(args.state_size, args.number_action)
