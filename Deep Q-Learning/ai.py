# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the ANN architecture
class NeuralNet (nn.Module):
    def __init__(self, input_size, action):
        """Initialize the Neural Network

        Args:
            input_size (Integer): Size of input Vector
            action (Integer): Number of possible actions
        """
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.total_actions = action
        self.full_connection1 = nn.Linear(self.input_size, 64)
        self.full_connection2 = nn.Linear(64, 32)
        self.full_connection3 = nn.Linear(32, self.total_actions)
    
    def forward(self, state):
        """Forward propagation function

        Args:
            state (Array): Array representing one state. I.e, the input vector

        Returns:
            Array: Q-values after the forward propagation
        """
        x = F.relu(self.full_connection1(state))
        x = F.relu(self.full_connection2(x))
        q_values = self.full_connection3(x)
        return q_values       

# Implementing Experience Replay
class ReplayMemory (object):
    def __init__(self, capacity):
        """Initialize Memory for Experience Replay

        Args:
            capacity (Integer): Number of state in the past that are store in memory
        """
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        """Push an event into the memory and also check for memory capacity

        Args:
            event (Array): A vector defining state, action and reward
        """
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """Select a fixed size sample from the memory at random to enhance learning

        Args:
            batch_size (Integer): The size of required sample from the memory

        Returns:
            List: List of sample from the memory
        """
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, dim=0)), samples)

class Dqn:
    def __init__(self, dimensions, action, gamma, mem_capacity, temperature):
        """Initializing the AI

        Args:
            dimensions (Integer): Number of dimensions in the input vertor / state
            action (Integer): Number of possible actions for the AI to take
            gamma (Float): Discounting / Decay factor
            mem_capacity (Integer): Capacity of replay memory
            temperature (Float): The temperature parameter for the surity of final action by agent
        """
        self.gamma = gamma
        self.reward_window = []
        self.model = NeuralNet(dimensions, action)
        self.memory = ReplayMemory(mem_capacity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(dimensions).unsqueeze(dim=0)
        self.last_reward = 0
        self.last_action = 0
        self.temperature = temperature

    def select_action(self, input_state):
        """Funtion to select the right action for the car

        Args:
            input_state (Vector): The input state for the Neural Network

        Returns:
            Integer: Action for the car ranging from 0 to 2
        """
        with torch.no_grad():
            probabilities = F.softmax(self.model(Variable(input_state)) * self.temperature, dim=1)
            action = probabilities.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """Function that is used for learning by the agent

        Args:
            batch_state (Tensor): Current batch of states
            batch_next_state (Tensor): Next batch of states
            batch_reward (Vector): Reward for this batch
            batch_action (Vector): Actions performed in this batch
        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        targets = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, signal):
        """Update the transitions and reward window of the class at every new state.

        Args:
            reward (Float): The reward from the new state of the car
            signal (Tensor): The new state of the car

        Returns:
            Integer: Predicted action for the car
        """
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        """Calculate the mean of the reward window

        Returns:
            Float: The mean reward
        """
        return sum(self.reward_window) / (len(self.reward_window) + 0.001)

    def save(self):
        """Save the brain of AI
        """
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, 'last_saved_brain.pth')

    def load(self):
        """Load the last saved brain of AI
        """
        if os.path.isfile('last_saved_brain.pth'):
            print('=> Last saved brain file found.')
            print('=> Loading the brain')
            checkpoint = torch.load('last_saved_brain.pth')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> Last brain loaded')
        else:
            print('=> Last saved brain not found')