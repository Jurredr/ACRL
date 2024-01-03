import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        """
        The critic network.
        :param beta: The learning rate.
        :param input_dims: The input dimensions.
        :param n_actions: The number of actions.
        :param fc1_dims: The number of neurons in the first fully connected layer.  (default: 256 as per the paper)
        :param fc2_dims: The number of neurons in the second fully connected layer. (default: 256 as per the paper)
        :param name: The name of the network.
        :param chkpt_dir: The directory to save the network to. (default: tmp/sac)
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # The first fully connected layer (deep neural network).
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        # The second fully connected layer (deep neural network).
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # The output layer (deep neural network).
        self.q = nn.Linear(self.fc2_dims, 1)

        # The optimizer; optimizes the network with the Adam optimizer and learning rate beta as per the paper.
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # The device to run the network on (uses the GPU if available).
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Moves the network to the device.
        self.to(self.device)

    def forward(self, state, action):
        """
        The forward pass of the network;
        How we feed a state action pair through the network.
        :param state: The state of the environment.
        :param action: The action to take.
        :return: The Q value.
        """
        # Concatenates the state and action.
        action_value = self.fc1(T.cat([state, action], dim=1))
        # Activate the first fully connected layer.
        action_value = F.relu(action_value)
        # Pass the output of the first fully connected layer to the second fully connected layer.
        action_value = self.fc2(action_value)
        # Activate the second fully connected layer.
        action_value = F.relu(self.fc2(action_value))

        # Get the Q value from the output layer.
        # The Q value is the output of the network; the state action pair.
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        """
        Saves the network to a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the network from a checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        """
        The value network.
        :param beta: The learning rate.
        :param input_dims: The input dimensions.
        :param fc1_dims: The number of neurons in the first fully connected layer.  (default: 256 as per the paper)
        :param fc2_dims: The number of neurons in the second fully connected layer. (default: 256 as per the paper)
        :param name: The name of the network.
        :param chkpt_dir: The directory to save the network to. (default: tmp/sac)
        """
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # The first fully connected layer (deep neural network).
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # The second fully connected layer (deep neural network).
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # The output layer (deep neural network).
        self.v = nn.Linear(self.fc2_dims, 1)

        # The optimizer; optimizes the network with the Adam optimizer and learning rate beta as per the paper.
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # The device to run the network on (uses the GPU if available).
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Moves the network to the device.
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        # Activate the first fully connected layer.
        state_value = F.relu(state_value)
        # Pass the output of the first fully connected layer to the second fully connected layer.
        state_value = self.fc2(state_value)
        # Activate the second fully connected layer.
        state_value = F.relu(self.fc2(state_value))

        # Get the value from the output layer.
        # The value is the output of the network; the state.
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        """
        Saves the network to a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the network from a checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        """
        The actor network.
        :param alpha: The learning rate.
        :param input_dims: The input dimensions.
        :param max_action: The maximum action.
        :param fc1_dims: The number of neurons in the first fully connected layer.  (default: 256 as per the paper)
        :param fc2_dims: The number of neurons in the second fully connected layer. (default: 256 as per the paper)
        :param n_actions: The number of actions. (default: 2)
        :param name: The name of the network.
        :param chkpt_dir: The directory to save the network to. (default: tmp/sac)
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        # The reparameterization noise to ensure we don't get 0 gradients.
        self.reparam_noise = 1e-6

        # The first fully connected layer (deep neural network).
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # The second fully connected layer (deep neural network).
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Output 1: the mean for the Gaussian distribution of the policy.
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # Output 2: the log standard deviation for the Gaussian distribution of the policy.
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        # The optimizer; optimizes the network with the Adam optimizer and learning rate alpha as per the paper.
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # The device to run the network on (uses the GPU if available).
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Moves the network to the device.
        self.to(self.device)

    def forward(self, state):
        """
        The forward pass of the network;
        How we feed a state through the network.
        :param state: The state of the environment.
        :return: The mean and log standard deviation of the Gaussian distribution of the policy.
        """
        prob = self.fc1(state)
        # Activate the first fully connected layer.
        prob = F.relu(prob)
        # Pass the output of the first fully connected layer to the second fully connected layer.
        prob = self.fc2(prob)
        # Activate the second fully connected layer.
        prob = F.relu(prob)

        # Get the mean and log standard deviation from the output layer.
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # The log standard deviation must be positive.
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Samples from the Gaussian distribution of the policy.
        The policy is a Gaussian distribution that gives the probability of taking an action.
        We need a Gaussian distribution because we have a continuous action space.
        :param state: The state of the environment.
        :param reparameterize: Whether to reparameterize the network. (default: True)
        :return: The action to take.
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # Re-parameterization trick.
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # The action must be in the range of the maximum action.
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        # The log probability of the action for the loss function for updating the weights of the neural network.
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        """
        Saves the network to a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the network from a checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
