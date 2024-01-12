import numpy as np
import torch as T
import torch.nn.functional as F
from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent():
    """
    The agent class that will be used to train the agent.
    """

    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, batch_size=256, reward_scale=2):
        """
        Initialize the agent.
        :param alpha: The learning rate of the actor network (default: 0.0003 from the paper)
        :param beta: The learning rate of the critic network (default: 0.0003 from the paper)
        :param input_dims: The dimensions of the input
        :param env: The environment
        :param gamma: The discount factor (default: 0.99 from the paper)
        :param n_actions: The number of actions (default: 2)
        :param max_size: The maximum size of the replay buffer (default: 1000000)
        :param tau: The target value network update rate (soft update, so slightly detune parameters) (default: 0.005)
        :param batch_size: The batch size (default: 256)
        :param reward_scale: The scale of the reward (default: 2)
        """
        self.gamma = gamma
        self.tau = tau

        # The memory of the agent
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        # The actor network
        self.actor = ActorNetwork(
            alpha, input_dims, n_actions=n_actions, name='actor', max_action=env.action_space.high)

        # The critic networks (we have two, we will take the minimum of the two)
        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name='critic_2')

        # The value and target value networks
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        # The scale of the reward
        self.scale = reward_scale

        # Set the target value network to the same parameters as the value network
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        Choose an action based on the observation.
        :param observation: The observation of the environment
        :return: The action to take
        """
        # Creating a tensor from a list of numpy.ndarrays is extremely slow, so we convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        # print(observation)
        # converted_observation = np.array([observation])
        # Convert the observation to a PyTorch tensor and send it to the device to get the state
        state = T.Tensor(observation).to(self.actor.device)
        # print("state", state)
        # Get the action from the actor network
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        # print("choose_action", actions)

        # Convert the action to a numpy array
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, episode_done):
        """
        Interface function between the agent and its memory.
        :param state: The state of the environment
        :param action: The action taken
        :param reward: The reward received
        :param new_state: The new state of the environment
        :param done: Whether the episode is done or not
        """
        self.memory.store_transition(
            state, action, reward, new_state, episode_done)

    def update_network_parameters(self, tau=None):
        """
        Update the parameters of the target value network to the parameters of the value network (with a slight detune).
        :param tau: The update rate
        """
        if tau is None:
            tau = self.tau

        # Get the parameters of the target value and value networks
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        # Convert the parameters to dictionaries
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # Update the parameters with a slight detune
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1 - tau) * target_value_state_dict[name].clone()

        # Load the new parameters into the target value network
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        """
        Save the models.
        """
        print('[ACRL] ... saving all models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        """
        Load the models.
        """
        print('[ACRL] ... loading all models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        """
        Learn from the memory.
        """
        # If the memory is not large enough, don't learn
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample our buffer: get the state, action, reward, new state, and done state from the memory
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        # Convert everything to PyTorch tensors and send them to the device
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Calculate values of the state and new state
        value = self.value(state).view(-1)
        value_ = self.value(state_).view(-1)
        value_[done] = 0.0

        # Get the actions and log probabilities from the actor network
        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # Calculate the Q values from the critic networks
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        # Take the minimum of the two to improve stability of learning (prevent overestimation bias)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Calculate the target value and handle value network loss
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Do the same as above for actor network loss
        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Calculate the actor network loss
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Handle the critic network loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # Handles inclusion of entropy and loss function to encourage exploration
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        # Take the sum of the two losses, backpropagate, and step the optimizers
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Finally, update the network parameters for the value function
        self.update_network_parameters()
