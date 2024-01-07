import numpy as np


class ReplayBuffer():
    """
    The replay buffer class, used to store the experiences of the agent.
    """

    def __init__(self, max_size, input_shape, n_actions):
        """
        Initialize the replay buffer.
        :param max_size: The maximum size of the buffer.
        :param input_shape: The shape of the input.
        :param n_actions: The number of actions.
        """
        # The maximum size of the buffer
        self.mem_size = max_size
        # The memory counter; keeps track of the current index in the buffer
        self.mem_cntr = 0
        # The memory; stores the experiences
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        # The next state memory; stores the next states after an action is taken
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        # The action memory; stores the actions taken
        self.action_memory = np.zeros((self.mem_size, n_actions))
        # The reward memory; stores the rewards received
        self.reward_memory = np.zeros(self.mem_size)
        # The terminal memory; stores whether the episode ended
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, episode_done):
        """
        Stores a transition in the replay buffer.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param state_: The next state.
        :param episode_done: Whether the episode is done.
        """
        # Get the index to store the transition
        index = self.mem_cntr % self.mem_size

        # Store the transition
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = episode_done

        # Increment the memory counter
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a batch from the replay buffer.
        In simple terms, this function returns a random batch of experiences.
        :param batch_size: The size of the batch to sample.
        :return: A batch of states, actions, rewards, next states, and terminal flags.
        """
        # Get the maximum memory size
        max_mem = min(self.mem_cntr, self.mem_size)

        # Get the random batch indices
        batch = np.random.choice(max_mem, batch_size)

        # Sample the memories; Get the states, actions, rewards, next states, and terminal flags
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        # Return the batch
        return states, actions, rewards, states_, dones
