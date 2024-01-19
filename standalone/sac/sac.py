from copy import deepcopy
import itertools
from gymnasium import Env
import numpy as np
import torch
from torch.optim import Adam
import time
import sac.core as core
from sac.utils.logx import EpisodeLogger, colorize


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        Initialize a replay buffer.
        """

        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        """
        Sample a batch of experiences from the buffer.
        """

        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class SacAgent():
    """
    Soft Actor-Critic (SAC)

    Args:
        env_fn : An OpenAI Gym environment.

        exp_name : Name for the experiment.

        load_path (str): Path to load the model from. (or None to not load)

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object.

        seed (int): Seed for random number generators.

        n_episodes (int): Number of episodes to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        save_freq (int): How often (in terms of gap between episodes) to save
            the current policy and value function.

        step_duration_limit (int): The maximum duration of a single step in the environment in ms.
    """

    def __init__(self, env: Env, exp_name, load_path=None, ac_kwargs=dict(hidden_sizes=[256]*2), seed=0, n_episodes=50,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
                 update_after=1000, update_every=50, save_freq=1, step_duration_limit=None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.save_freq = save_freq
        self.n_episodes = n_episodes
        self.step_duration_limit = step_duration_limit

        # Setup the logger
        logger = EpisodeLogger(exp_name=exp_name)
        self.logger = logger

        # Load / save
        if load_path is not None:
            # TODO: Fix this
            logger.restore_state(load_path)
        else:
            logger.save_config(locals())

        # Set the amount of threads to the amount that's available
        torch.set_num_threads(torch.get_num_threads())

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks
        if load_path is not None:
            ac = torch.load(load_path)
        else:
            ac = core.MLPActorCritic(
                env.observation_space, env.action_space, **ac_kwargs)
        self.ac = ac
        ac_targ = deepcopy(ac)
        self.ac_targ = ac_targ

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
        self.q_params = q_params

        # Experience buffer
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module)
                           for module in [ac.pi, ac.q1, ac.q2])

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(q_params, lr=lr)

        # Set up model saving
        logger.setup_pytorch_saver(ac)

        logger.log(
            '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def _compute_loss_q(self, data):
        """
        Compute the Q-losses for the Q-networks.
        :param data: The data to use for the loss computation
        """
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def _compute_loss_pi(self, data):
        """
        Compute the pi loss for the policy.
        :param data: The data to use for the loss computation
        """
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def _update(self, data):
        """
        Perform a single update of the SAC model.
        :param data: The data to use for the update
        """
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self._compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self._compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def _get_action(self, observation, deterministic=False):
        """
        Get an action from the actor-critic.
        :param observation: The observation to get an action for
        :param deterministic: Whether to use a deterministic policy (default: False)
        """
        return self.ac.act(torch.as_tensor(observation, dtype=torch.float32), deterministic)

    def train(self):
        """
        Perform SAC training on the environment.
        """
        start_time = time.time()
        total_steps = 0

        logger = self.logger
        env = self.env

        dist_highscore = 0

        # Main loop: collect experience in env and update/log each episode
        for e in range(self.n_episodes):
            print(colorize("Starting episode: " + str(e +
                  1) + "/" + str(self.n_episodes), "yellow"))
            ep_start_time = time.time()
            observation, _ = env.reset()
            ep_reward, ep_steps = 0, 0
            done = False

            # We store the array speed, x, y, z, of the car in a numpy array for every step
            drive_data = [[observation[1], observation[2],
                           observation[3], observation[4]]]

            while not done:
                step_start_time = time.time()

                # Check if the location is too close to 1.0 when the episode has just started, this means the car started before the start/finish line
                if observation[0] >= 0.9 and ep_steps == 0:
                    while observation[0] >= 0.9:
                        observation, _, _, _, _ = env.unwrapped.step([0.5, 0.0], ep_steps, ignore_done=True)
                        time.sleep(0.5)
                    env.unwrapped.controller.perform(-1.0, 0.0)
                    time.sleep(1.0)
                    env.unwrapped.controller.perform(0.0, 0.0)
                    continue

                # Uniform-random action sampling for the first start_steps steps (before running real policy)
                # This is to encourage exploration
                if total_steps > self.start_steps:
                    action = self._get_action(observation)
                else:
                    action = env.action_space.sample()

                # Step the env
                observation_, reward, terminated, truncated, _ = env.unwrapped.step(
                    action, ep_steps)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

                # Store step reward and delta progress
                logger.store(StepReward=reward)
                logger.store(DeltaProg=(observation_[0] - observation[0]))

                # Store experience to replay buffer
                self.replay_buffer.store(observation, action,
                                         reward, observation_, done)

                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                observation = observation_
                total_steps += 1

                # We store the speed, x, y, z, of the car in an array for every step
                drive_data.append(
                    [observation[1], observation[2], observation[3], observation[4]])

                if total_steps >= self.update_after and total_steps % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(
                            self.batch_size)
                        self._update(data=batch)

                if self.step_duration_limit is not None:
                    # Stall the program if the episode was too short
                    step_duration = time.time() - step_start_time
                    if step_duration < self.step_duration_limit:
                        print(colorize("Step duration not reached yet, stalling for " + str(self.step_duration_limit -
                                                                                            step_duration) + "ms...", "yellow"))
                        time.sleep(self.step_duration_limit - step_duration)

            # Drive data to numpy arrays
            speed = np.array([x[0] for x in drive_data])
            avg_speed = np.mean(speed)
            x_path = np.array([x[1] for x in drive_data])
            y_path = np.array([x[2] for x in drive_data])
            z_path = np.array([x[3] for x in drive_data])

            # Save the drive data to a JSON file
            logger.save_drive_data(e, speed, x_path, y_path, z_path)

            # Update the highscore
            if observation[0] > dist_highscore:
                dist_highscore = observation[0]

            # Save model
            if e % self.save_freq == 0 or (e == self.n_episodes-1):
                logger.save_state({'env': env}, save_env=False, itr=None)

            # Log info about episode
            logger.log_tabular('Episode', e + 1)
            logger.log_tabular('EpReward', ep_reward)
            logger.log_tabular('EpSteps', ep_steps)
            logger.log_tabular('EpDist', observation[0])
            logger.log_tabular('EpAvgSpeed', avg_speed)
            logger.log_tabular('DistHigh', dist_highscore)
            logger.log_tabular('StepReward', with_min_and_max=True)
            logger.log_tabular('DeltaProg', with_min_and_max=True)
            logger.log_tabular('TotalSteps', total_steps)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('EpTime', time.time()-ep_start_time)
            logger.log_tabular('TotalTime', time.time()-start_time)
            logger.dump_tabular()

        # Close the environment
        self.env.close()
