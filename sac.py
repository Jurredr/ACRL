# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np

# # Define the actor network
# class Actor(nn.Module):
#   def __init__(self, state_dim, action_dim):
#     super(Actor, self).__init__()
#     self.fc1 = nn.Linear(state_dim, 256)
#     self.fc2 = nn.Linear(256, 256)
#     self.fc3 = nn.Linear(256, action_dim)

#   def forward(self, state):
#     x = F.relu(self.fc1(state))
#     x = F.relu(self.fc2(x))
#     action = torch.tanh(self.fc3(x))
#     return action

# # Define the critic network
# class Critic(nn.Module):
#   def __init__(self, state_dim, action_dim):
#     super(Critic, self).__init__()
#     self.fc1 = nn.Linear(state_dim + action_dim, 256)
#     self.fc2 = nn.Linear(256, 256)
#     self.fc3 = nn.Linear(256, 1)

#   def forward(self, state, action):
#     x = torch.cat([state, action], dim=1)
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     value = self.fc3(x)
#     return value

# # Define the SAC agent
# class SACAgent:
#   def __init__(self, state_dim, action_dim):
#     self.actor = Actor(state_dim, action_dim)
#     self.critic1 = Critic(state_dim, action_dim)
#     self.critic2 = Critic(state_dim, action_dim)
#     self.target_critic1 = Critic(state_dim, action_dim)
#     self.target_critic2 = Critic(state_dim, action_dim)
#     self.target_critic1.load_state_dict(self.critic1.state_dict())
#     self.target_critic2.load_state_dict(self.critic2.state_dict())
#     self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
#     self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
#     self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)
#     self.gamma = 0.99
#     self.tau = 0.005

#   def select_action(self, state):
#     state = torch.FloatTensor(state).unsqueeze(0)
#     action = self.actor(state).detach().numpy()[0]
#     return action

#   def update(self, state, action, next_state, reward, done):
#     state = torch.FloatTensor(state)
#     action = torch.FloatTensor(action)
#     next_state = torch.FloatTensor(next_state)
#     reward = torch.FloatTensor(reward)
#     done = torch.FloatTensor(done)

#     # Update critic networks
#     target_value = reward + self.gamma * (1 - done) * torch.min(
#       self.target_critic1(next_state, self.actor(next_state)),
#       self.target_critic2(next_state, self.actor(next_state))
#     )
#     q1 = self.critic1(state, action)
#     q2 = self.critic2(state, action)
#     critic1_loss = F.mse_loss(q1, target_value.detach())
#     critic2_loss = F.mse_loss(q2, target_value.detach())
#     self.critic1_optimizer.zero_grad()
#     critic1_loss.backward()
#     self.critic1_optimizer.step()
#     self.critic2_optimizer.zero_grad()
#     critic2_loss.backward()
#     self.critic2_optimizer.step()

#     # Update actor network
#     actor_loss = -self.critic1(state, self.actor(state)).mean()
#     self.actor_optimizer.zero_grad()
#     actor_loss.backward()
#     self.actor_optimizer.step()

#     # Update target critic networks
#     for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
#       target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#     for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
#       target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# # Usage example
# state_dim = 4
# action_dim = 2
# agent = SACAgent(state_dim, action_dim)

# # Training loop
# for episode in range(num_episodes):
#   state = env.reset()
#   done = False
#   total_reward = 0

#   while not done:
#     action = agent.select_action(state)
#     next_state, reward, done, _ = env.step(action)
#     agent.update(state, action, next_state, reward, done)
#     state = next_state
#     total_reward += reward

#   print(f"Episode: {episode}, Total Reward: {total_reward}")
