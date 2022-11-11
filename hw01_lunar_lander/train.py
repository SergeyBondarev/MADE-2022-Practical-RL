from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TAU = 1e-3
SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.qnetwork_local = copy.deepcopy(self.model).to(device)
        self.qnetwork_target = copy.deepcopy(self.model).to(device)

        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=100_000)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return list(zip(*random.sample(self.replay_buffer, BATCH_SIZE)))
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        states, actions, next_states, rewards, dones = batch

        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().to(device).view(-1, 1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).view(-1)
        dones = torch.from_numpy(np.array(dones)).float().to(device)

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions.reshape(-1, 1))
        loss = F.mse_loss(q_expected, q_targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_values = self.qnetwork_local(state).numpy()
        return np.argmax(action_values)

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.qnetwork_local, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")

    env.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()

        best_reward = -np.inf
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            if np.mean(rewards) > best_reward:
                dqn.save()
                best_reward = np.mean(rewards)
