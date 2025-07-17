import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import random
from collections import deque


class ChessActorCritic(nn.Module):
    """Actor-Critic network for chess."""
    
    def __init__(self, input_channels: int = 12, hidden_layers: List[int] = [512, 256, 128], 
                 num_actions: int = 4352, dropout: float = 0.1):
        super(ChessActorCritic, self).__init__()
        
        # Shared layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        conv_output_size = 256 * 4 * 4
        
        layers = []
        prev_size = conv_output_size
        for hidden_size in hidden_layers:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        self.actor = nn.Linear(prev_size, num_actions)
        self.critic = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = x.flatten(1)
        x = self.shared_layers(x)
        
        return self.actor(x), self.critic(x)
    
    def get_action_probs(self, x, legal_actions: List[int] = None):
        """Get action probabilities with masking."""
        action_logits, value = self.forward(x)
        
        if legal_actions is not None:
            # Mask illegal actions
            mask = torch.ones(action_logits.size(1), device=action_logits.device) * float('-inf')
            mask[legal_actions] = 0
            action_logits = action_logits + mask
        
        return F.softmax(action_logits, dim=-1), value


class PPOMemory:
    """Memory buffer for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.legal_actions_list = []
        self.dones = []
        
    def push(self, state, action, reward, value, action_prob, legal_actions, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.legal_actions_list.append(legal_actions)
        self.dones.append(done)
    
    def get_batch(self):
        return (np.array(self.states), np.array(self.actions), np.array(self.rewards),
                np.array(self.values), np.array(self.action_probs), self.legal_actions_list,
                np.array(self.dones))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.legal_actions_list.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO agent for chess."""
    
    def __init__(self, state_shape, num_actions=4352, hidden_layers=[512, 256, 128], 
                 learning_rate=0.0003, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, device='auto'):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        self.network = ChessActorCritic(
            input_channels=state_shape[2],
            hidden_layers=hidden_layers,
            num_actions=num_actions
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = PPOMemory()
        self.training = True
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float]:
        """Select action using current policy."""
        if not legal_actions:
            return 0, 1.0, 0.0
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network.get_action_probs(state_tensor, legal_actions)
            
            if self.training:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                action_prob = action_probs[0, action.item()].item()
            else:
                action = action_probs.argmax()
                action_prob = action_probs[0, action.item()].item()
            
            return action.item(), action_prob, value.item()
    
    def compute_gae_returns(self, rewards, values, dones):
        """Compute GAE returns."""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        next_value = 0
        next_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage
            returns[t] = advantages[t] + values[t]
            
            next_value = values[t]
            next_advantage = advantages[t]
        
        return returns, advantages
    
    def train_step(self, states, actions, returns, advantages, old_action_probs, legal_actions_list):
        """Perform one PPO training step."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_action_probs = torch.FloatTensor(old_action_probs).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_probs, values = self.network.get_action_probs(states, legal_actions_list)
        action_probs_taken = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        ratio = action_probs_taken / (old_action_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * (-entropy)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': -entropy.item()
        }
    
    def save(self, filepath: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def set_training(self, training: bool):
        self.training = training
        self.network.train(training) 