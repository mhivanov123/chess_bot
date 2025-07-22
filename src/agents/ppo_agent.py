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
    
    def __init__(self, input_channels: int = 13, hidden_layers: List[int] = [512, 256, 128], 
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
        
        # Initialize weights properly to prevent very negative logits
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights to prevent very negative logits."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                # Use Kaiming initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for the actor head to ensure reasonable initial logits
        # Initialize with small positive values to avoid very negative logits
        nn.init.xavier_uniform_(self.actor.weight, gain=0.01)  # Small gain to keep logits reasonable
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)
        
        # Initialize critic head normally
        nn.init.xavier_uniform_(self.critic.weight)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)
        
    def forward(self, x):
        # Ensure x is a torch tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # If x is 1D, add batch dimension
        if x.dim() == 1:
            # Could be flat or (8,8,13)
            if x.shape[0] == 8*8*13:
                x = x.view(1, 8, 8, 13)
            elif x.shape == (8, 8, 13):
                x = x.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected 1D input shape: {x.shape}")
        # If x is 2D, could be (batch, 832) or (batch, 8, 8, 13) flattened
        elif x.dim() == 2:
            if x.shape[1] == 8*8*13:
                x = x.view(-1, 8, 8, 13)
            else:
                raise ValueError(f"Unexpected 2D input shape: {x.shape}")
        # If x is 3D, could be (8,8,13)
        elif x.dim() == 3:
            if x.shape == (8, 8, 13):
                x = x.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected 3D input shape: {x.shape}")
        # Now x should be (batch, 8, 8, 13)
        if x.dim() != 4 or x.shape[1:] != (8, 8, 13):
            raise ValueError(f"Input to forward() must be (batch, 8, 8, 13), got {x.shape}")
        # Permute to (batch, 13, 8, 8)
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
        """Get action probabilities without masking."""
        action_logits, value = self.forward(x)
        
        # No masking - let the agent learn what moves are legal
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value
    
    def get_action_probs_batch(self, x, legal_actions_list: List[List[int]] = None):
        """Get action probabilities for a batch with individual masking."""
        action_logits, value = self.forward(x)
        
        if legal_actions_list is not None:
            # Apply masking for each sample in the batch
            for i, legal_actions in enumerate(legal_actions_list):
                if legal_actions:  # Only mask if there are legal actions
                    mask = torch.ones(action_logits.size(1), device=action_logits.device) * float('-inf')
                    # Ensure legal actions are within bounds
                    valid_actions = [a for a in legal_actions if 0 <= a < action_logits.size(1)]
                    if valid_actions:
                        mask[valid_actions] = 0
                    action_logits[i] = action_logits[i] + mask
                else:
                    # If no valid actions, set all to equal probability
                    action_logits[i] = torch.zeros_like(action_logits[i])
        
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
        self.returns = []  # NEW
        self.advantages = []  # NEW
    
    def push(self, state, action, reward, value, action_prob, legal_actions, done, ret=None, adv=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.legal_actions_list.append(legal_actions)
        self.dones.append(done)
        self.returns.append(ret)
        self.advantages.append(adv)
    
    def get_batch(self):
        return (np.array(self.states), np.array(self.actions), np.array(self.rewards),
                np.array(self.values), np.array(self.action_probs), self.legal_actions_list,
                np.array(self.dones), np.array(self.returns), np.array(self.advantages))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.legal_actions_list.clear()
        self.dones.clear()
        self.returns.clear()
        self.advantages.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO agent for chess."""
    
    def __init__(self, state_shape, num_actions=4352, hidden_layers=[512, 256, 128], 
                 learning_rate=0.0003, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.1, max_grad_norm=0.5, device='auto',
                 entropy_annealing=False, entropy_annealing_start=0.1, 
                 entropy_annealing_end=0.01, entropy_annealing_steps=5000):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions
        
        # Entropy annealing parameters
        self.entropy_annealing = entropy_annealing
        self.entropy_annealing_start = entropy_annealing_start
        self.entropy_annealing_end = entropy_annealing_end
        self.entropy_annealing_steps = entropy_annealing_steps
        self.total_steps = 0
        
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
        """Select action using current policy, with action masking (only legal actions can be selected)."""
        if not legal_actions:
            # No legal actions: return dummy action (should only happen in terminal states)
            return 0, 0.0, 0.0
        
        # Update epsilon if annealing is enabled
        self.update_epsilon()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Use action masking: only legal actions are considered
            action_probs, value = self.network.get_action_probs(state_tensor, legal_actions)
            action_logits, _ = self.network.forward(state_tensor)
            logits_for_legal = action_logits[0, legal_actions]
            # Numerically stable: clamp logits before softmax
            logits_for_legal = torch.clamp(logits_for_legal, min=-20, max=20)
            masked_probs = torch.softmax(logits_for_legal, dim=-1)
            # Clamp probabilities before log and normalization
            masked_probs = torch.clamp(masked_probs, min=1e-8, max=1.0)
            masked_probs = masked_probs / masked_probs.sum()  # Renormalize
            
            # Epsilon-greedy exploration
            if self.training and np.random.random() < self.epsilon:
                action_idx = np.random.randint(len(legal_actions))
                action_prob = 1.0 / len(legal_actions)
            else:
                if self.training:
                    action_dist = torch.distributions.Categorical(masked_probs)
                    action_idx = action_dist.sample().item()
                else:
                    action_idx = masked_probs.argmax().item()
                action_prob = masked_probs[action_idx].item()
            selected_action = legal_actions[action_idx]
            return selected_action, action_prob, value.item()
    
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
    
    def update_entropy_coef(self):
        """Update entropy coefficient based on annealing schedule."""
        if self.entropy_annealing and self.total_steps < self.entropy_annealing_steps:
            # Linear annealing
            progress = self.total_steps / self.entropy_annealing_steps
            self.entropy_coef = self.entropy_annealing_start + progress * (self.entropy_annealing_end - self.entropy_annealing_start)
            
            # Ensure minimum entropy coefficient to prevent complete determinism
            min_entropy_coef = 0.001
            self.entropy_coef = max(self.entropy_coef, min_entropy_coef)
    
    def train_step(self, states, actions, old_action_probs, returns, advantages, legal_actions_list=None):
        """Perform one PPO training step."""
        assert states is not None and (not hasattr(states, 'shape') or states.shape[0] > 0), \
            "train_step received empty batch! Debug your trajectory collection."
        
        # Update entropy coefficient if annealing is enabled
        self.update_entropy_coef()
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_action_probs = torch.FloatTensor(old_action_probs).to(self.device)
        
        # Numerically stable advantage normalization
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        # Use the batch version of get_action_probs
        action_probs, values = self.network.get_action_probs_batch(states, legal_actions_list)
        # Clamp probabilities before log and normalization
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        action_probs_taken = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Numerically stable ratio calculation
        ratio = action_probs_taken / (old_action_probs + 1e-8)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values.squeeze(), returns)
        # Numerically stable entropy calculation
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * (-entropy)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Increment total steps for entropy annealing
        self.total_steps += 1
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': -entropy.item(),
            'entropy_coef': self.entropy_coef,
            'entropy': entropy.item()
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
    
    def check_network_health(self, state: np.ndarray) -> Dict[str, Any]:
        """Check network health by analyzing outputs for a given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.network.forward(state_tensor)
            
            # Analyze logits
            logits_stats = {
                'min': action_logits.min().item(),
                'max': action_logits.max().item(),
                'mean': action_logits.mean().item(),
                'std': action_logits.std().item(),
                'has_nan': torch.isnan(action_logits).any().item(),
                'has_inf': torch.isinf(action_logits).any().item(),
                'all_negative': (action_logits < 0).all().item(),
                'all_zero': (action_logits == 0).all().item(),
            }
            
            # Analyze probabilities after softmax
            action_probs = F.softmax(action_logits, dim=-1)
            probs_stats = {
                'min': action_probs.min().item(),
                'max': action_probs.max().item(),
                'mean': action_probs.mean().item(),
                'std': action_probs.std().item(),
                'sum': action_probs.sum().item(),
                'has_nan': torch.isnan(action_probs).any().item(),
                'all_zero': (action_probs == 0).all().item(),
            }
            
            # Analyze network parameters
            param_stats = {}
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    param_stats[name] = {
                        'mean': param.mean().item(),
                        'std': param.std().item(),
                        'min': param.min().item(),
                        'max': param.max().item(),
                        'has_nan': torch.isnan(param).any().item(),
                        'has_inf': torch.isinf(param).any().item(),
                    }
            
            return {
                'logits': logits_stats,
                'probabilities': probs_stats,
                'value': value.item(),
                'parameters': param_stats
            } 