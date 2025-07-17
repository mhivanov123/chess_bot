import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional
import copy


class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation.
    Takes 8x8x12 input (piece positions) and outputs Q-values for all actions.
    """
    
    def __init__(self, input_channels: int = 12, hidden_layers: List[int] = [512, 256, 128], 
                 num_actions: int = 4352, dropout: float = 0.1):
        super(ChessNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size after conv layers
        conv_output_size = 256 * 4 * 4
        
        # Fully connected layers
        layers = []
        prev_size = conv_output_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_actions))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: (batch_size, 8, 8, 12)
        # Convert to (batch_size, 12, 8, 8) for conv layers
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pooling
        x = self.pool(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        x = x.flatten(1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, legal_actions: List[int]):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done, legal_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, legal_actions = zip(*batch)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), legal_actions)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for chess.
    """
    
    def __init__(self, state_shape: Tuple[int, int, int], num_actions: int,
                 hidden_layers: List[int] = [512, 256, 128], learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, device: str = 'auto'):
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = ChessNet(
            input_channels=state_shape[2],
            hidden_layers=hidden_layers,
            num_actions=num_actions
        ).to(self.device)
        
        self.target_network = ChessNet(
            input_channels=state_shape[2],
            hidden_layers=hidden_layers,
            num_actions=num_actions
        ).to(self.device)
        
        # Copy weights from Q-network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training state
        self.training = True
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        Select action using epsilon-greedy policy.
        Only considers legal actions.
        """
        # Safety check for empty legal actions
        if not legal_actions:
            print("Warning: No legal actions available!")
            return 0  # Return first action as fallback
        
        if self.training and random.random() < self.epsilon:
            # Random action from legal moves
            return random.choice(legal_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Mask illegal actions with large negative values
                mask = torch.ones(self.num_actions, device=self.device) * float('-inf')
                mask[legal_actions] = 0
                q_values = q_values + mask
                
                return q_values.argmax().item()
    
    def train_step(self, batch: Tuple) -> float:
        """
        Perform one training step.
        Returns the loss value.
        """
        states, actions, rewards, next_states, dones, legal_actions_list = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (with target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            
            # Mask illegal actions
            for i, legal_actions in enumerate(legal_actions_list):
                mask = torch.ones(self.num_actions, device=self.device) * float('-inf')
                mask[legal_actions] = 0
                next_q_values[i] = next_q_values[i] + mask
            
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save the model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.q_network.train(training)
        self.target_network.train(training) 