import numpy as np
import torch
from tqdm import tqdm
import os
import time
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from ..environment.chess_env import ChessEnv
from ..agents.dqn_agent import DQNAgent, ReplayBuffer
from ..utils.config import load_config


class ChessTrainer:
    """
    Main trainer class for training chess agents.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.eval_results = []
        
    def setup_environment(self):
        """Setup the chess environment."""
        env_config = self.config['environment']
        self.env = ChessEnv(
            max_moves=env_config['max_moves'],
            reward_material_factor=env_config['reward_material_factor']
        )
        
    def setup_agent(self):
        """Setup the DQN agent."""
        agent_config = self.config['agent']
        network_config = self.config['network']
        training_config = self.config['training']
        
        self.agent = DQNAgent(
            state_shape=self.env.observation_space.shape,
            num_actions=self.env.action_space.n,
            hidden_layers=network_config['hidden_layers'],
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            epsilon=agent_config['epsilon_start'],
            epsilon_min=agent_config['epsilon_end'],
            epsilon_decay=agent_config['epsilon_decay']
        )
        
        # Setup replay buffer
        self.replay_buffer = ReplayBuffer(training_config['buffer_size'])
        
        # Training parameters
        self.batch_size = training_config['batch_size']
        self.min_buffer_size = training_config['min_buffer_size']
        self.update_freq = training_config['update_freq']
        self.target_update_freq = agent_config['target_update_freq']
        self.num_episodes = training_config['num_episodes']
        self.eval_freq = training_config['eval_freq']
        self.save_freq = training_config['save_freq']
        
    def setup_logging(self):
        """Setup logging directories and tensorboard."""
        log_config = self.config['logging']
        
        # Create directories
        os.makedirs(log_config['log_dir'], exist_ok=True)
        os.makedirs(log_config['model_dir'], exist_ok=True)
        
        # Setup tensorboard if enabled
        if log_config['tensorboard']:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_config['log_dir'])
                self.use_tensorboard = True
            except ImportError:
                print("Tensorboard not available, logging disabled")
                self.use_tensorboard = False
        else:
            self.use_tensorboard = False
    
    def train_episode(self) -> Tuple[float, int, List[float]]:
        """
        Train for one episode.
        
        Returns:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in the episode
            episode_losses: List of losses during the episode
        """
        state, info = self.env.reset()
        legal_actions = info['legal_actions']
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        while True:
            # Select action
            action = self.agent.select_action(state, legal_actions)
            
            # Take action
            next_state, reward, done, truncated, info = self.env.step(action)
            next_legal_actions = info['legal_actions']
            
            # Store transition
            self.replay_buffer.push(
                state, action, reward, next_state, done, legal_actions
            )
            
            # Train if enough samples
            if len(self.replay_buffer) >= self.min_buffer_size and episode_length % self.update_freq == 0:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.agent.train_step(batch)
                episode_losses.append(loss)
                self.losses.append(loss)
                
                # Update target network
                if len(self.losses) % self.target_update_freq == 0:
                    self.agent.update_target_network()
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            legal_actions = next_legal_actions
            
            if done or truncated:
                break
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        return episode_reward, episode_length, episode_losses
    
    def evaluate_agent(self, num_games: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent by playing games against itself.
        
        Args:
            num_games: Number of games to play for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.agent.set_training(False)
        
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        for _ in range(num_games):
            state, info = self.env.reset()
            legal_actions = info['legal_actions']
            game_moves = 0
            
            while True:
                action = self.agent.select_action(state, legal_actions)
                state, reward, done, truncated, info = self.env.step(action)
                legal_actions = info['legal_actions']
                game_moves += 1
                
                if done or truncated:
                    break
            
            total_moves += game_moves
            
            # Determine game outcome
            if self.env.board.is_checkmate():
                if self.env.board.turn:  # White's turn but checkmated
                    losses += 1  # Black won
                else:
                    wins += 1    # White won
            else:
                draws += 1  # Draw
        
        self.agent.set_training(True)
        
        return {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games,
            'avg_moves': total_moves / num_games
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        
        start_time = time.time()
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Train one episode
            episode_reward, episode_length, episode_losses = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log to tensorboard
            if self.use_tensorboard:
                self.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
                self.writer.add_scalar('Training/Episode_Length', episode_length, episode)
                self.writer.add_scalar('Training/Epsilon', self.agent.epsilon, episode)
                
                if episode_losses:
                    avg_loss = np.mean(episode_losses)
                    self.writer.add_scalar('Training/Average_Loss', avg_loss, episode)
            
            # Evaluation
            if episode % self.eval_freq == 0:
                eval_results = self.evaluate_agent()
                self.eval_results.append(eval_results)
                
                print(f"\nEpisode {episode}:")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Length: {episode_length}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Win Rate: {eval_results['win_rate']:.2f}")
                print(f"  Draw Rate: {eval_results['draw_rate']:.2f}")
                print(f"  Loss Rate: {eval_results['loss_rate']:.2f}")
                
                if self.use_tensorboard:
                    self.writer.add_scalar('Evaluation/Win_Rate', eval_results['win_rate'], episode)
                    self.writer.add_scalar('Evaluation/Draw_Rate', eval_results['draw_rate'], episode)
                    self.writer.add_scalar('Evaluation/Loss_Rate', eval_results['loss_rate'], episode)
                    self.writer.add_scalar('Evaluation/Avg_Moves', eval_results['avg_moves'], episode)
            
            # Save model
            if episode % self.save_freq == 0:
                model_path = os.path.join(
                    self.config['logging']['model_dir'],
                    f'chess_agent_episode_{episode}.pth'
                )
                self.agent.save(model_path)
                print(f"Model saved to {model_path}")
        
        # Final save
        final_model_path = os.path.join(
            self.config['logging']['model_dir'],
            'chess_agent_final.pth'
        )
        self.agent.save(final_model_path)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final model saved to {final_model_path}")
        
        if self.use_tensorboard:
            self.writer.close()
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        
        # Losses
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Evaluation metrics
        if self.eval_results:
            episodes = [i * self.eval_freq for i in range(len(self.eval_results))]
            win_rates = [r['win_rate'] for r in self.eval_results]
            draw_rates = [r['draw_rate'] for r in self.eval_results]
            loss_rates = [r['loss_rate'] for r in self.eval_results]
            
            axes[1, 1].plot(episodes, win_rates, label='Win Rate')
            axes[1, 1].plot(episodes, draw_rates, label='Draw Rate')
            axes[1, 1].plot(episodes, loss_rates, label='Loss Rate')
            axes[1, 1].set_title('Evaluation Metrics')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 