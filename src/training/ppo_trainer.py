import numpy as np
import torch
from tqdm import tqdm
import os
import time
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Fix the import to use absolute imports
from src.environment.chess_env import ChessEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.config import load_config


class PPOTrainer:
    """
    PPO trainer for chess agents.
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
            reward_material_factor=env_config['reward_material_factor'],
            reward_win=env_config.get('reward_win', 1.0),
            reward_draw=env_config.get('reward_draw', 0.0),
            reward_loss=env_config.get('reward_loss', -1.0)
        )
        
    def setup_agent(self):
        """Setup the PPO agent."""
        agent_config = self.config['agent']
        network_config = self.config['network']
        training_config = self.config['training']
        
        self.agent = PPOAgent(
            state_shape=self.env.observation_space.shape,
            num_actions=self.env.action_space.n,
            hidden_layers=network_config['hidden_layers'],
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            gae_lambda=agent_config['gae_lambda'],
            clip_ratio=agent_config['clip_ratio'],
            value_coef=agent_config['value_coef'],
            entropy_coef=agent_config['entropy_coef'],
            max_grad_norm=agent_config['max_grad_norm'],
            entropy_annealing=agent_config.get('entropy_annealing', False),
            entropy_annealing_start=agent_config.get('entropy_annealing_start', 0.1),
            entropy_annealing_end=agent_config.get('entropy_annealing_end', 0.01),
            entropy_annealing_steps=agent_config.get('entropy_annealing_steps', 5000)
        )
        
        # Training parameters
        self.steps_per_update = training_config['steps_per_update']
        self.num_episodes = training_config['num_episodes']
        self.eval_freq = training_config['eval_freq']
        self.save_freq = training_config['save_freq']
        self.ppo_epochs = training_config['ppo_epochs']
        
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
    
    def collect_trajectory(self) -> Tuple[float, int]:
        """Collect a trajectory for both white and black, ensuring both see terminal reward. Clean, robust version."""
        state, info = self.env.reset()
        legal_actions = info['legal_actions']
        # 0: white, 1: black
        trajs = [
            {k: [] for k in ['states', 'actions', 'action_probs', 'rewards', 'values', 'dones', 'legal_actions']},
            {k: [] for k in ['states', 'actions', 'action_probs', 'rewards', 'values', 'dones', 'legal_actions']}
        ]
        turn = 0
        done = False
        
        # DEBUG: Check network health at the start of trajectory collection
        if hasattr(self, '_health_check_count'):
            self._health_check_count += 1
        else:
            self._health_check_count = 0
            
        if self._health_check_count % 50 == 0:  # Check every 50 trajectories
            health_report = self.agent.check_network_health(state)
            print(f"\n=== NETWORK HEALTH CHECK (trajectory {self._health_check_count}) ===")
            print(f"Logits: min={health_report['logits']['min']:.6f}, max={health_report['logits']['max']:.6f}, mean={health_report['logits']['mean']:.6f}")
            print(f"Probabilities: min={health_report['probabilities']['min']:.6f}, max={health_report['probabilities']['max']:.6f}, sum={health_report['probabilities']['sum']:.6f}")
            print(f"Value: {health_report['value']:.6f}")
            if health_report['logits']['all_negative']:
                print("⚠️  WARNING: All logits are negative!")
            if health_report['probabilities']['all_zero']:
                print("⚠️  WARNING: All probabilities are zero!")
            print("=" * 50)
        
        while not done:
            # Store current state for current player
            traj = trajs[turn]
            traj['states'].append(state)
            traj['legal_actions'].append(legal_actions)
            # Select action
            action, action_prob, value = self.agent.select_action(state, legal_actions)
            
            # Assert action is legal before stepping
            traj['actions'].append(action)
            traj['action_probs'].append(action_prob)
            traj['values'].append(value)
            # Step environment
            next_state, reward, done, truncated, info = self.env.step(action)
            traj['rewards'].append(reward)
            traj['dones'].append(done)
            if not done:
                state = next_state
                legal_actions = info['legal_actions']
                turn = 1 - turn
        # At this point, the game is over. The loser is the player who did not just move.
        loser = 1 - turn
        # Only add dummy terminal state if loser made at least one move
        if len(trajs[loser]['states']) > 0:
            # The loser's final state is the state after the winner's move (next_state)
            # The reward is the terminal reward for losing
            if self.env.board.is_checkmate():
                terminal_reward = self.env.reward_loss
            elif self.env.board.is_stalemate() or self.env.board.is_insufficient_material() or self.env.move_count >= self.env.max_moves:
                terminal_reward = self.env.reward_draw
            else:
                terminal_reward = 0.0
            for k, v in zip(
                ['states', 'actions', 'action_probs', 'values', 'legal_actions', 'rewards', 'dones'],
                [next_state, 0, 0.0, 0.0, [], terminal_reward, True]
            ):
                trajs[loser][k].append(v)
        # Compute returns/advantages, then remove dummy state for loser
        full_returns = []
        combined = []
        for color in [0, 1]:
            traj = trajs[color]
            if not traj['rewards']:
                continue
            returns, advantages = self.agent.compute_gae_returns(
                np.array(traj['rewards']), np.array(traj['values']), np.array(traj['dones'])
            )
            full_returns.append(traj['rewards'][:])
            # Remove dummy terminal state for loser
            if color == loser and len(traj['rewards']) > 1:
                for k in traj:
                    traj[k].pop()
                
                returns = returns[:-1]
                advantages = advantages[:-1]
                
            for i in range(len(traj['states'])):
                combined.append({
                    'state': traj['states'][i],
                    'action': traj['actions'][i],
                    'reward': traj['rewards'][i],
                    'value': traj['values'][i],
                    'action_prob': traj['action_probs'][i],
                    'legal_actions': traj['legal_actions'][i],
                    'done': traj['dones'][i],
                    'return': returns[i],
                    'advantage': advantages[i],
                })
        # Push to memory
        for t in combined:
            self.agent.memory.push(
                t['state'], t['action'], t['reward'], t['value'], t['action_prob'], t['legal_actions'], t['done'], t['return'], t['advantage']
            )
        # Return white's total reward and length for logging
        return sum(full_returns[0]), len(full_returns[0]), sum(full_returns[1]), len(full_returns[1])
    
    def update_policy(self) -> List[Dict[str, float]]:
        """
        Update policy using collected trajectory.
        Returns: List of loss dictionaries for each PPO epoch
        """
        # Get trajectory data
        states, actions, rewards, values, action_probs, legal_actions_list, dones, returns, advantages = \
            self.agent.memory.get_batch()
        # Train for multiple epochs
        epoch_losses = []
        for _ in range(self.ppo_epochs):
            loss_dict = self.agent.train_step(
                states, actions, returns, advantages, action_probs, legal_actions_list
            )
            epoch_losses.append(loss_dict)
        # Clear memory
        self.agent.memory.clear()
        return epoch_losses
    
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
        
        for game in range(num_games):
            state, info = self.env.reset()
            legal_actions = info['legal_actions']
            game_moves = 0
            
            while True:
                action, _, _ = self.agent.select_action(state, legal_actions)
                state, reward, done, truncated, info = self.env.step(action)
                legal_actions = info['legal_actions']
                game_moves += 1
                
                if done or truncated:
                    break
            
            total_moves += game_moves
            
            # ADD THESE PRINT STATEMENTS HERE
            print(f"Game {game + 1} ended after {game_moves} moves:")
            if self.env.board.is_checkmate():
                print("  -> CHECKMATE!")
                if self.env.board.turn:  # White's turn but checkmated
                    print("  -> Black wins!")
                    losses += 1
                else:
                    print("  -> White wins!")
                    wins += 1
            elif self.env.board.is_stalemate():
                print("  -> STALEMATE!")
                draws += 1
            elif self.env.board.is_insufficient_material():
                print("  -> INSUFFICIENT MATERIAL!")
                draws += 1
            elif self.env.move_count >= self.env.max_moves:
                print("  -> MOVE LIMIT REACHED!")
                draws += 1
            else:
                print("  -> OTHER TERMINATION!")
                draws += 1
            
            # Print final board state
            print(f"  Final FEN: {self.env.board.fen()}")
            print("  Final board:")
            print(self.env.board)
            print("-" * 50)
        
        self.agent.set_training(True)
        
        return {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games,
            'avg_moves': total_moves / num_games
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting PPO training for {self.num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        
        start_time = time.time()
        total_steps = 0
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Collect trajectory
            white_reward, white_length, black_reward, black_length = self.collect_trajectory()
            total_steps += white_length
            
            # ADD THIS DEBUG PRINT - Print every episode for now
            print(f"Episode {episode}: white_reward={white_reward:.2f}, black_reward={black_reward:.2f}, white_length={white_length}, black_length={black_length}")
            
            # Check what happened in the last game
            if self.env.board.is_checkmate():
                print(f"  -> Last game ended in CHECKMATE!")
            elif self.env.board.is_stalemate():
                print(f"  -> Last game ended in STALEMATE!")
            elif self.env.board.is_insufficient_material():
                print(f"  -> Last game ended in INSUFFICIENT MATERIAL!")
            elif self.env.move_count >= self.env.max_moves:
                print(f"  -> Last game ended by MOVE LIMIT!")
            else:
                print(f"  -> Last game ended for OTHER REASON!")
            
            # Print the final board state
            print(f"  Final FEN: {self.env.board.fen()}")
            print("-" * 50)
            
            # Store metrics
            self.episode_rewards.append(white_reward)
            self.episode_lengths.append(white_length)
            
            # Update policy if enough steps collected
            if total_steps >= self.steps_per_update:
                epoch_losses = self.update_policy()
                
                # Store average losses
                avg_losses = {}
                for key in epoch_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
                
                self.losses.append(avg_losses)
                total_steps = 0
                
                # Log to tensorboard
                if self.use_tensorboard:
                    for key, value in avg_losses.items():
                        self.writer.add_scalar(f'Training/{key}', value, episode)
                    
                    # Log entropy coefficient for monitoring exploration
                    if 'entropy_coef' in avg_losses:
                        self.writer.add_scalar('Training/Entropy_Coefficient', avg_losses['entropy_coef'], episode)
                    if 'entropy' in avg_losses:
                        self.writer.add_scalar('Training/Entropy', avg_losses['entropy'], episode)
                
                # Print entropy info for monitoring
                if 'entropy_coef' in avg_losses and 'entropy' in avg_losses:
                    print(f"  Entropy: {avg_losses['entropy']:.4f}, Entropy Coef: {avg_losses['entropy_coef']:.4f}")
            
            # Log episode metrics
            if self.use_tensorboard:
                self.writer.add_scalar('Training/Episode_Reward', white_reward, episode)
                self.writer.add_scalar('Training/Episode_Length', white_length, episode)
            
            # Evaluation - ADD DEBUG HERE TOO
            if episode % self.eval_freq == 0:
                print(f"\n=== EVALUATION AT EPISODE {episode} ===")
                eval_results = self.evaluate_agent()
                self.eval_results.append(eval_results)
                
                print(f"Evaluation Results:")
                print(f"  Win Rate: {eval_results['win_rate']:.3f}")
                print(f"  Draw Rate: {eval_results['draw_rate']:.3f}")
                print(f"  Loss Rate: {eval_results['loss_rate']:.3f}")
                print(f"  Avg Moves: {eval_results['avg_moves']:.1f}")
                print("=" * 50)
                
                # Log to tensorboard
                if self.use_tensorboard:
                    for key, value in eval_results.items():
                        self.writer.add_scalar(f'Evaluation/{key}', value, episode)
            
            # Save model
            if episode % self.save_freq == 0:
                model_path = os.path.join(self.config['logging']['model_dir'], 
                                         f'chess_ppo_episode_{episode}.pth')
                self.agent.save(model_path)
                print(f"Model saved to {model_path}")
        
        # Final evaluation
        print(f"\n=== FINAL EVALUATION ===")
        final_eval = self.evaluate_agent()
        print(f"Final Results:")
        print(f"  Win Rate: {final_eval['win_rate']:.3f}")
        print(f"  Draw Rate: {final_eval['draw_rate']:.3f}")
        print(f"  Loss Rate: {final_eval['loss_rate']:.3f}")
        print(f"  Avg Moves: {final_eval['avg_moves']:.1f}")
        
        # Save final model
        final_model_path = os.path.join(self.config['logging']['model_dir'], 'chess_ppo_final.pth')
        self.agent.save(final_model_path)
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
            policy_losses = [loss['policy_loss'] for loss in self.losses]
            value_losses = [loss['value_loss'] for loss in self.losses]
            
            axes[1, 0].plot(policy_losses, label='Policy Loss')
            axes[1, 0].plot(value_losses, label='Value Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
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