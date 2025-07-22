#!/usr/bin/env python3
"""
Comprehensive diagnostic script for the PPO network issue.
This script will help identify the root cause of all-zero probabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.environment.chess_env import ChessEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.config import load_config
import matplotlib.pyplot as plt

def analyze_network_behavior():
    """Analyze the network behavior comprehensively."""
    
    # Setup environment and agent
    env = ChessEnv()
    state, info = env.reset()
    
    # Create agent with default config
    agent = PPOAgent(
        state_shape=(8, 8, 13),
        num_actions=4352,
        hidden_layers=[512, 256, 128],
        learning_rate=0.0003
    )
    
    print("=== NETWORK DIAGNOSTIC ANALYSIS ===")
    print(f"Device: {agent.device}")
    print(f"State shape: {state.shape}")
    print(f"Legal actions: {len(info['legal_actions'])}")
    print(f"Legal actions sample: {info['legal_actions'][:10]}")
    
    # Test 1: Check initial network outputs
    print("\n--- Test 1: Initial Network Outputs ---")
    health_report = agent.check_network_health(state)
    
    print(f"Logits stats:")
    for key, value in health_report['logits'].items():
        print(f"  {key}: {value}")
    
    print(f"\nProbabilities stats:")
    for key, value in health_report['probabilities'].items():
        print(f"  {key}: {value}")
    
    print(f"\nValue: {health_report['value']}")
    
    # Test 2: Check if the issue is systematic
    print("\n--- Test 2: Systematic Analysis ---")
    test_states = []
    
    # Generate a few different board positions
    for i in range(5):
        env.reset()
        # Make some random moves to get different positions
        for _ in range(i * 3):
            legal_actions = env._get_legal_actions()
            if legal_actions:
                action = np.random.choice(legal_actions)
                state, _, done, _, info = env.step(action)
                if done:
                    break
        test_states.append(state)
    
    all_logits = []
    all_probs = []
    
    for i, test_state in enumerate(test_states):
        health = agent.check_network_health(test_state)
        all_logits.append(health['logits'])
        all_probs.append(health['probabilities'])
        print(f"Position {i+1}: logits_min={health['logits']['min']:.6f}, probs_max={health['probabilities']['max']:.6f}")
    
    # Test 3: Check parameter statistics
    print("\n--- Test 3: Parameter Analysis ---")
    param_stats = health_report['parameters']
    
    print("Network parameter statistics:")
    for name, stats in param_stats.items():
        print(f"  {name}:")
        for key, value in stats.items():
            print(f"    {key}: {value:.6f}")
    
    # Test 4: Check if the issue is in the forward pass
    print("\n--- Test 4: Forward Pass Analysis ---")
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        # Check intermediate activations
        x = state_tensor.permute(0, 3, 1, 2)  # (batch, 13, 8, 8)
        
        # Conv layers
        x1 = F.relu(agent.network.conv1(x))
        x2 = F.relu(agent.network.conv2(x1))
        x3 = F.relu(agent.network.conv3(x2))
        
        print(f"Conv1 output: mean={x1.mean().item():.6f}, std={x1.std().item():.6f}")
        print(f"Conv2 output: mean={x2.mean().item():.6f}, std={x2.std().item():.6f}")
        print(f"Conv3 output: mean={x3.mean().item():.6f}, std={x3.std().item():.6f}")
        
        # Pooling
        x_pool = agent.network.pool(x3)
        print(f"Pool output: mean={x_pool.mean().item():.6f}, std={x_pool.std().item():.6f}")
        
        # Flatten
        x_flat = x_pool.reshape(x_pool.size(0), -1)
        print(f"Flattened: mean={x_flat.mean().item():.6f}, std={x_flat.std().item():.6f}")
        
        # Shared layers
        x_shared = agent.network.shared_layers(x_flat)
        print(f"Shared layers output: mean={x_shared.mean().item():.6f}, std={x_shared.std().item():.6f}")
        
        # Actor head
        logits = agent.network.actor(x_shared)
        print(f"Actor logits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
        
        # Critic head
        value = agent.network.critic(x_shared)
        print(f"Critic value: {value.item():.6f}")
    
    # Test 5: Check if the issue is in the action selection
    print("\n--- Test 5: Action Selection Test ---")
    try:
        action, prob, value = agent.select_action(state, info['legal_actions'])
        print(f"Selected action: {action}")
        print(f"Action probability: {prob}")
        print(f"Value: {value}")
    except Exception as e:
        print(f"Error in action selection: {e}")
    
    # Test 6: Check if the issue is in the training step
    print("\n--- Test 6: Training Step Test ---")
    try:
        # Create dummy training data
        dummy_states = np.array([state] * 4)
        dummy_actions = np.array([info['legal_actions'][0]] * 4)
        dummy_rewards = np.array([0.1] * 4)
        dummy_values = np.array([0.0] * 4)
        dummy_action_probs = np.array([0.1] * 4)
        dummy_returns = np.array([0.1] * 4)
        dummy_advantages = np.array([0.0] * 4)
        dummy_legal_actions_list = [info['legal_actions']] * 4
        
        loss_dict = agent.train_step(
            dummy_states, dummy_actions, dummy_action_probs,
            dummy_returns, dummy_advantages, dummy_legal_actions_list
        )
        
        print("Training step successful:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error in training step: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary and recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    if health_report['logits']['all_negative']:
        print("❌ PROBLEM: All logits are negative!")
        print("   This causes softmax to produce very small probabilities.")
        print("   RECOMMENDATION: Check network initialization and training stability.")
    
    if health_report['probabilities']['all_zero']:
        print("❌ PROBLEM: All probabilities are zero!")
        print("   This causes the Categorical distribution to fail.")
        print("   RECOMMENDATION: Fix the logits issue above.")
    
    if health_report['logits']['has_nan'] or health_report['probabilities']['has_nan']:
        print("❌ PROBLEM: NaN values detected!")
        print("   RECOMMENDATION: Check for gradient explosion or numerical instability.")
    
    if health_report['logits']['has_inf'] or health_report['probabilities']['has_inf']:
        print("❌ PROBLEM: Infinite values detected!")
        print("   RECOMMENDATION: Check for gradient explosion.")
    
    # Check if the issue is systematic across positions
    all_negative_count = sum(1 for logits in all_logits if logits['all_negative'])
    all_zero_count = sum(1 for probs in all_probs if probs['all_zero'])
    
    if all_negative_count == len(all_logits):
        print("❌ PROBLEM: Issue is systematic across all positions!")
        print("   RECOMMENDATION: This is likely a network initialization or architecture issue.")
    else:
        print(f"✅ Issue is not systematic: {all_negative_count}/{len(all_logits)} positions affected.")
    
    print("\n=== POTENTIAL FIXES ===")
    print("1. Better weight initialization (already implemented)")
    print("2. Add gradient clipping")
    print("3. Use a smaller learning rate")
    print("4. Add batch normalization")
    print("5. Use a different activation function")
    print("6. Check for exploding gradients during training")

if __name__ == "__main__":
    analyze_network_behavior() 