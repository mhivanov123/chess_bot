#!/usr/bin/env python3
"""
Comprehensive test script to verify the chess RL framework works correctly.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.ppo_agent import PPOAgent
from training.ppo_trainer import PPOTrainer
from utils.config import load_config

def test_environment():
    """Test the chess environment."""
    print("Testing chess environment...")
    env = ChessEnv()
    state, info = env.reset()
    assert state.shape == (8, 8, 13)
    assert isinstance(info['legal_actions'], list)
    print(f"Initial state shape: {state.shape}")
    print(f"Legal actions: {len(info['legal_actions'])}")
    
    # Test that legal actions actually correspond to legal moves
    legal_moves = list(env.board.legal_moves)
    assert len(info['legal_actions']) == len(legal_moves), "Legal actions count should match legal moves count"
    
    for i in range(5):
        legal_actions = info['legal_actions']
        if not legal_actions:
            break
        action = legal_actions[0]
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Move {i+1}: Reward = {reward:.3f}, Done = {done}")
        if done:
            break
    print("Environment test passed!\n")

def test_action_space_mapping():
    """Test that action space mapping is consistent."""
    print("Testing action space mapping...")
    env = ChessEnv()
    state, info = env.reset()
    
    # Test that every legal action maps to a legal move
    legal_actions = info['legal_actions']
    legal_moves = list(env.board.legal_moves)
    
    print(f"Testing {len(legal_actions)} legal actions...")
    
    for i, action in enumerate(legal_actions[:10]):  # Test first 10
        # Convert action to move
        move = env._action_to_move(action)
        
        # Check if move is actually legal
        if move not in legal_moves:
            print(f"❌ Action {action} maps to illegal move {move}")
            print(f"   Legal moves: {[env.board.san(m) for m in legal_moves[:5]]}")
            assert False, f"Action {action} maps to illegal move {move}"
        
        # Test round-trip conversion
        reconstructed_action = env._move_to_action(move)
        if action != reconstructed_action:
            print(f"❌ Round-trip conversion failed: {action} -> {move} -> {reconstructed_action}")
            assert False, f"Round-trip conversion failed: {action} -> {move} -> {reconstructed_action}"
    
    print("✅ Action space mapping test passed!\n")

def test_ppo_agent_init():
    print("Testing PPOAgent initialization...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    assert isinstance(agent, PPOAgent)
    print("PPOAgent initialization test passed!\n")

def test_ppo_agent_forward():
    print("Testing PPOAgent forward pass...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_logits, value = agent.network.forward(state_tensor)
    assert action_logits.shape[1] == agent.num_actions
    assert value.shape == (1, 1)
    print("PPOAgent forward pass test passed!\n")

def test_ppo_agent_action_selection():
    print("Testing PPOAgent action selection...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    legal_actions = info['legal_actions']
    action, action_prob, value = agent.select_action(state, legal_actions)
    assert isinstance(action, int)
    assert 0 <= action < agent.num_actions
    assert isinstance(action_prob, float)
    print(f"Selected action: {action}, prob: {action_prob:.4f}, value: {value:.4f}")
    print("PPOAgent action selection test passed!\n")

def test_ppo_agent_illegal_action():
    print("Testing PPOAgent illegal action handling...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    
    # Pick an action that is not legal
    all_actions = set(range(agent.num_actions))
    legal_actions = set(info['legal_actions'])
    illegal_actions = list(all_actions - legal_actions)
    
    if illegal_actions:
        action = illegal_actions[0]
        next_state, reward, done, truncated, info = env.step(action)
        assert done, "Game should terminate on illegal move"
        assert reward == -10.0, f"Expected -10.0 reward for illegal move, got {reward}"
        assert info.get('illegal_move', False), "Should mark move as illegal"
        print("✅ Illegal action correctly handled by environment.")
    else:
        print("No illegal actions available in initial position (rare).")
    print("PPOAgent illegal action test passed!\n")

def test_ppo_agent_legal_moves_only():
    """Test that agent only selects legal moves when using legal action selection."""
    print("Testing PPOAgent legal move selection...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    
    # Test multiple moves to ensure agent selects legal moves
    for i in range(10):
        legal_actions = info['legal_actions']
        if not legal_actions:
            break
            
        action, action_prob, value = agent.select_action(state, legal_actions)
        
        # Check if action is legal
        if action not in legal_actions:
            print(f"❌ Agent selected illegal action {action} on move {i+1}")
            print(f"   Legal actions: {legal_actions[:5]}...")
            assert False, f"Agent selected illegal action {action}"
        
        # Take the action
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        
        if done:
            break
    
    print("✅ Agent only selected legal moves!")
    print("PPOAgent legal move selection test passed!\n")

def test_ppo_agent_train_step():
    print("Testing PPOAgent train_step...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    
    # Collect a short trajectory
    states, actions, rewards, values, action_probs, legal_actions_list, dones = [], [], [], [], [], [], []
    for _ in range(5):
        legal_actions = info['legal_actions']
        action, action_prob, value = agent.select_action(state, legal_actions)
        next_state, reward, done, truncated, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        action_probs.append(action_prob)
        legal_actions_list.append(legal_actions)
        dones.append(done)
        state = next_state
        if done:
            break
    
    # Fake returns/advantages for test
    returns, advantages = agent.compute_gae_returns(np.array(rewards), np.array(values), np.array(dones))
    agent.train_step(np.array(states), np.array(actions), returns, advantages, np.array(action_probs), legal_actions_list)
    print("PPOAgent train_step test passed!\n")

def test_ppo_agent_save_load():
    print("Testing PPOAgent save/load...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    tmp_path = "ppo_agent_test.pth"
    agent.save(tmp_path)
    agent2 = PPOAgent(state_shape=env.observation_space.shape)
    agent2.load(tmp_path)
    os.remove(tmp_path)
    print("PPOAgent save/load test passed!\n")

def test_ppo_agent_gae():
    print("Testing PPOAgent GAE computation...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    rewards = np.array([1, 0, 0, 1, 0], dtype=np.float32)
    values = np.array([0.5, 0.2, 0.1, 0.3, 0.0], dtype=np.float32)
    dones = np.array([0, 0, 0, 0, 1], dtype=np.bool_)
    returns, advantages = agent.compute_gae_returns(rewards, values, dones)
    assert returns.shape == rewards.shape
    assert advantages.shape == rewards.shape
    print("PPOAgent GAE computation test passed!\n")

def test_ppo_agent_episode():
    print("Testing PPOAgent end-to-end episode...")
    env = ChessEnv()
    agent = PPOAgent(state_shape=env.observation_space.shape)
    state, info = env.reset()
    total_reward = 0
    illegal_moves = 0
    
    for move_num in range(20):
        legal_actions = info['legal_actions']
        action, action_prob, value = agent.select_action(state, legal_actions)
        
        # Check if action is legal before taking it
        if action not in legal_actions:
            illegal_moves += 1
            print(f"❌ Illegal action {action} selected on move {move_num + 1}")
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            if info.get('illegal_move', False):
                print(f"❌ Game terminated due to illegal move on move {move_num + 1}")
            break
    
    print(f"Episode finished with total reward: {total_reward:.2f}")
    print(f"Illegal moves made: {illegal_moves}")
    
    # The test should fail if there are illegal moves
    assert illegal_moves == 0, f"Agent made {illegal_moves} illegal moves"
    print("PPOAgent end-to-end episode test passed!\n")

def test_ppo_trainer_trajectory():
    print("Testing PPOTrainer trajectory collection...")
    config_path = "configs/ppo_config.yaml"
    trainer = PPOTrainer(config_path)
    reward, length = trainer.collect_trajectory()
    assert isinstance(reward, float)
    assert isinstance(length, int)
    assert length > 0
    print(f"Collected trajectory: reward={reward:.2f}, length={length}")
    print("PPOTrainer trajectory collection test passed!\n")

def test_ppo_trainer_evaluate():
    print("Testing PPOTrainer evaluation...")
    config_path = "configs/ppo_config.yaml"
    trainer = PPOTrainer(config_path)
    results = trainer.evaluate_agent(num_games=2)
    assert isinstance(results, dict)
    
    # Check for expected keys in results
    expected_keys = ['win_rate', 'draw_rate', 'loss_rate', 'avg_moves']
    found_keys = [key for key in expected_keys if key in results]
    assert len(found_keys) > 0, f"Expected at least one of {expected_keys} in results"
    
    # Check that results are reasonable
    assert 0 <= results['win_rate'] <= 1, f"Win rate should be between 0 and 1, got {results['win_rate']}"
    assert 0 <= results['draw_rate'] <= 1, f"Draw rate should be between 0 and 1, got {results['draw_rate']}"
    assert 0 <= results['loss_rate'] <= 1, f"Loss rate should be between 0 and 1, got {results['loss_rate']}"
    assert abs(results['win_rate'] + results['draw_rate'] + results['loss_rate'] - 1.0) < 1e-6, "Rates should sum to 1"
    
    print(f"Evaluation results: {results}")
    print("PPOTrainer evaluation test passed!\n")

def test_ppo_trainer_no_illegal_moves():
    """Test that PPOTrainer doesn't produce illegal moves."""
    print("Testing PPOTrainer for illegal moves...")
    config_path = "configs/ppo_config.yaml"
    trainer = PPOTrainer(config_path)
    
    # Collect a trajectory and check for illegal moves
    reward, length = trainer.collect_trajectory()
    
    # Check the memory for any illegal moves
    states, actions, rewards, values, action_probs, legal_actions_list, dones = trainer.agent.memory.get_batch()
    
    illegal_count = 0
    for i, (action, legal_actions) in enumerate(zip(actions, legal_actions_list)):
        if action not in legal_actions:
            illegal_count += 1
            print(f"❌ Illegal action {action} found in trajectory at step {i}")
    
    assert illegal_count == 0, f"Found {illegal_count} illegal moves in trajectory"
    print("✅ No illegal moves found in trajectory!")
    print("PPOTrainer illegal move test passed!\n")

def assert_move_is_legal_and_san(env, action):
    move = env._action_to_move(action)
    is_legal = move in env.board.legal_moves
    try:
        move_san = env.board.san(move)
        san_success = True
    except Exception:
        san_success = False
    if not is_legal or not san_success:
        print(f"❌ Detected invalid move or SAN failure! Action: {action}, Move: {move}, Is legal: {is_legal}, SAN success: {san_success}")
        print(f"    Board FEN: {env.board.fen()}")
        print(f"    Legal moves: {[env.board.san(m) for m in env.board.legal_moves]}")
        assert False, "Invalid move or SAN conversion failure detected!"

if __name__ == "__main__":
    test_environment()
    test_action_space_mapping()  # NEW: Test action space mapping
    test_ppo_agent_init()
    test_ppo_agent_forward()
    test_ppo_agent_action_selection()
    test_ppo_agent_illegal_action()
    test_ppo_agent_legal_moves_only()  # NEW: Test legal move selection
    test_ppo_agent_train_step()
    test_ppo_agent_save_load()
    test_ppo_agent_gae()
    test_ppo_agent_episode()
    test_ppo_trainer_trajectory()
    test_ppo_trainer_evaluate()
    test_ppo_trainer_no_illegal_moves()  # NEW: Test trainer for illegal moves
    print("All tests passed!") 