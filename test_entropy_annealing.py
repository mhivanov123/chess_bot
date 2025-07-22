#!/usr/bin/env python3
"""
Test script to verify entropy annealing is working correctly.
"""

import numpy as np
from src.agents.ppo_agent import PPOAgent

def test_entropy_annealing():
    """Test entropy annealing functionality."""
    
    # Create agent with entropy annealing enabled
    agent = PPOAgent(
        state_shape=(8, 8, 13),
        num_actions=4352,
        entropy_coef=0.1,
        entropy_annealing=True,
        entropy_annealing_start=0.1,
        entropy_annealing_end=0.01,
        entropy_annealing_steps=100
    )
    
    print("=== ENTROPY ANNEALING TEST ===")
    print(f"Initial entropy coefficient: {agent.entropy_coef}")
    
    # Simulate training steps and check entropy coefficient
    for step in range(0, 120, 10):
        agent.total_steps = step
        old_coef = agent.entropy_coef
        agent.update_entropy_coef()
        new_coef = agent.entropy_coef
        
        print(f"Step {step:3d}: {old_coef:.4f} -> {new_coef:.4f}")
    
    print(f"\nFinal entropy coefficient: {agent.entropy_coef}")
    
    # Test that it stops changing after annealing_steps
    agent.total_steps = 150
    agent.update_entropy_coef()
    print(f"After annealing period: {agent.entropy_coef}")
    
    print("\n✅ Entropy annealing test completed!")

if __name__ == "__main__":
    test_entropy_annealing() 