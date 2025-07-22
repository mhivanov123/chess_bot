import sys
import os
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.chess_env import ChessEnv
from agents.ppo_agent import PPOAgent

def load_agent(model_path, device='cpu'):
    """Load an agent with debugging information."""
    print(f"Loading agent from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint FIRST to see what architecture it expects
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Check the actor layer size in the checkpoint
    if 'network_state_dict' in checkpoint:
        state_dict = checkpoint['network_state_dict']
        actor_weight_key = 'actor.weight'
        if actor_weight_key in state_dict:
            actor_output_size = state_dict[actor_weight_key].shape[0]
            print(f"Checkpoint actor output size: {actor_output_size}")
        else:
            print("Warning: Could not find actor.weight in checkpoint")
            raise ValueError("Invalid checkpoint format")
    else:
        print("Warning: Could not find network_state_dict in checkpoint")
        raise ValueError("Invalid checkpoint format")
    
    # Create environment to get the correct state shape
    dummy_env = ChessEnv()
    print(f"Environment action space size: {dummy_env.action_space.n}")
    print(f"Environment state shape: {dummy_env.observation_space.shape}")
    
    # Create agent with the EXACT action space size from the checkpoint
    agent = PPOAgent(
        state_shape=dummy_env.observation_space.shape,
        num_actions=actor_output_size  # Use the checkpoint's action space size
    )
    
    try:
        agent.load(model_path)
        print(f"Successfully loaded agent with {actor_output_size} actions")
    except Exception as e:
        print(f"Error loading agent: {e}")
        raise
    
    agent.set_training(False)
    return agent

def play_match(agent_white, agent_black, verbose=False):
    """Play a match between two agents."""
    env = ChessEnv()
    state, info = env.reset()
    legal_actions = info['legal_actions']
    done = False
    current_agent = agent_white
    move_count = 0
    
    print(f"Starting game with {len(legal_actions)} legal moves")
    if verbose:
        print("Initial board:")
        print(env.board)

    while not done:
        try:
            action, _, _ = current_agent.select_action(state, legal_actions)
            state, reward, done, truncated, info = env.step(action)
            legal_actions = info['legal_actions']
            move_count += 1
            
            if verbose:
                print(f"Move {move_count}: {info.get('last_move_san', 'Unknown')}")
                print(env.board)
                print('-' * 30)
            
            # Alternate agent
            current_agent = agent_black if current_agent == agent_white else agent_white
            
        except Exception as e:
            print(f"Error during move {move_count}: {e}")
            return "Error"

    # Determine result
    if env.board.is_checkmate():
        winner = "Black" if env.board.turn else "White"
        print(f"Checkmate! {winner} wins in {move_count} moves.")
        return winner
    elif env.board.is_stalemate():
        print("Stalemate!")
        return "Draw"
    elif env.board.is_insufficient_material():
        print("Insufficient material!")
        return "Draw"
    elif env.move_count >= env.max_moves:
        print("Move limit reached!")
        return "Draw"
    else:
        print("Other termination!")
        return "Draw"

def play_batch_matches(agent_white, agent_black, num_matches=10, verbose=False):
    """Play a batch of matches between two agents."""
    results = []
    for _ in range(num_matches):
        result = play_match(agent_white, agent_black, verbose=verbose)
        results.append(result)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play two PPO agents against each other.")
    parser.add_argument("--white", type=str, required=True, help="Path to model for White")
    parser.add_argument("--black", type=str, required=True, help="Path to model for Black")
    parser.add_argument("--verbose", action="store_true", help="Print board after each move")
    args = parser.parse_args()

    try:
        print("=" * 50)
        print("LOADING WHITE AGENT")
        print("=" * 50)
        agent_white = load_agent(args.white)
        
        print("=" * 50)
        print("LOADING BLACK AGENT")
        print("=" * 50)
        agent_black = load_agent(args.black)
        
        print("=" * 50)
        print("STARTING MATCH")
        print("=" * 50)
        results = play_batch_matches(agent_white, agent_black, num_matches=100, verbose=args.verbose)
        print(f"Final Results: {results}")
        print(f"Win rate: {results.count('White') / len(results)}")
        print(f"Draw rate: {results.count('Draw') / len(results)}")
        print(f"Loss rate: {results.count('Black') / len(results)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 