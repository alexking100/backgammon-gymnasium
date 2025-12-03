"""
TD(0) Learning for Backgammon
=============================

This file implements TD(0) - Temporal Difference Learning with a Value Network,
similar to the approach used in TD-Gammon (Tesauro, 1992).

This file teaches you:
1. How Gymnasium works
2. How to build a Value Network with PyTorch
3. How to train using TD(0) learning
4. The TD update formula: V(s) ← V(s) + α × [r + γ × V(s') - V(s)]

Note: We use a Value Function V(s) approach rather than Q(s,a) because
backgammon has a variable action space that changes with each dice roll.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN

# =============================================================================
# PART 1: GYMNASIUM BASICS
# =============================================================================

# Create the environment
env = gym.make('gym_backgammon:backgammon-v0')
unwrapped_env = env.unwrapped

# STEP 1: Reset the environment (start a new game)
agent_color, first_roll, observation = env.reset()

print(f"\n1. env.reset() returns:")
print(f"   - agent_color: {agent_color} ({COLORS[agent_color]})")
print(f"   - first_roll: {first_roll}")
print(f"   - observation: array of {len(observation)} features")

# STEP 2: Get valid actions for this dice roll
actions = unwrapped_env.get_valid_actions(first_roll)
print(f"\n2. get_valid_actions(roll) returns:")
print(f"   - {len(actions)} legal actions for roll {first_roll}")
print(f"   - Example action: {list(actions)[0] if actions else 'None'}")

# STEP 3: Take an action
if actions:
    action = random.choice(list(actions))
    observation_next, reward, done, info = env.step(action)
    
    print(f"\n3. env.step(action) returns:")
    print(f"   - observation: array of {len(observation_next)} features")
    print(f"   - reward: {reward}")
    print(f"   - done: {done}")
    print(f"   - info: {info}")

# STEP 4: The full game loop
print(f"\n4. Full game loop structure:")
print("""
    agent_color, roll, obs = env.reset()
    done = False
    
    while not done:
        actions = env.get_valid_actions(roll)
        action = pick_action(actions)  # Your AI goes here!
        obs, reward, done, info = env.step(action)
        
        if not done:
            agent_color = env.get_opponent_agent()
            roll = roll_dice(agent_color)
""")

# =============================================================================
# PART 2: THE NEURAL NETWORK
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Building a Value Network")
print("=" * 60)

class ValueNetwork(nn.Module):
    """
    A neural network that estimates the probability of winning from a given state.
    
    Input: 198-dimensional state vector
    Output: Single value between 0 and 1 (probability of WHITE winning)
    """
    
    def __init__(self, input_size=198, hidden_size=80):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 198 -> 80
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 80 -> 80
            nn.ReLU(),
            nn.Linear(hidden_size, 1),            # 80 -> 1
            nn.Sigmoid()                          # Squash to [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)

# Create the network
value_net = ValueNetwork()
print(f"\nNetwork architecture:")
print(value_net)

# Test it with a sample state
sample_state = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension
sample_value = value_net(sample_state)
print(f"\nSample prediction: V(initial_state) = {sample_value.item():.4f}")

# =============================================================================
# PART 3: CHOOSING ACTIONS WITH THE VALUE NETWORK
# =============================================================================


def choose_best_action(env, value_net, actions, current_player):
    """
    For each legal action, simulate it and evaluate the resulting state.
    Pick the action that leads to the best state for the current player.
    """
    if not actions:
        return None
    
    best_action = None
    best_value = float('-inf') if current_player == WHITE else float('inf')
    
    # Save current state
    game_state = env.game.save_state()
    
    for action in actions:
        # Simulate the action
        env.game.execute_play(current_player, action)
        
        # Get the resulting state features
        next_state = env.game.get_board_features(current_player)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Evaluate with neural network (no gradient needed for inference)
        with torch.no_grad():
            value = value_net(next_state_tensor).item()
        
        # WHITE wants to maximize, BLACK wants to minimize
        if current_player == WHITE:
            if value > best_value:
                best_value = value
                best_action = action
        else:
            if value < best_value:
                best_value = value
                best_action = action
        
        # Restore state to try next action
        env.game.restore_state(game_state)
    
    return best_action, best_value

print("""
How choose_best_action works:

    For each legal action:
        1. Simulate: "What if I take this action?"
        2. Evaluate: "How good is the resulting position?"
        3. Compare: "Is this better than my best so far?"
    
    Return the action that leads to the best position!
""")

# =============================================================================
# PART 4: TRAINING WITH TD LEARNING
# =============================================================================

print("=" * 60)
print("PART 4: Training Loop")
print("=" * 60)

def train_value_network(num_episodes=100, alpha=0.1, gamma=0.99, verbose=True):
    """
    Train the value network using Temporal Difference learning.
    
    The key equation:
        V(s) <- V(s) + α * [reward + γ * V(s') - V(s)]
        TD Target = [reward + γ * V(s') - V(s)]
    """
    
    # Create fresh network and optimizer
    value_net = ValueNetwork()
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    
    # Track statistics
    wins = {WHITE: 0, BLACK: 0}
    episode_lengths = []
    
    for episode in range(num_episodes):
        # Reset environment
        agent_color, roll, obs = env.reset()
        done = False
        moves = 0
        
        # Store trajectory for learning: (state, next_state, is_terminal, reward)
        trajectory = []
        current_state = torch.FloatTensor(obs).unsqueeze(0)
        
        while not done:
            # Get valid actions
            actions = unwrapped_env.get_valid_actions(roll)
            
            if actions:
                # Choose action (with some exploration)
                if random.random() < 0.1:  # 10% exploration
                    action = random.choice(list(actions))
                else:
                    action, _ = choose_best_action(unwrapped_env, value_net, actions, agent_color)
                
                # Take action
                obs, reward, done, info = env.step(action)
                moves += 1
                
                # Get next state
                next_state = torch.FloatTensor(obs).unsqueeze(0)
                
                # Store transition
                trajectory.append({
                    'state': current_state,
                    'next_state': next_state,
                    'done': done,
                    'reward': reward
                })
                
                current_state = next_state
            
            if not done:
                # Switch players
                agent_color = unwrapped_env.get_opponent_agent()
                # Roll dice for next player
                if agent_color == WHITE:
                    roll = (-random.randint(1,6), -random.randint(1,6))
                else:
                    roll = (random.randint(1,6), random.randint(1,6))
        
        # Game ended - update statistics
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1
        episode_lengths.append(moves)
        
        # =================================================================
        # PROPER TD LEARNING: Update ALL states in the trajectory
        # =================================================================
        # Final reward: 1 if WHITE won, 0 if BLACK won
        final_reward = 1.0 if winner == WHITE else 0.0
        
        value_net.train()
        total_loss = 0
        
        # Go through trajectory and update each state
        for i, transition in enumerate(trajectory):
            optimizer.zero_grad()
            
            state = transition['state']
            next_state = transition['next_state']
            is_terminal = transition['done']
            
            # Predict current state value
            predicted_value = value_net(state)
            
            # Calculate TD target using the FULL formula:
            # TD Target = r + γ × V(s')
            #
            # In backgammon:
            #   - During game: r = 0, so target = 0 + γ × V(s') = γ × V(s')
            #   - Terminal: r = final_reward, no next state, so target = r
            
            with torch.no_grad():
                if is_terminal:
                    # Terminal state: TD Target = reward (no future)
                    reward = final_reward
                    td_target = torch.tensor([[reward]])
                else:
                    # Non-terminal: TD Target = r + γ × V(s')
                    # In backgammon, intermediate reward r = 0
                    reward = 0.0
                    next_value = value_net(next_state)
                    td_target = reward + gamma * next_value  # Full formula!
            
            # TD error and update
            loss = nn.MSELoss()(predicted_value, td_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            white_pct = wins[WHITE] / (episode + 1) * 100
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = total_loss / len(trajectory) if trajectory else 0
            print(f"Episode {episode+1:4d} | White wins: {white_pct:5.1f}% | Avg moves: {avg_length:.1f} | Loss: {avg_loss:.4f}")
    
    return value_net, wins

# =============================================================================
# PART 5: RUN TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Training!")
print("=" * 60)
print("\nStarting training...\n")

trained_net, final_wins = train_value_network(num_episodes=1000)

print(f"\n" + "=" * 60)
print(f"Training complete!")
print(f"Final stats: WHITE {final_wins[WHITE]} wins, BLACK {final_wins[BLACK]} wins")
print("=" * 60)

# =============================================================================
# PART 6: TEST THE TRAINED AGENT
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Testing the trained agent vs Random")
print("=" * 60)

def test_agent(value_net, num_games=20):
    """Test trained agent (WHITE) vs random agent (BLACK)"""
    wins = {WHITE: 0, BLACK: 0}
    
    for game in range(num_games):
        agent_color, roll, obs = env.reset()
        done = False
        
        while not done:
            actions = unwrapped_env.get_valid_actions(roll)
            
            if actions:
                if agent_color == WHITE:
                    # Trained agent
                    action, _ = choose_best_action(unwrapped_env, value_net, actions, agent_color)
                else:
                    # Random agent
                    action = random.choice(list(actions))
                
                obs, reward, done, info = env.step(action)
            
            if not done:
                agent_color = unwrapped_env.get_opponent_agent()
                if agent_color == WHITE:
                    roll = (-random.randint(1,6), -random.randint(1,6))
                else:
                    roll = (random.randint(1,6), random.randint(1,6))
        
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1
    
    return wins

test_results = test_agent(trained_net, num_games=20)
print(f"\nTrained WHITE vs Random BLACK over 20 games:")
print(f"  WHITE (trained): {test_results[WHITE]} wins ({test_results[WHITE]/20*100:.0f}%)")
print(f"  BLACK (random):  {test_results[BLACK]} wins ({test_results[BLACK]/20*100:.0f}%)")

env.close()
print("\nDone! 🎲")
