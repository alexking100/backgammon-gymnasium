"""
TD(0) Classes and Functions for Backgammon
==========================================

Reusable components for TD(0) learning:
- TrainingLog: Logs training metrics and game replays
- ValueNetwork: Neural network for state value estimation
- choose_best_action: Action selection using value network
- train_value_network: Main training loop
- test_agent: Test trained agent against random player
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

# =============================================================================
# CONFIGURATION (can be overridden when calling functions)
# =============================================================================

DEFAULT_LOG_EVERY_N_EPISODES = 10


# =============================================================================
# TRAINING LOGGER
# =============================================================================

class TrainingLog:
    """Simple logger to track what happens during training."""
    
    def __init__(self):
        self.metrics = {
            'episode': [],
            'winner': [],
            'game_length': [],
            'avg_loss': [],
        }
        self.game_replays = []
        self.initial_values = []
    
    def log_episode(self, episode, winner, length, avg_loss):
        self.metrics['episode'].append(episode)
        self.metrics['winner'].append(winner)
        self.metrics['game_length'].append(length)
        self.metrics['avg_loss'].append(avg_loss)
    
    def log_game(self, episode, moves):
        """Store a full game replay."""
        self.game_replays.append({'episode': episode, 'moves': moves})
    
    def log_initial_value(self, episode, value):
        self.initial_values.append({'episode': episode, 'value': value})
    
    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        total = len(self.metrics['episode'])
        if total == 0:
            print("No episodes logged yet.")
            return
            
        white_wins = sum(1 for w in self.metrics['winner'] if w == WHITE)
        black_wins = sum(1 for w in self.metrics['winner'] if w == BLACK)
        
        print(f"Total episodes: {total}")
        print(f"White wins: {white_wins} ({white_wins/total*100:.1f}%)")
        print(f"Black wins: {black_wins} ({black_wins/total*100:.1f}%)")
        print(f"Avg game length: {np.mean(self.metrics['game_length']):.1f} moves")
        print(f"Final avg loss: {np.mean(self.metrics['avg_loss'][-10:]):.4f}")
        
        if self.initial_values:
            print(f"\nV(initial_state) evolution:")
            for v in self.initial_values[::max(1, len(self.initial_values)//5)]:
                print(f"  Episode {v['episode']:4d}: {v['value']:.4f}")
    
    def print_game_replay(self, game_idx=0):
        """Print a detailed replay of a logged game."""
        if not self.game_replays:
            print("No games logged!")
            return
        
        game = self.game_replays[min(game_idx, len(self.game_replays)-1)]
        print(f"\n" + "="*60)
        print(f"GAME REPLAY - Episode {game['episode']}")
        print("="*60)
        
        for move in game['moves']:
            player = move['player']
            roll = move['roll']
            action = move['action']
            v_current = move.get('v_current')
            v_next = move.get('v_next') or move.get('value')
            random_str = " (RANDOM)" if move.get('random', False) else ""
            
            print(f"\nMove {move['move_num']:3d} | {player:5s} | Roll: {roll}")
            print(f"  Action: {action}")
            
            if v_current is not None:
                print(f"  V(current): {v_current:.4f}  →  ", end="")
                if v_next is not None:
                    print(f"V(after_move): {v_next:.4f}{random_str}")
                else:
                    print(f"V(after_move): N/A{random_str}")
            elif v_next is not None:
                print(f"  V(next_state): {v_next:.4f}{random_str}")
            elif move.get('random', False):
                print(f"  V: N/A (RANDOM)")
            
            if 'alternatives' in move and move['alternatives']:
                print(f"  Alternatives considered:")
                for alt in move['alternatives'][:3]:
                    print(f"    V={alt['value']:.4f} | {alt['action'][:60]}...")
    
    def save(self, filepath="./examples/training_logs/training_log.json"):
        """Save logs to JSON file."""
        data = {
            'metrics': self.metrics,
            'game_replays': self.game_replays,
            'initial_values': self.initial_values
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Log saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath="./examples/training_logs/training_log.json"):
        """Load logs from JSON file."""
        log = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
        log.metrics = data['metrics']
        log.game_replays = data['game_replays']
        log.initial_values = data['initial_values']
        print(f"✅ Log loaded from: {filepath}")
        return log


# =============================================================================
# VALUE NETWORK
# =============================================================================

class ValueNetwork(nn.Module):
    """
    Neural network that estimates probability of WHITE winning from a given state.
    
    Input: 198-dimensional state vector (TD-Gammon encoding)
    Output: Single value in [0, 1] (probability of WHITE winning)
    """
    
    def __init__(self, input_size=198, hidden_size=80):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# ACTION SELECTION
# =============================================================================

def choose_best_action(env, value_net, actions, current_player, return_alternatives=False):
    """
    For each legal action, simulate it and evaluate the resulting state.
    Pick the action that leads to the best state for the current player.
    
    Args:
        env: The unwrapped backgammon environment
        value_net: The value network for evaluation
        actions: Set of legal actions
        current_player: WHITE or BLACK
        return_alternatives: If True, return list of all action values for logging
    
    Returns:
        (best_action, best_value) or (best_action, best_value, alternatives)
    """
    if not actions:
        return (None, None, []) if return_alternatives else (None, None)
    
    best_action = None
    best_value = float('-inf') if current_player == WHITE else float('inf')
    action_values = []
    
    # Save current state
    game_state = env.game.save_state()
    
    for action in actions:
        # Simulate the action
        env.game.execute_play(current_player, action)
        
        # Get the resulting state features
        next_state = env.game.get_board_features(current_player)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Evaluate with neural network
        with torch.no_grad():
            value = value_net(next_state_tensor).item()
        
        if return_alternatives:
            action_values.append({'action': str(action), 'value': value})
        
        # WHITE maximizes, BLACK minimizes
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
    
    if return_alternatives:
        action_values.sort(key=lambda x: x['value'], reverse=(current_player == WHITE))
        return best_action, best_value, action_values
    
    return best_action, best_value


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_value_network(num_episodes=100, gamma=0.99, verbose=True, logger=None, 
                        log_every_n=DEFAULT_LOG_EVERY_N_EPISODES):
    """
    Train the value network using TD(0) learning.
    
    The key equation:
        V(s) <- V(s) + α × [r + γ × V(s') - V(s)]
    
    Args:
        num_episodes: Number of games to play
        gamma: Discount factor
        verbose: Print progress every 50 episodes
        logger: TrainingLog instance (or None)
        log_every_n: Log full game every N episodes
    
    Returns:
        (trained_value_net, wins_dict)
    """
    # Create environment
    env = gym.make('gym_backgammon:backgammon-v0')
    unwrapped_env = env.unwrapped
    
    # Create network and optimizer
    value_net = ValueNetwork()
    optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    
    # Statistics
    wins = {WHITE: 0, BLACK: 0}
    episode_lengths = []
    
    # Get initial state for tracking V evolution
    _, _, init_obs = env.reset()
    init_state_tensor = torch.FloatTensor(init_obs).unsqueeze(0)
    
    for episode in range(num_episodes):
        agent_color, roll, obs = env.reset()
        done = False
        moves = 0
        
        trajectory = []
        current_state = torch.FloatTensor(obs).unsqueeze(0)
        
        # Should we log this full game?
        log_this_game = (logger is not None and 
                        (episode < 5 or 
                         episode % log_every_n == 0 or 
                         episode >= num_episodes - 5))
        game_moves_log = [] if log_this_game else None
        
        while not done:
            actions = unwrapped_env.get_valid_actions(roll)
            
            if actions:
                # 10% random exploration
                is_random = random.random() < 0.1
                if is_random:
                    action = random.choice(list(actions))
                    chosen_value = None
                    alternatives = []
                else:
                    if log_this_game:
                        action, chosen_value, alternatives = choose_best_action(
                            unwrapped_env, value_net, actions, agent_color, return_alternatives=True)
                    else:
                        action, chosen_value = choose_best_action(
                            unwrapped_env, value_net, actions, agent_color)
                        alternatives = []
                
                # Log move
                if log_this_game:
                    with torch.no_grad():
                        v_current = value_net(current_state).item()
                    
                    game_moves_log.append({
                        'move_num': moves,
                        'player': COLORS[agent_color],
                        'roll': str(roll),
                        'action': str(action),
                        'v_current': v_current,
                        'v_next': chosen_value,
                        'random': is_random,
                        'num_actions': len(actions),
                        'alternatives': alternatives[:5] if alternatives else []
                    })
                
                obs, reward, done, info = env.step(action)
                moves += 1
                
                next_state = torch.FloatTensor(obs).unsqueeze(0)
                
                trajectory.append({
                    'state': current_state,
                    'next_state': next_state,
                    'done': done,
                    'reward': reward
                })
                
                current_state = next_state
            
            if not done:
                agent_color = unwrapped_env.get_opponent_agent()
                if agent_color == WHITE:
                    roll = (-random.randint(1,6), -random.randint(1,6))
                else:
                    roll = (random.randint(1,6), random.randint(1,6))
        
        # Update statistics
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1
        episode_lengths.append(moves)
        
        # TD Learning updates
        final_reward = 1.0 if winner == WHITE else 0.0
        value_net.train()
        total_loss = 0
        
        for transition in trajectory:
            optimizer.zero_grad()
            
            state = transition['state']
            next_state = transition['next_state']
            is_terminal = transition['done']
            
            predicted_value = value_net(state)
            
            with torch.no_grad():
                if is_terminal:
                    td_target = torch.tensor([[final_reward]])
                else:
                    next_value = value_net(next_state)
                    td_target = 0.0 + gamma * next_value
            
            loss = nn.MSELoss()(predicted_value, td_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(trajectory) if trajectory else 0
        
        # Log metrics
        if logger is not None:
            logger.log_episode(episode, winner, moves, avg_loss)
            if log_this_game and game_moves_log:
                logger.log_game(episode, game_moves_log)
            if episode % 20 == 0:
                with torch.no_grad():
                    init_value = value_net(init_state_tensor).item()
                logger.log_initial_value(episode, init_value)
        
        if verbose and (episode + 1) % 50 == 0:
            white_pct = wins[WHITE] / (episode + 1) * 100
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1:4d} | White wins: {white_pct:5.1f}% | Avg moves: {avg_length:.1f} | Loss: {avg_loss:.4f}")
    
    env.close()
    return value_net, wins


# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_agent(value_net, num_games=50):
    """Test trained agent (WHITE) vs random agent (BLACK)."""
    env = gym.make('gym_backgammon:backgammon-v0')
    unwrapped_env = env.unwrapped
    wins = {WHITE: 0, BLACK: 0}
    
    for game in range(num_games):
        agent_color, roll, obs = env.reset()
        done = False
        
        while not done:
            actions = unwrapped_env.get_valid_actions(roll)
            
            if actions:
                if agent_color == WHITE:
                    action, _ = choose_best_action(unwrapped_env, value_net, actions, agent_color)
                else:
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
    
    env.close()
    return wins

