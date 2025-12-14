# TD0 Analysis Script
# ===================
# Loads and analyzes training logs from TD0 training runs.

import sys
sys.path.insert(0, '.')  # Add project root to path

from examples.TD0_classes import TrainingLog
import matplotlib.pyplot as plt
import numpy as np

# Load the saved training log (run TD0.py first!)
log = TrainingLog.load("./examples/training_logs/training_log.json")

# Quick summary
log.print_summary()

# Show an early game
print("\n" + "="*60)
print("EARLY GAME REPLAY:")
log.print_game_replay(game_idx=0)

# Show a late game
print("\n" + "="*60)
print("LATE GAME REPLAY:")
log.print_game_replay(game_idx=-1)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Win rate over time (rolling average)
episodes = log.metrics['episode']
winners = log.metrics['winner']
white_wins = [1 if w == 0 else 0 for w in winners]  # WHITE = 0
rolling_win_rate = np.convolve(white_wins, np.ones(20)/20, mode='valid')
axes[0,0].plot(rolling_win_rate)
axes[0,0].set_title('White Win Rate (rolling avg)')
axes[0,0].set_xlabel('Episode')
axes[0,0].set_ylabel('Win Rate')

# Game length over time
axes[0,1].plot(log.metrics['game_length'], alpha=0.5)
axes[0,1].set_title('Game Length')
axes[0,1].set_xlabel('Episode')
axes[0,1].set_ylabel('Moves')

# Loss over time
axes[1,0].plot(log.metrics['avg_loss'])
axes[1,0].set_title('Average Loss')
axes[1,0].set_xlabel('Episode')
axes[1,0].set_ylabel('Loss')

# V(initial_state) evolution
if log.initial_values:
    init_eps = [v['episode'] for v in log.initial_values]
    init_vals = [v['value'] for v in log.initial_values]
    axes[1,1].plot(init_eps, init_vals, 'o-')
    axes[1,1].set_title('V(initial_state) Over Training')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Value')

plt.tight_layout()
plt.savefig('./analysis/training_curves.png')
plt.show()

print("\n✅ Training curves saved to ./analysis/training_curves.png")


# =============================================================================
# PLOT VALUE FUNCTION THROUGHOUT GAMES
# =============================================================================
# This shows V(s) at each move during games at different training stages.
# If learning is working, we should see:
# - Early games: V(s) is noisy/random
# - Late games: V(s) smoothly trends toward 1 (WHITE win) or 0 (BLACK win)

print("\n" + "="*60)
print("VALUE FUNCTION THROUGHOUT GAMES")
print("="*60)

if log.game_replays:
    # Select games from different training stages
    num_games = len(log.game_replays)
    
    # Pick games spread throughout training
    if num_games >= 4:
        game_indices = [0, num_games//3, 2*num_games//3, -1]
    else:
        game_indices = list(range(num_games))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, game_idx in enumerate(game_indices):
        if idx >= 4:
            break
            
        game = log.game_replays[game_idx]
        episode = game['episode']
        moves = game['moves']
        
        # Extract V(current_state) for each move
        move_nums = []
        v_values = []
        colors = []  # Color by player
        
        for move in moves:
            move_nums.append(move['move_num'])
            # Use v_current (value before making move)
            v = move.get('v_current')
            if v is not None:
                v_values.append(v)
            else:
                v_values.append(np.nan)
            
            # Color by player: blue for WHITE, red for BLACK
            colors.append('blue' if move['player'] == 'WHITE' else 'red')
        
        # Find winner from last move or metrics
        # Try to determine winner from the episode number
        winner_idx = None
        for i, ep in enumerate(log.metrics['episode']):
            if ep == episode:
                winner_idx = i
                break
        
        if winner_idx is not None:
            winner = log.metrics['winner'][winner_idx]
            winner_str = "WHITE won" if winner == 0 else "BLACK won"
            final_v_expected = 1.0 if winner == 0 else 0.0
        else:
            winner_str = "Unknown"
            final_v_expected = None
        
        ax = axes[idx]
        
        # Plot V(s) trajectory
        ax.scatter(move_nums, v_values, c=colors, alpha=0.6, s=30)
        ax.plot(move_nums, v_values, 'gray', alpha=0.3, linewidth=1)
        
        # Add expected final value line
        if final_v_expected is not None:
            ax.axhline(y=final_v_expected, color='green', linestyle='--', 
                      alpha=0.5, label=f'Expected final V={final_v_expected}')
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_title(f'Episode {episode} ({winner_str})\n{len(moves)} moves')
        ax.set_xlabel('Move Number')
        ax.set_ylabel('V(s) - Predicted WHITE win probability')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='WHITE move'),
                         Patch(facecolor='red', label='BLACK move')]
        ax.legend(handles=legend_elements, loc='lower right')
    
    plt.suptitle('Value Function V(s) Throughout Games at Different Training Stages\n'
                 '(Should converge toward actual outcome in later games)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./analysis/value_trajectories.png', dpi=150)
    plt.show()
    
    print("\n✅ Value trajectories saved to ./analysis/value_trajectories.png")
    
    # Also print some statistics
    print("\n" + "-"*60)
    print("VALUE PREDICTION ANALYSIS:")
    print("-"*60)
    
    for game_idx in game_indices[:4]:
        game = log.game_replays[game_idx]
        episode = game['episode']
        moves = game['moves']
        
        v_values = [m.get('v_current') for m in moves if m.get('v_current') is not None]
        
        if v_values:
            # Find winner
            winner_idx = None
            for i, ep in enumerate(log.metrics['episode']):
                if ep == episode:
                    winner_idx = i
                    break
            
            if winner_idx is not None:
                winner = log.metrics['winner'][winner_idx]
                actual_outcome = 1.0 if winner == 0 else 0.0
                
                # How close was the final prediction?
                final_v = v_values[-1]
                prediction_error = abs(final_v - actual_outcome)
                
                # Average V throughout game
                avg_v = np.mean(v_values)
                
                print(f"\nEpisode {episode}:")
                print(f"  Winner: {'WHITE' if winner == 0 else 'BLACK'}")
                print(f"  Avg V(s) during game: {avg_v:.3f}")
                print(f"  Final V(s) prediction: {final_v:.3f}")
                print(f"  Actual outcome: {actual_outcome:.1f}")
                print(f"  Final prediction error: {prediction_error:.3f}")
else:
    print("No game replays logged! Enable logging and run TD0.py again.")
