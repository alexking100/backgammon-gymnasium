"""
TD(0) Learning for Backgammon - Training Script
===============================================

This script demonstrates TD(0) learning for backgammon, similar to TD-Gammon.

Run this file directly to train an agent:
    python examples/TD0.py

For analysis, import from TD0_classes:
    from examples.TD0_classes import TrainingLog, ValueNetwork, train_value_network
"""

import gymnasium as gym
import torch
import random

from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

# Import reusable classes and functions
from TD0_classes import (
    TrainingLog, 
    ValueNetwork, 
    choose_best_action,
    train_value_network, 
    test_agent
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_LOGGING = False
LOG_EVERY_N_EPISODES = 40
NUM_TRAINING_EPISODES = 2000
NUM_GAMES_TEST = 100


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # PART 1: GYMNASIUM BASICS DEMO
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 1: Gymnasium Basics")
    print("=" * 60)
    
    env = gym.make('gym_backgammon:backgammon-v0')
    unwrapped_env = env.unwrapped

    agent_color, first_roll, observation = env.reset()

    print(f"\n1. env.reset() returns:")
    print(f"   - agent_color: {agent_color} ({COLORS[agent_color]})")
    print(f"   - first_roll: {first_roll}")
    print(f"   - observation: array of {len(observation)} features")

    actions = unwrapped_env.get_valid_actions(first_roll)
    print(f"\n2. get_valid_actions(roll) returns:")
    print(f"   - {len(actions)} legal actions for roll {first_roll}")
    print(f"   - Example action: {list(actions)[0] if actions else 'None'}")

    if actions:
        action = random.choice(list(actions))
        observation_next, reward, done, info = env.step(action)
        
        print(f"\n3. env.step(action) returns:")
        print(f"   - observation: array of {len(observation_next)} features")
        print(f"   - reward: {reward}")
        print(f"   - done: {done}")
        print(f"   - info: {info}")

    env.close()

    # =========================================================================
    # PART 2: VALUE NETWORK DEMO
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 2: Building a Value Network")
    print("=" * 60)

    demo_net = ValueNetwork()
    print(f"\nNetwork architecture:")
    print(demo_net)

    sample_state = torch.FloatTensor(observation).unsqueeze(0)
    sample_value = demo_net(sample_state)
    print(f"\nSample prediction: V(initial_state) = {sample_value.item():.4f}")

    print("""
    How choose_best_action works:

        For each legal action:
            1. Simulate: "What if I take this action?"
            2. Evaluate: "How good is the resulting position?"
            3. Compare: "Is this better than my best so far?"
        
        Return the action that leads to the best position!
    """)

    # =========================================================================
    # PART 3: TRAINING
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 3: Training!")
    print("=" * 60)
    print(f"\nStarting training for {NUM_TRAINING_EPISODES} episodes...")
    print(f"Logging enabled: {ENABLE_LOGGING}")
    if ENABLE_LOGGING:
        print(f"Full game logs every {LOG_EVERY_N_EPISODES} episodes (plus first/last 5)")
    print()

    training_log = TrainingLog() if ENABLE_LOGGING else None
    
    trained_net, final_wins = train_value_network(
        num_episodes=NUM_TRAINING_EPISODES, 
        logger=training_log,
        log_every_n=LOG_EVERY_N_EPISODES
    )

    print(f"\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Final stats: WHITE {final_wins[WHITE]} wins, BLACK {final_wins[BLACK]} wins")
    print("=" * 60)

    # Save and show logging summary
    if ENABLE_LOGGING and training_log:
        training_log.print_summary()
        training_log.save("./examples/training_logs/training_log.json")
        
        if len(training_log.game_replays) >= 2:
            print("\n" + "="*60)
            print("SAMPLE GAME FROM EARLY TRAINING:")
            training_log.print_game_replay(0)
            
            print("\n" + "="*60)
            print("SAMPLE GAME FROM LATE TRAINING:")
            training_log.print_game_replay(-1)

    # =========================================================================
    # PART 4: TEST THE TRAINED AGENT
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 4: Testing the trained agent vs Random")
    print("=" * 60)

    test_results = test_agent(trained_net, num_games=NUM_GAMES_TEST)
    print(f"\nTrained WHITE vs Random BLACK over {NUM_GAMES_TEST} games:")
    print(f"  WHITE (trained): {test_results[WHITE]} wins ({test_results[WHITE]/NUM_GAMES_TEST*100:.0f}%)")
    print(f"  BLACK (random):  {test_results[BLACK]} wins ({test_results[BLACK]/NUM_GAMES_TEST*100:.0f}%)")

    print("\nDone! 🎲")
