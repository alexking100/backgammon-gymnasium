#!/usr/bin/env python3
"""
Step-by-step backgammon game walkthrough
This demonstrates exactly how each function works in the game flow
"""

import gymnasium as gym
import gym_backgammon
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
import random
import time

def explain_function(func_name, description, details=None):
    """Helper to explain what each function does"""
    print(f"\n🔧 FUNCTION: {func_name}")
    print(f"📋 PURPOSE: {description}")
    if details:
        print(f"📝 DETAILS: {details}")
    print("-" * 60)

def wait_for_user():
    """Wait for user to press enter to continue"""
    input("\n⏯️  Press Enter to continue...")

def main():
    print("=" * 80)
    print("🎯 BACKGAMMON STEP-BY-STEP WALKTHROUGH")
    print("=" * 80)
    print("This demo will show you exactly how each function works!")
    
    wait_for_user()
    
    # ================================================================
    # STEP 1: ENVIRONMENT INITIALIZATION
    # ================================================================
    explain_function(
        "gym.make('gym_backgammon:backgammon-v0')",
        "Creates the backgammon environment",
        "Calls BackgammonEnv.__init__() which sets up the game board, action/observation spaces"
    )
    
    env = gym.make('gym_backgammon:backgammon-v0')
    unwrapped_env = env.unwrapped
    
    print("✅ Environment created!")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space shape: {env.observation_space.shape}")
    
    wait_for_user()
    
    # ================================================================
    # STEP 2: GAME RESET - INITIAL DICE ROLL
    # ================================================================
    explain_function(
        "env.reset()",
        "Initializes a new game and determines who goes first",
        "Lines 68-90 in backgammon_env.py: Rolls dice until different, higher roll goes first"
    )
    
    print("🎲 Rolling initial dice to determine who goes first...")
    agent_color, first_roll, observation = env.reset(seed=42)  # Seed for reproducibility
    
    print(f"   - Initial dice roll: {first_roll}")
    print(f"   - Starting player: {agent_color} ({TOKEN[agent_color]} - {COLORS[agent_color]})")
    print(f"   - Why negative roll? White moves in negative direction (23→0), Black positive (0→23)")
    
    wait_for_user()
    
    # ================================================================
    # STEP 3: BOARD RENDERING
    # ================================================================
    explain_function(
        "env.render()",
        "Displays the current board state",
        "Calls game.render() in backgammon.py lines 1345-1363: ASCII art representation"
    )
    
    print("🎨 Rendering initial board state:")
    env.render()
    
    print("\n📖 BOARD EXPLANATION:")
    print("   - Points 0-23: Standard backgammon board positions")
    print("   - X = White checkers, O = Black checkers")
    print("   - BAR: Hit checkers must re-enter from here")
    print("   - OFF: Checkers that have been borne off (won)")
    print("   - White's home: points 0-5, Black's home: points 18-23")
    
    wait_for_user()
    
    # ================================================================
    # STEP 4: GETTING VALID MOVES
    # ================================================================
    explain_function(
        "unwrapped_env.get_valid_actions(roll)",
        "Generates all legal moves for the current dice roll",
        "Calls get_valid_plays() in backgammon.py line 1382: Complex logic for move generation"
    )
    
    print(f"🎯 Finding valid moves for roll {first_roll}...")
    actions = unwrapped_env.get_valid_actions(first_roll)
    
    print(f"   - Number of valid actions: {len(actions)}")
    print("   - Sample actions (first 5):")
    for i, action in enumerate(list(actions)[:5]):
        print(f"     {i+1}. {action}")
        if action:
            for move in action:
                if move:
                    src, dest = move
                    print(f"        Move checker from point {src} to point {dest}")
    
    print("\n📖 ACTION FORMAT:")
    print("   - Each action is a tuple of moves: ((src1, dest1), (src2, dest2), ...)")
    print("   - Each move is (source_point, destination_point)")
    print("   - None means no move (when fewer dice can be used)")
    
    wait_for_user()
    
    # ================================================================
    # STEP 5: MOVE SELECTION AND EXECUTION
    # ================================================================
    explain_function(
        "env.step(action)",
        "Executes the chosen move and updates game state",
        "Calls execute_play() to move checkers, then returns new observation/reward/done/info"
    )
    
    # Choose a random action for demo
    if actions:
        chosen_action = random.choice(list(actions))
        print(f"🎮 Executing chosen action: {chosen_action}")
        
        if chosen_action:
            print("   📋 This action will:")
            for move in chosen_action:
                if move:
                    src, dest = move
                    direction = "towards home" if (agent_color == WHITE and dest < src) or (agent_color == BLACK and dest > src) else "away from home"
                    print(f"      - Move checker from point {src} to point {dest} ({direction})")
        
        # Execute the move
        observation_next, reward, done, info = env.step(chosen_action)
        winner = info.get('winner', None)
        
        print(f"   ✅ Move executed!")
        print(f"   - Reward: {reward}")
        print(f"   - Game done: {done}")
        print(f"   - Winner: {winner}")
        
        wait_for_user()
        
        # ================================================================
        # STEP 6: BOARD AFTER MOVE
        # ================================================================
        explain_function(
            "env.render() (after move)",
            "Shows the board state after the move",
            "Notice how the checkers have moved according to the executed action"
        )
        
        print("🎨 Board state after the move:")
        env.render()
        
        wait_for_user()
        
        # ================================================================
        # STEP 7: PLAYER SWITCHING
        # ================================================================
        explain_function(
            "unwrapped_env.get_opponent_agent()",
            "Switches to the other player's turn",
            "Changes current_agent from WHITE to BLACK or vice versa"
        )
        
        current_player = unwrapped_env.current_agent
        print(f"   - Current player before switch: {current_player} ({TOKEN[current_player]})")
        
        new_player = unwrapped_env.get_opponent_agent()
        print(f"   - Current player after switch: {new_player} ({TOKEN[new_player]})")
        
        wait_for_user()
        
        # ================================================================
        # STEP 8: DICE ROLLING FOR NEXT TURN
        # ================================================================
        explain_function(
            "agent.roll_dice()",
            "Generates dice roll for the new player's turn",
            "Random integers 1-6, with direction based on player (WHITE gets negative)"
        )
        
        # Simulate next player's dice roll
        if new_player == WHITE:
            next_roll = (-random.randint(1, 6), -random.randint(1, 6))
        else:
            next_roll = (random.randint(1, 6), random.randint(1, 6))
        
        print(f"🎲 {TOKEN[new_player]} ({COLORS[new_player]}) rolls: {next_roll}")
        print(f"   - Why this direction? {COLORS[new_player]} moves {'backwards (23→0)' if new_player == WHITE else 'forwards (0→23)'}")
        
        # Get valid actions for new roll
        next_actions = unwrapped_env.get_valid_actions(next_roll)
        print(f"   - Valid actions available: {len(next_actions)}")
        
    wait_for_user()
    
    # ================================================================
    # SUMMARY OF GAME FLOW
    # ================================================================
    print("\n" + "=" * 80)
    print("📚 COMPLETE GAME FLOW SUMMARY")
    print("=" * 80)
    
    flow_steps = [
        ("1. env.reset()", "Initialize game, roll for first player"),
        ("2. env.render()", "Display board state"),
        ("3. agent.roll_dice()", "Generate dice roll for current player"),
        ("4. get_valid_actions(roll)", "Find all legal moves"),
        ("5. agent.choose_action()", "Select move (random or strategic)"),
        ("6. env.step(action)", "Execute move, update game state"),
        ("7. env.render()", "Show updated board"),
        ("8. get_opponent_agent()", "Switch to other player"),
        ("9. Check if game done", "Someone won or max turns reached"),
        ("10. Repeat from step 3", "Continue until game ends")
    ]
    
    for step, description in flow_steps:
        print(f"   {step:<25} → {description}")
    
    print("\n🎯 KEY FUNCTIONS BREAKDOWN:")
    print("   📍 Rendering: backgammon.py:1345 - render()")
    print("   🎲 Dice: play_random_agent.py:22 - roll_dice()")
    print("   🎯 Valid moves: backgammon.py:1382 - get_valid_plays()")
    print("   🎮 Move execution: backgammon.py:1413 - execute_play()")
    print("   🔄 Game loop: play_random_agent.py:40 - main game loop")
    
    env.close()
    
    print("\n" + "=" * 80)
    print("🎉 DEMO COMPLETE!")
    print("You now understand exactly how each part of the backgammon engine works!")
    print("=" * 80)

if __name__ == "__main__":
    main()