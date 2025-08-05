# Backgammon Gymnasium Environment

A modern backgammon environment compatible with the latest Gymnasium framework (v0.28+), based on the original [gym-backgammon](https://github.com/dellalibera/gym-backgammon) by Alessio Della Libera.

## Changes Made

### Gymnasium Compatibility Updates
- ✅ **Fixed missing `action_space`**: Added required `Discrete` action space for Gymnasium compatibility
- ✅ **Updated `reset()` method**: Now accepts `seed` and `options` parameters as required by modern Gymnasium
- ✅ **Fixed `step()` method**: Returns proper `(observation, reward, done, info)` format with info as dictionary
- ✅ **Updated `render()` method**: Removed deprecated `mode` parameter for Gymnasium compatibility
- ✅ **Fixed metadata format**: Changed from `'render.modes'` to `'render_modes'`

### Environment Fixes
- ✅ **OrderEnforcing wrapper compatibility**: Updated random agent example to work with Gymnasium's environment wrappers
- ✅ **Info dictionary**: Winner information now properly passed in `info['winner']` field

## Installation

```bash
git clone https://github.com/alexking100/backgammon-gymnasium.git
cd backgammon-gymnasium
pip install -e .
```

## Usage

### Random Agent Example
```bash
python examples/play_random_agent.py
```

This will run a game between two random agents with visual board rendering.

### Basic Environment Usage
```python
import gymnasium as gym
import gym_backgammon

env = gym.make('gym_backgammon:backgammon-v0')
unwrapped_env = env.unwrapped  # For custom methods

agent_color, first_roll, observation = env.reset()
actions = unwrapped_env.get_valid_actions(first_roll)
# ... your game logic
```

## Game Features

- **Full backgammon rules**: Including hitting, bearing off, and doubling cube
- **Visual rendering**: ASCII art board display
- **Gymnasium-compatible**: Works with modern RL frameworks
- **Random agents**: Example implementation for testing
- **Custom action space**: Handles complex backgammon moves

## Repository Structure

```
backgammon-gymnasium/
├── gym_backgammon/
│   ├── envs/
│   │   ├── backgammon.py      # Core game logic
│   │   ├── backgammon_env.py  # Gymnasium environment wrapper
│   │   └── rendering.py       # Visual rendering
│   └── __init__.py
├── examples/
│   └── play_random_agent.py   # Example random vs random game
├── tests/
└── setup.py
```

## Requirements

- Python 3.7+
- Gymnasium
- NumPy
- Pyglet (for visual rendering)

## Contributing

Feel free to submit issues and pull requests. This repository is actively maintained and updated for modern RL frameworks.

## License

MIT License - see [LICENSE](LICENSE) file

## Attribution

This project is based on [gym-backgammon](https://github.com/dellalibera/gym-backgammon) by Alessio Della Libera. The original work provided the foundation for the backgammon game logic and initial Gym environment structure.

### Original Copyright
```
MIT License
Copyright (c) 2019 Alessio Della Libera
```

## Roadmap

- [ ] Add more sophisticated AI agents
- [ ] Implement tournament play
- [ ] Add doubling cube mechanics
- [ ] Performance optimizations
- [ ] Enhanced visual rendering options