# manual analysis script for Alex to look at the data

import sys
sys.path.insert(0, '.')  # Add project root to path

from examples.TD0_classes import TrainingLog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the saved training log (run TD0.py first!)
log = TrainingLog.load("./examples/training_logs/training_log.json")

game_replays = pd.DataFrame(log.game_replays)
print(game_replays.head())

initial_values = pd.DataFrame(log.initial_values)
print(initial_values.head())

metrics = pd.DataFrame(log.metrics)
print(metrics.head())