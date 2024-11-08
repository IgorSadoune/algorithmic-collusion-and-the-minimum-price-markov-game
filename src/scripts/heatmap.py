#src/scripts/heatmap.py

import yaml
import os
import sys

# Add src directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.convergence_rates import ConvergenceRates

# Load the config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Save heatmap
heatmap = ConvergenceRates(config)
heatmap.main()

