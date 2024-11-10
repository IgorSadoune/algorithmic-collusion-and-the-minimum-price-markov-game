#src/scripts/plot.py

import yaml
import os
import sys

# Load the config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Add src directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.plotter import Plotter

# Plot metrics and heatmap
plotter = Plotter(config=config)
plotter.main()
