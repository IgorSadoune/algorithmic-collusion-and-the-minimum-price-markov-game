#src/scripts/plot.py

import yaml
import os
import sys
import argparse 

# Load the config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Add src directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.plotter import HarryPlotter

# Parsing argument
parser = argparse.ArgumentParser(description='Plot configuration')
parser.add_argument('--file', type=str, default='all', help='if "all", will plot all the metrics based on all the files in the plots/ directory, otherwise use the given path only')
args = parser.parse_args()

metrics_directory = "src/metrics/"

if args.file == 'all':
    # Loop over all JSON files in the metrics directory
    for filename in os.listdir(metrics_directory):
        if filename.endswith('.json'):
            metrics_path = os.path.join(metrics_directory, filename)
            for mode in ["training", "evaluation"]:
                plotter = HarryPlotter(config=config, metrics_path=metrics_path, mode=mode)
                plotter.plot_all_metrics()
else:
    filename = args.file
    metrics_path = os.path.join(metrics_directory, filename)
    for mode in ["training", "evaluation"]:
        plotter = HarryPlotter(config=config, metrics_path=metrics_path, mode=mode)
        plotter.plot_all_metrics()