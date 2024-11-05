#src/scripts/plot.py

import os
import sys
import argparse 

# Add src directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.plotter import HarryPlotter

# Parsing argument
parser = argparse.ArgumentParser(description='Plot configuration')
parser.add_argument('--file', type=str, default='all', help='if "all", will plot all the metrics based on all the files in the plots/ directory, otherwise use the given path only')
args = parser.parse_args()

metrics_directory = "src/metrics/"
plot_directory = "src/plots/"

# Create plot directory if it doesn't exist
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

if args.file == 'all':
    # Loop over all JSON files in the metrics directory
    for file_name in os.listdir(metrics_directory):
        if file_name.endswith('.json'):
            metrics_path = os.path.join(metrics_directory, file_name)
            for mode in ["training", "evaluation"]:
                    plotter = HarryPlotter(metrics_path=metrics_path, plot_path=plot_directory, mode=mode)
                    plotter.plot_all_metrics()
else:
    file_name = args.file
    metrics_path = os.path.join(metrics_directory, file_name)
    for mode in ["training", "evaluation"]:
                    plotter = HarryPlotter(metrics_path=metrics_path, plot_path=plot_directory, mode=mode)
                    plotter.plot_all_metrics()