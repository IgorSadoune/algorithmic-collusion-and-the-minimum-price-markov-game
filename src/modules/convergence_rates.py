#src/modules/heatmap.py

import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, Any

class ConvergenceRates:
    def __init__(self, config: Dict[str, Any]):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.metrics_dir = os.path.join(root, config['common']['metrics_path'])
        self.plot_dir = os.path.join(root, config['common']['plots_path'])

        self.output_path = os.path.join(self.metrics_dir, 'convergence_rates.json')

        self.num_repeats = config['common']['num_repeats']

        self.output = []

    def _extract_convergence_rates(self, filename: str) -> None:
        # Initialize counters for each outcome type
        pareto_count = 0
        nash_count = 0
        other_suboptimal_count = 0

        file_path = os.path.join(self.metrics_dir, filename)
        with open(file_path, 'r') as f:
            file = json.load(f)
            
        for item in file:
            if item['episode'] == 99:  # Last episode (100th episode, index 99)
                jaf = item['joint_action_frequencies']
                argmax_index = np.argmax(jaf)
                if argmax_index == 0 and jaf[argmax_index] > 0.5:
                    nash_count += 1
                elif argmax_index == len(jaf) - 1 and jaf[argmax_index] > 0.5:
                    pareto_count += 1
                else:
                    other_suboptimal_count += 1

        # Calculate percentages
        pareto_rate = round(pareto_count / self.num_repeats, 2)
        nash_rate = round(nash_count / self.num_repeats, 2)
        suboptimal_rate = round(other_suboptimal_count / self.num_repeats, 2)

        # Store the results in a dictionary
        experiment_id = re.sub(r'(_metrics|\.json)', '', os.path.basename(filename))
        convergence_rates_dict = {
            experiment_id: [pareto_rate, nash_rate, suboptimal_rate]
            }
        
        self.output.append(convergence_rates_dict)

    def _store_convergence_rates(self) -> None:
        with open(self.output_path, 'w') as f:
            json.dump(self.output, f, indent=4)

    def _plot_heatmap(self):
       # Use the sorted list of configurations from Nash-heavy to Pareto-heavy
        configs = [
            "ts_2_0.0", "eg_2_0.0", "eg_2_0.5", "ts_2_0.5", "mappo_2_0.0", "mappo_2_0.5",
            "d3qn_2_0.5", "d3qnom_2_0.5", "d3qn_2_0.0", "d3qnom_5_0.0", "d3qnom_5_0.5", "ts_5_0.0",
            "ts_5_0.5", "d3qnom_2_0.0", "eg_5_0.0", "d3qn_5_0.0",
            "mappo_5_0.0", "eg_5_0.5", "ucb_5_0.0", "mappo_5_0.5",
            "d3qn_5_0.5", "ucb_5_0.5", "ucb_2_0.5", "ucb_2_0.0"
        ]

        # Initialize a 1D array to hold the data for the heatmap
        heatmap_data = np.zeros((1, len(configs)))

        # Populate heatmap data with the corresponding convergence rates
        for dict in self.output:
            for key, value in dict.items():
                if key in configs:
                    config_idx = configs.index(key)
                    # Calculate a score that represents the balance between Nash and Pareto
                    score = value[0] - value[1]  # Pareto Rate - Nash Rate
                    heatmap_data[0, config_idx] = score

        # Define the custom colormap from blue to grey to red
        colors = ["#87CEFA", "#B2FFFF", "#ffd1dc"]
        nash_pareto_cmap = LinearSegmentedColormap.from_list("nash_pareto_cmap", colors)

        # Set up the style for a cleaner plot
        sns.set_theme(style="whitegrid")

        # Create formatted tick labels
        formatted_ticks = []
        for config in configs:
            agent, params = config.split('_', 1)
            n, beta = params.split('_')
            formatted_tick = f"{agent.upper()}, n={n}, \u03c3(\u03b2)={beta}"
            formatted_ticks.append(formatted_tick)

        # Create a 1D heatmap with the custom color map
        plt.figure(figsize=(15, 2))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap=nash_pareto_cmap,  # Custom colormap
            center=0,
            xticklabels=formatted_ticks,
            yticklabels=[''],
            linewidths=0.5,
            linecolor='white',
            cbar=False
        )

        # Add labels "Nash" and "Pareto" at the ends of the heatmap
        plt.text(-0.5, 0, 'Nash', fontsize=10, weight='bold', color='#87CEFA', ha='center')
        plt.text(len(configs) + 0.5, 0, 'Pareto', fontsize=10, weight='bold', color='#ffd1dc', ha='center')

        # Customize the appearance of axis labels and ticks
        plt.xticks(rotation=45, ha='right', fontsize=8)  # Diagonal labels for better readability
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def main(self) -> None:
        for filename in os.listdir(self.metrics_dir):
            if not filename in ['convergence_rates.json']:
                self._extract_convergence_rates(filename)
        self._store_convergence_rates()
        self._plot_heatmap()
                    

        