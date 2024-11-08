#src/modules/heatmap.py

import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class ConvergenceRates:
    def __init__(self, config: Dict[str, Any]):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.metrics_dir = os.path.join(root, config['common']['metrics_path'])
        plot_dir = os.path.join(root, config['common']['plots_path'])

        self.output_path = os.path.join(self.metrics_dir, 'convergence_rates.json')
        self.heatmap_path = os.path.join(plot_dir, 'heatmap.png')

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
                # Extract data for heatmap
        agents = ['mappo', 'd3qn', 'd3qnom', 'eg', 'ts', 'ucb']
        configs = ['2_0.0', '2_0.5', '5_0.0', '5_0.5']
        heatmap_data = np.zeros((len(configs), len(agents)))

        # Populate heatmap data with Pareto optimal convergence rates
        for key, value in self.output.items():
            agent, config = key.split('_', 1)
            if agent in agents and config in configs:
                config_idx = configs.index(config)
                agent_idx = agents.index(agent)
                heatmap_data[config_idx, agent_idx] = value[0]  # Pareto optimal rate

        # Modify agent and config labels
        agent_labels = ['MAPPO', 'D3QN', 'D3QN-OM', 'E-Greedy', 'Thompson Sampling', 'UCB']
        config_labels = [r'$n=2, \sigma(\beta)=0.0$', r'$n=2, \sigma(\beta)=0.5$', r'$n=5, \sigma(\beta)=0.0$', r'$n=5, \sigma(\beta)=0.5$']

        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='Oranges', xticklabels=agent_labels, yticklabels=config_labels, cbar_kws={'label': 'Pareto Optimal Convergence Rate (%)'})
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels diagonally
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.xlabel('Agent')
        plt.ylabel('Config')
        plt.title('Pareto Optimal Convergence Rates Heatmap')
        plt.tight_layout()
        plt.savefig(self.heatmap_path, dpi=300, bbox_inches='tight')

    def main(self) -> None:
        for filename in os.listdir(self.metrics_dir):
            if not filename in ['convergence_rates.json']:
                self._extract_convergence_rates(filename)
        self._store_convergence_rates()
        self._plot_heatmap()
                    

        