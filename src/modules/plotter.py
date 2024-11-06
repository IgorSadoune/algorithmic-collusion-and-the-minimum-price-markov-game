import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

class HarryPlotter:
    def __init__(self,
                 config: Dict[str, Any],
                 metrics_path: str,
                 mode: str = 'training'
                 ):

        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.plots_path = os.path.join(self.root, config['common']['plots_path'])
        self.metrics_path = metrics_path
        self.mode = mode

        with open(self.metrics_path, 'r') as f:
            self.data = json.load(f)

        # Extract dimensions for metrics
        self.num_repeats = len(set(entry['repeat'] for entry in self.data[self.mode]))
        self.num_episodes = len(set(entry['episode'] for entry in self.data[self.mode]))
        self.num_agents = len(self.data[self.mode][0]['actions'])
        self.joint_action_size = len(self.data[self.mode][0]['joint_action_frequencies'])

    def extract_metrics(self):
        metrics = {
            'agent_metrics': {},
            'joint_action_frequencies': None
        }

        # Initialize arrays for storing metrics
        for key in self.data[self.mode][0].keys():
            if key not in ['repeat', 'episode', 'actions', 'rewards', 'joint_action_frequencies', 'metrics']:
                metrics['agent_metrics'][key] = np.zeros((self.num_repeats, self.num_episodes, self.num_agents))
        metrics['joint_action_frequencies'] = np.zeros((self.num_repeats, self.num_episodes, self.joint_action_size))

        # Fill metrics
        for entry in self.data[self.mode]:
            repeat = entry['repeat']
            episode = entry['episode']

            for key, value in entry.items():
                if key in metrics['agent_metrics']:
                    for agent_idx, agent_value in enumerate(value):
                        metrics['agent_metrics'][key][repeat, episode, agent_idx] = agent_value
                elif key == 'joint action frequencies':
                    metrics['joint_action_frequencies'][repeat, episode] = value
                elif key == 'metrics':
                    for metric_key, metric_value in value['metrics'].items():
                        if isinstance(metric_value, list):
                            if metric_key not in metrics['agent_metrics']:
                                metrics['agent_metrics'][metric_key] = np.zeros((self.num_repeats, self.num_episodes, self.num_agents))
                            for agent_idx, agent_value in enumerate(metric_value):
                                metrics['agent_metrics'][metric_key][repeat, episode, agent_idx] = agent_value

        return metrics

    def plot_metric(self, metric_data, metric_name, xlabel='Episodes', ylabel=None):
        if ylabel is None:
            ylabel = " ".join(metric_name.split('_')).title()

        plt.style.use('ggplot')
        plt.figure(figsize=(5, 5))

        # Average across repeats
        avg_metric = np.mean(metric_data, axis=0)
        std_metric = np.std(metric_data, axis=1)
        x_values = np.arange(self.num_episodes)

        if metric_name == 'joint_action_frequencies':
            # Plot each joint action frequency separately
            plt.plot(x_values, avg_metric[:, 0], color='darkgrey', linestyle='-')
            plt.fill_between(x_values, avg_metric[:, 0] - std_metric[:, 0], avg_metric[:, 0] + std_metric[:, 0], alpha=0.1, color='dimgrey')
            plt.plot(x_values, avg_metric[:, -1], color='darkgrey', linestyle='--')
            plt.fill_between(x_values, avg_metric[:, -1] - std_metric[:, -1], avg_metric[:, -1] + std_metric[:, -1], alpha=0.1, color='dimgrey')
        elif metric_data.ndim == 3:  # agent-specific metrics
            for agent_idx in range(self.num_agents):
                plt.plot(x_values, avg_metric[:, agent_idx], color='darkgrey')
                plt.fill_between(x_values, avg_metric[:, agent_idx] - std_metric[:, agent_idx], avg_metric[:, agent_idx] + std_metric[:, agent_idx], alpha=0.1, color='dimgrey')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Constructing filename from input JSON
        base_filename = re.sub(r'(_metrics|\.json)', '', os.path.basename(self.metrics_path))
        mode_suffix = "training" if self.mode == 'training' else "evaluation"
        plot_filename = f'{base_filename}_{mode_suffix}_{metric_name}_plot.png'
        agent_directory = re.match(r'([^_]+)_', plot_filename).group(1)
        plot_directory = os.path.join(self.plots_path, agent_directory, mode_suffix)

        # Ensure plot directory exists
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plt.savefig(os.path.join(plot_directory, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_metrics(self):
        metrics = self.extract_metrics()

        # Plot agent-specific metrics
        for metric_name, metric_data in metrics['agent_metrics'].items():
            self.plot_metric(metric_data, metric_name)

        # Plot joint action frequencies
        if metrics['joint_action_frequencies'] is not None:
            self.plot_metric(metrics['joint_action_frequencies'], 'joint_action_frequencies')