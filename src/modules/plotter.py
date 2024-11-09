import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

class HarryPlotter:
    def __init__(self,
                 config: Dict[str, Any],
                 metrics_path: str
                 ):

        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.plots_path = os.path.join(self.root, config['common']['plots_path'])
        self.metrics_path = metrics_path

        with open(self.metrics_path, 'r') as f:
            self.data = json.load(f)

        # Extract dimensions for metrics
        self.num_repeats = len(set(entry['repeat'] for entry in self.data))
        self.num_episodes = len(set(entry['episode'] for entry in self.data))
        self.joint_action_size = len(self.data[0]['joint_action_frequencies'])

    def _extract_metrics(self):

        # Initialize array for joint action frequencies
        joint_action_frequencies = np.zeros((self.num_repeats, self.num_episodes, self.joint_action_size))
        
        # Initialize a dictionary to hold agent metrics
        agent_metrics = {}
        
        # Populate the arrays
        for item in self.data:
            repeat = item['repeat']
            episode = item['episode']
            
            # Joint action frequencies
            joint_action_frequencies[repeat, episode, :] = item['joint_action_frequencies']
            
            # Agent metrics - dynamically handle all keys and take the average across agents
            for key, values in item['agent_metrics'].items():
                if key not in agent_metrics:
                    if key in ['q_values', 'policies']:
                        # Average q_values and policiy values for each action across agents
                        agent_metrics[key] = np.zeros((self.num_repeats, self.num_episodes, 2))
                    else:
                        agent_metrics[key] = np.zeros((self.num_repeats, self.num_episodes))

                if key in ['q_values', 'policies']:
                    agent_metrics[key][repeat, episode] = np.mean(values, axis=0)
                else:    
                    agent_metrics[key][repeat, episode] = np.mean(values)
                
        self.metrics = {'joint_action_frequencies': joint_action_frequencies,
                   'agent_metrics': agent_metrics}

    def _plot_metric(self, metric_data, metric_name):

        # Constructing filename from input JSON
        experiment_id = re.sub(r'(_metrics|\.json)', '', os.path.basename(self.metrics_path))
        plot_filename = f'{metric_name}_{experiment_id}_plot.png'
        metric_directory = metric_name
        plot_directory = os.path.join(self.plots_path, metric_directory)

        plt.style.use('classic')
        plt.figure(figsize=(5, 5))

        # Average across repeats
        avg_metric = np.mean(metric_data, axis=0)
        std_metric = np.std(metric_data, axis=0)
        x_values = np.arange(self.num_episodes)

        if metric_name in ['joint_action_frequencies', 'q_values', 'policies']:
            plt.plot(x_values, avg_metric[:, 0], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, 0], marker = 's', linestyle='none', color='black', label='Defect')
            plt.fill_between(x_values, avg_metric[:, 0] - std_metric[:, 0], avg_metric[:, 0] + std_metric[:, 0], alpha=0.2, color='dimgrey')
            plt.plot(x_values, avg_metric[:, -1], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, -1], marker = 'o', linestyle='none', color='black', label='Cooperate')
            plt.fill_between(x_values, avg_metric[:, -1] - std_metric[:, -1], avg_metric[:, -1] + std_metric[:, -1], alpha=0.2, color='dimgrey')
            if (experiment_id, metric_name) in [('ucb_2_0.0', 'joint_action_frequencies'), ('d3qn_2_0.0', 'q_values')]:
                plt.legend(loc='upper left')
        elif metric_name in ['loss', 'actor_loss', 'critic_loss', 'cumulative_regret']:
            plt.plot(x_values, avg_metric, color='black')
            plt.fill_between(x_values, avg_metric - std_metric, avg_metric + std_metric, alpha=0.2, color='dimgrey')

        if metric_name == 'cumulative_regret':
            plt.ylim(-1.0, 4.0)
        elif metric_name == 'loss':
            plt.ylim(-0.1, 0.3)
            plt.xlim(0, 20)
        elif metric_name in ['policies', 'joint_action_frequencies']:
            plt.ylim(0.0, 1.0)
        elif metric_name == 'q_values':
            plt.ylim(0.0, 0.6)
        elif metric_name == 'actor_loss':
            plt.ylim(-0.2, 0.1)
        else: # critic_loss
            plt.ylim(0.0, 0.1)

        # Ensure plot directory exists
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plt.savefig(os.path.join(plot_directory, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_metrics(self):
        self._extract_metrics()

        # Plot agent-specific metrics
        for metric_name, metric_data in self.metrics['agent_metrics'].items():
            self._plot_metric(metric_data, metric_name)

        # Plot joint action frequencies
        if self.metrics['joint_action_frequencies'] is not None:
            self._plot_metric(self.metrics['joint_action_frequencies'], 'joint_action_frequencies')