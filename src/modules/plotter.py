import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class Plotter:
    def __init__(self, config: Dict[str, Any]) -> None:

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.metrics_dir = os.path.join(root, config['common']['metrics_path'])
        self.plot_dir = os.path.join(root, config['common']['plots_path'])
        self.convergence_scores_path = os.path.join(self.metrics_dir, 'convergence_scores.json')
        self.average_last_jaf_path = os.path.join(self.metrics_dir, 'average_last_jaf.json')
        self.num_repeats = config['common']['num_repeats']
        self.num_episodes = config['common']['num_episodes']
        self.convergence_scores = {}
        self.average_last_jaf = {}

    def _extract_metrics(self, filename: str) -> None:

        file_path = os.path.join(self.metrics_dir, filename)
        with open(file_path, 'r') as f:
            file = json.load(f)

        self.joint_action_size = len(file[0]['joint_action_frequencies'])
        joint_action_frequencies = np.zeros((self.num_repeats, self.num_episodes, self.joint_action_size))
        
        # Initialize a dictionary to hold agent metrics
        agent_metrics = {}
        
        # Populate the arrays
        for item in file:
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
        
        self.experiment_id = re.sub(r'(_metrics|\.json)', '', os.path.basename(filename))

    def _get_convergence_scores(self) -> None:

        def compute_convergence_score(average_last_jaf: np.array) -> float:
            score = average_last_jaf[-1] - average_last_jaf[0]
            return float(score)
        
        average_last_jaf = np.mean(self.metrics['joint_action_frequencies'], axis=0)[self.num_episodes-1]
        self.average_last_jaf[self.experiment_id] = {'defect':average_last_jaf[0],
                                                    'cooperate': average_last_jaf[-1],
                                                    'other': 1. - (average_last_jaf[-1] + average_last_jaf[0])
                                                    }
        score = compute_convergence_score(average_last_jaf)
        self.convergence_scores[self.experiment_id] = score

    @staticmethod
    def min_max_scale(convergence_scores: List[float]) -> Dict[str, float]:
        scaler = MinMaxScaler(feature_range=(0,1))
        experiment_ids = list(convergence_scores.keys())
        scores = np.array(list(convergence_scores.values())).reshape(-1,1)
        scaled_scores = scaler.fit_transform(scores).flatten()
        return dict(zip(experiment_ids, scaled_scores))
    
    def _store_convergence_scores(self) -> None:
        with open(self.convergence_scores_path, 'w') as f:
            json.dump(self.convergence_scores, f, indent=4)
        with open(self.average_last_jaf_path, 'w') as f:
            json.dump(self.average_last_jaf, f, indent=4)

    def _plot_metric(self, metric_data: List[float], metric_name: str) -> None:

        plot_filename = f'{metric_name}_{self.experiment_id}_plot.png'
        metric_directory = metric_name
        plot_directory = os.path.join(self.plot_dir, metric_directory)

        plt.style.use('classic')
        plt.figure(figsize=(5, 5))

        # Average across repeats
        avg_metric = np.mean(metric_data, axis=0)
        if metric_name in ['joint_action_frequencies']:
            avg_metric_other = 1. - (avg_metric[:, 0] + avg_metric[:, 1])
            std_metric_otehr = np.std(avg_metric_other, axis=0)
        std_metric = np.std(metric_data, axis=0)
        x_values = np.arange(self.num_episodes)

        if metric_name in ['joint_action_frequencies']:
            plt.plot(x_values, avg_metric[:, 0], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, 0], marker = 's', linestyle='none', color='black', label='Defect')
            plt.fill_between(x_values, avg_metric[:, 0] - std_metric[:, 0], avg_metric[:, 0] + std_metric[:, 0], alpha=0.2, color='dimgrey')
            plt.plot(x_values, avg_metric[:, -1], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, -1], marker = 'o', linestyle='none', color='black', label='Cooperate')
            plt.fill_between(x_values, avg_metric[:, -1] - std_metric[:, -1], avg_metric[:, -1] + std_metric[:, -1], alpha=0.2, color='dimgrey')
            plt.plot(x_values, avg_metric_other, linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric_other[::5], marker = '^', linestyle='none', color='black', label='Other')
            plt.fill_between(x_values, avg_metric_other - std_metric_otehr, avg_metric_other + std_metric_otehr, alpha=0.2, color='dimgrey')
            if self.experiment_id in ['ucb_2_0.0']:
                plt.legend(loc='upper left')
        elif metric_name in ['q_values', 'policies']:
            plt.plot(x_values, avg_metric[:, 0], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, 0], marker = 's', linestyle='none', color='black', label='Defect')
            plt.fill_between(x_values, avg_metric[:, 0] - std_metric[:, 0], avg_metric[:, 0] + std_metric[:, 0], alpha=0.2, color='dimgrey')
            plt.plot(x_values, avg_metric[:, -1], linestyle = '-', color='black')
            plt.plot(x_values[::5], avg_metric[::5, -1], marker = 'o', linestyle='none', color='black', label='Cooperate')
            plt.fill_between(x_values, avg_metric[:, -1] - std_metric[:, -1], avg_metric[:, -1] + std_metric[:, -1], alpha=0.2, color='dimgrey')
            if self.experiment_id in ['d3qn_2_0.0', 'q_values']:
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

    def _plot_heatmap(self):
       # Use the sorted list of configurations from Nash-heavy to Pareto-heavy
        configs = ['ts_2_0.0',
                    'ts_2_0.5',
                    'eg_2_0.0',
                    'eg_2_0.5',
                    'mappo_2_0.0',
                    'mappo_2_0.5',
                    'ts_5_0.0',
                    'ts_5_0.5',
                    'eg_5_0.0',
                    'mappo_5_0.0',
                    'eg_5_0.5',
                    'mappo_5_0.5',
                    'd3qn_2_0.0',
                    'd3qnom_2_0.0',
                    'd3qn_2_0.5',
                    'd3qnom_2_0.5',
                    'ucb_5_0.5',
                    'd3qnom_5_0.5',
                    'd3qn_5_0.5',
                    'd3qnom_5_0.0',
                    'd3qn_5_0.0',
                    'ucb_5_0.0',
                    'ucb_2_0.5',
                    'ucb_2_0.0']   


        # Initialize a 1D array to hold the data for the heatmap
        heatmap_data = np.zeros((1, len(configs)))

        # Populate heatmap data with the corresponding convergence rates
        for key, value in self.convergence_scores.items():
            if key in configs:
                config_idx = configs.index(key)
                # Calculate a score that represents the balance between Nash and Pareto
                heatmap_data[0, config_idx] = value

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
            if not filename in ['convergence_scores.json', 'average_last_jaf.json']:
                self._extract_metrics(filename)
                self._get_convergence_scores()
                self._plot_metric(self.metrics['joint_action_frequencies'], 'joint_action_frequencies')
                for metric_name, metric_data in self.metrics['agent_metrics'].items():
                    self._plot_metric(metric_data, metric_name)
        self.convergence_scores = self.min_max_scale(self.convergence_scores)
        self._store_convergence_scores()
        self._plot_heatmap()
                    