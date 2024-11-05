import json
import os
import numpy as np
import matplotlib.pyplot as plt

class HarryPlotter:
    def __init__(self, metrics_path: str, plot_path: str):
        self.metrics_path = metrics_path
        self.plot_path = plot_path

        with open(os.path.join(self.data_directory, file_name), 'r') as f:
            self.data = json.load(f)

    def extract_metric(self, metric_name):
        agent_metrics = {}
        num_repeats = len(self.data)

        for repeat in self.data:
            for agent_idx, agent_data in enumerate(repeat["episodes"]):
                if agent_idx not in agent_metrics:
                    agent_metrics[agent_idx] = []
                metric_values = [episode[metric_name] for episode in agent_data]
                if len(agent_metrics[agent_idx]) == 0:
                    agent_metrics[agent_idx] = np.array(metric_values)
                else:
                    agent_metrics[agent_idx] += np.array(metric_values)

        # Average across repeats
        for agent_idx in agent_metrics:
            agent_metrics[agent_idx] /= num_repeats

        return agent_metrics

    def plot_metric(self, metric_name, title=None, xlabel='Episodes', ylabel=None):
        """
        Plot a specific metric for each agent.

        Parameters:
        - metric_name (str): The name of the metric to plot.
        - title (str): The title of the plot.
        - xlabel (str): The label for the x-axis.
        - ylabel (str): The label for the y-axis.
        """
        metric_data = self.extract_metric(metric_name)

        if ylabel is None:
            ylabel = metric_name.upper()

        plt.style.use('ggplot')
        plt.figure(figsize=(5, 5))

        for agent_idx, values in metric_data.items():
            plt.plot(values, label=f'Agent {agent_idx + 1}')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title if title else f'{metric_name} across episodes')
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().set_facecolor('#f0f0f0')
        plt.savefig(os.path.join(self.plot_path, f'{metric_name}_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Usage example:
# plotter = RLMetricPlotter(data_directory="/path/to/metrics/json/files", plot_path="/path/to/save/plots")
# plotter.plot_metric("reward", title="Average Reward for Each Agent")
# plotter.plot_metric("action frequencies", title="Average Action Frequencies for Each Agent")
