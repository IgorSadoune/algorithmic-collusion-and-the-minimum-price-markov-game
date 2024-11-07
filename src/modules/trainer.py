#src/modules/trainer.py

import json
import logging
import os
from tqdm import tqdm
import torch
import numpy as np
import random
from typing import Dict, Any, List
from modules.mpmg import MPMGEnv


class Trainer(MPMGEnv):
    def __init__(self,
                config: Dict[str, Any],
                num_agents: int,
                sigma_beta: float,
                agent: Dict[str, Any],
                device: Any
                ):
        
        super().__init__(n_agents=num_agents, sigma_beta=sigma_beta)

        # Initialize experiment
        self.num_agents = num_agents
        self.sigma_beta = sigma_beta
        self.device = device
        self.agent_name = list(agent.keys())[0]
        self.agent_class = list(agent.values())[0]
        self.experiment_id = f"{self.num_agents}_{self.sigma_beta}"
        self.agent_ids = [id for id in range(self.num_agents)]
        self.metrics_buffer = []

        # Config
        self.logs_path = config['common']['logs_path']
        self.metrics_path = config['common']['metrics_path']
        self.num_repeats = config['common']['num_repeats']
        self.num_episodes = config['common']['num_episodes']
        self.seed = config['common']['seed'] 
        self.config = config

        # Initialize centralized logging and metrics
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.logs_path = os.path.join(self.root, self.logs_path)
        self.metrics_path = os.path.join(self.root, self.metrics_path)
    
        os.makedirs(self.logs_path, exist_ok=True)
        self.log_filename = os.path.join(self.logs_path, f"{self.agent_name}_{self.experiment_id}_training_log.log")
        logging.basicConfig(
            filename=self.log_filename,
            filemode='w',  # Overwrite log file each time the script runs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        os.makedirs(self.metrics_path, exist_ok=True)
        self.metrics_filename = os.path.join(self.metrics_path, f"{self.agent_name}_{self.experiment_id}_metrics.json")
        
        # Overwrite the metrics file with an empty structure each time
        with open(self.metrics_filename, 'w') as file:
            json.dump({}, file)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"Random seed set to {seed}")

    def _log_metrics(self,
                    repeat: int,
                    episode: int,
                    actions: List[bool],
                    action_frequencies: List[float],
                    joint_action_frequencies: List[float],
                    rewards: List[float],
                    cumulative_return: float,
                    agent_metrics: Dict[str, List]
                    ):
        """Logs metrics to a JSON file incrementally with buffering."""

        metrics = {
            "repeat": repeat,
            "episode": episode,
            "actions": actions,
            "collusive_action_frequency": action_frequencies,
            "joint_action_frequencies": joint_action_frequencies,
            "rewards": rewards,
            "cumulative_return": cumulative_return,
            "agent_metrics": agent_metrics
        }

        # Add metrics to buffer
        self.metrics_buffer.append(metrics)

        # Write the entire buffer to file
        if (episode + 1) == self.num_episodes:
            with open(self.metrics_filename, 'w') as file:
                json.dump(self.metrics_buffer, file, indent=4)

    @staticmethod
    def _flatten_state(state_dict):
        return np.concatenate([v.flatten() for v in state_dict.values()])

    def train(self):
        self.logger.info(f"Initializing experiment {self.experiment_id} on {self.device}")
        # self.logger.info("Training configuration: %s", self.config[self.agent_name])

        for repeat in tqdm(range(self.num_repeats)):
            self.logger.info("Repeat %s", repeat)

            repeat_seed = self.seed + repeat
            self._set_seed(repeat_seed)

            # Initializing agents
            self.agents = [self.agent_class(agent_id=agent_id,
                                            state_dim=self.state_size,
                                            action_dim=self.action_size,
                                            config=self.config,
                                            device=self.device) for _, agent_id in enumerate(self.agent_ids)]
            self.logger.info(f"Initializing {self.num_agents} {self.agent_name} agents")

            # Initial state
            state_dict = self.reset(seed=repeat_seed)
            state = self._flatten_state(state_dict)

            # Training
            self.logger.info("Training")
            cumulative_return = 0.0
            for episode in range(self.num_episodes):
                self.logger.info("Episode %s", episode)

                actions = [agent.act(state) for agent in self.agents]
                rewards, next_state_dict, _ = self.step(actions)
                next_state = self._flatten_state(next_state_dict)

                # Transition
                state_dict = next_state_dict
                state = next_state

                # Agents' update 
                aggregated_metrics = {}
                for agent in self.agents:
                    agent.remember(state, actions, rewards, next_state, True)
                    agent.learn()

                    # Agent specific metrics
                    agent_metrics = agent.get_metrics()
                    for key, value in agent_metrics.items():
                        if key not in aggregated_metrics:
                            aggregated_metrics[key] = []
                        aggregated_metrics[key].append(value)
                
                # Track metrics
                action_frequencies = self.action_frequencies.tolist()
                joint_action_frequencies = self.joint_action_frequencies.tolist()
                rewards = rewards.tolist()
                cumulative_return += sum(rewards)
                self._log_metrics(repeat=repeat, 
                                  episode=episode, 
                                  actions=actions, 
                                  rewards=rewards,
                                  action_frequencies=action_frequencies,
                                  joint_action_frequencies=joint_action_frequencies,
                                  cumulative_return=cumulative_return,
                                  agent_metrics=aggregated_metrics)

        self.logger.info("End of the experiment")
