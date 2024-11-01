#root/modules/trainer.py

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

        # Config
        self.logs_path = config['common']['logs_path']
        self.metrics_path = config['common']['metrics_path']
        self.num_repeats = config['common']['num_repeats']
        self.num_episodes = config['common']['num_episodes']
        self.num_eval_episodes = config['common']['num_eval_episodes']
        self.log_interval = config['common']['log_interval']
        self.config = config 

        # Initialize centralized logging and metrics
        self.metrics_buffer = []
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        with open(self.metrics_filename, mode='w') as file:
            json.dump({"training": [], "evaluation": []}, file)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"Random seed set to {seed}")

    def _log_metrics(self,
                    mode: str,
                    repeat: int,
                    episode: int,
                    actions: List[bool],
                    action_frequencies: List[float],
                    joint_action_frequencies: List[float],
                    rewards: List[float],
                    **kwargs):
            """Logs metrics to a JSON file incrementally with buffering.
            Mode should be 'training' or 'evaluation'.
            """

            # Convert any non-serializable input in kwargs to a serializable format
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    kwargs[key] = value.tolist()
                elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    kwargs[key] = str(value)  # Convert non-serializable types to string

            metrics = {
                "repeat": repeat,
                "episode": episode,
                "actions": actions,
                "rewards": rewards,
                "action frequencies": action_frequencies,
                "joint action frequencies": joint_action_frequencies,
                "metrics": kwargs
            }

            # Add metrics to buffer
            self.metrics_buffer.append(metrics)

            # Write buffer to file every `log_interval` steps
            if (episode + 1) % self.log_interval == 0 or episode == repeat - 1:
                with open(self.metrics_filename, mode='r+') as file:
                    try:
                        data = json.load(file)
                        if not isinstance(data, dict) or "training" not in data or "evaluation" not in data:
                            data = {"training": [], "evaluation": []}
                    except json.JSONDecodeError:
                        data = {"training": [], "evaluation": []}

                    if mode == "training":
                        data["training"].extend(self.metrics_buffer)
                    elif mode == "evaluation":
                        data["evaluation"].extend(self.metrics_buffer)
                    else:
                        raise ValueError("Mode must be 'training' or 'evaluation'")

                    # Write updated data back to file
                    file.seek(0)
                    json.dump(data, file, indent=4)
                    file.truncate()

                # Clear the buffer after writing
                self.metrics_buffer = []

    @staticmethod
    def _flatten_state(state_dict):
        return np.concatenate([v.flatten() for v in state_dict.values()])

    def train(self):
        self.logger.info(f"Initializing experiment {self.experiment_id} on {self.device}")
        # self.logger.info("Training configuration: %s", self.config[self.agent_name])

        for repeat in tqdm(range(self.num_repeats)):
            self.logger.info("Repeat %s", repeat)

            repeat_seed = self.config['common']['seed'] + repeat
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

                    # Aggregate agent specific metrics
                    agent_metrics = agent.get_metrics()
                    for key, value in agent_metrics.items():
                        if key not in aggregated_metrics:
                            aggregated_metrics[key] = []
                        aggregated_metrics[key].append(value)
                
                # Track metrics
                action_frequencies = self.action_frequencies.tolist()
                joint_action_frequencies = self.joint_action_frequencies.tolist()
                rewards = rewards.tolist()
                self._log_metrics(mode='training', 
                                  repeat=repeat, 
                                  episode=episode, 
                                  actions=actions, 
                                  rewards=rewards,
                                  action_frequencies=action_frequencies,
                                  joint_action_frequencies=joint_action_frequencies,
                                  metrics=aggregated_metrics)
            
            # Evaluation
            self.logger.info("Evaluation")
            cumulative_returns = 0.0
            state_dict = self.reset(seed=repeat_seed)
            state = self._flatten_state(state_dict)
            for eval_episode in range(self.num_eval_episodes):
                actions = [agent.act(state, exploit=True) for agent in self.agents]
                rewards, next_state_dict, _ = self.step(actions)
                
                # Next state
                state_dict = next_state_dict
                state = self._flatten_state(state_dict)

                # Track metrics
                cumulative_returns += sum(rewards)
                action_frequencies = self.action_frequencies.tolist()
                joint_action_frequencies = self.joint_action_frequencies.tolist()
                rewards = rewards.tolist()
                self._log_metrics(mode='evaluation', 
                                  repeat=repeat, 
                                  episode=eval_episode, 
                                  actions=actions, 
                                  rewards=rewards,
                                  action_frequencies=action_frequencies,
                                  joint_action_frequencies=joint_action_frequencies,
                                  metrics={"cumulative return": cumulative_returns})

        self.logger.info("End of the experiment")