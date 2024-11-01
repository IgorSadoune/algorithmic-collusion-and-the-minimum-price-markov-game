#root/modules/agents/eg_agent.py

import numpy as np 
from typing import Dict, Any, List


class EpsilonGreedyAgent:
    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: Any
                 ): 
        
        self.agent_id = agent_id
        self.n_arms = action_dim
        self.epsilon = config['Epsilon Greedy']['epsilon']

        self.memory = []

        self.action_values = np.zeros(self.n_arms)  # Estimated values
        self.action_counts = np.zeros(self.n_arms)  # Count of each action taken
        self.n_optimal_pulls = 0
        self.n_exploration_pulls = 0
        self.n_total_pulls = 0

    def act(self, state=None, exploit: bool=False) -> int:
        if not exploit and np.random.random() < self.epsilon:
            self.n_exploration_pulls += 1
            self.chosen_arm = np.random.choice(self.n_arms)
        else:
            self.chosen_arm = np.argmax(self.action_values)
        return int(self.chosen_arm )

    def remember(self, state: np.ndarray, actions: List[bool], rewards: List[float], next_state: np.ndarray, dones: bool):
        self.memory.append((state, actions, rewards, next_state, dones))

    def learn(self):
        if len(self.memory) == 0:
            return None, None
        
        states, actions_, rewards_, next_states, dones = zip(*self.memory)
        self.memory = []

        reward = np.array(rewards_)[:, self.agent_id]

        self.action_counts[self.chosen_arm] += 1
        n = self.action_counts[self.chosen_arm]
        value = self.action_values[self.chosen_arm]
        self.action_values[self.chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

        # Regret
        optimal_reward = max(self.action_values)
        self.regret = float(optimal_reward - reward)
        
        if self.chosen_arm == np.argmax(self.action_values):
            self.n_optimal_pulls += 1
        self.n_total_pulls += 1

        self.action_value = float(self.action_values[1])

    def get_metrics(self) -> Dict[str, float]:
        metrics_dict = {"regret": self.regret,
                        "optimal pulls": self.n_optimal_pulls,
                        "action value": self.action_value
                        }
        return metrics_dict