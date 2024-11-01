#root/modules/agents/ucb_agent.py

import numpy as np 
from typing import Dict, Any, List


class UCBAgent:
    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: Any
                 ): 
        
        self.agent_id = agent_id
        self.n_arms = action_dim
        self.C = config['UCB']['C']

        self.counts = np.ones(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.memory = []

    def act(self, state=None, exploit: bool=False) -> int:
        n = sum(self.counts)
        confidence_bounds = self.values + np.sqrt((self.C * np.log(n)) / (self.counts + 1e-5))
        self.chosen_arm = np.argmax(confidence_bounds)
        return int(self.chosen_arm)

    def remember(self, state: np.ndarray, actions: List[bool], rewards: List[float], next_state: np.ndarray, dones: bool):
        self.memory.append((state, actions, rewards, next_state, dones))

    def learn(self):
        if len(self.memory) == 0:
            return None, None
        
        states, actions_, rewards_, next_states, dones = zip(*self.memory)
        self.memory = []
    
        reward = np.array(rewards_)[:, self.agent_id]

        self.counts[self.chosen_arm] += 1
        n = self.counts[self.chosen_arm]
        value = self.values[self.chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        
        # Regret
        optimal_reward = max(self.values)
        self.regret = float(optimal_reward - reward)
        self.values[self.chosen_arm] = new_value
        self.action_value = self.values[1]

    def get_metrics(self) -> Dict[str, float]:
        metrics_dict = {"regret": self.regret,
                        "action value": self.action_value
                        }
        return metrics_dict