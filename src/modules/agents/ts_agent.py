#src/modules/agents/ts_agent.py

import numpy as np 
from typing import Dict, Any, List
from scipy.stats import beta

class ThompsonSamplingAgent:
    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: Any
                 ): 
        
        self.agent_id = agent_id
        self.n_arms = action_dim

        self.alpha = np.ones(action_dim)  # Initialize alpha to 1 for each arm
        self.beta = np.ones(self.n_arms)   # Initialize beta to 1 for each arm
        self.cumulative_regret = 0.0     
        self.memory = []

    def act(self, state=None, exploit: bool=False) -> int:
        # Sample from the Beta distribution for each arm
        sampled_values = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        self.chosen_arm= np.argmax(sampled_values)
        return int(self.chosen_arm)

    def remember(self, state: np.ndarray, actions: List[bool], rewards: List[float], next_state: np.ndarray, dones: bool):
        self.memory.append((state, actions, rewards, next_state, dones))

    def learn(self):
        if len(self.memory) == 0:
            return None, None
        
        states, actions_, rewards_, next_states, dones = zip(*self.memory)
        self.memory = []
    
        reward = np.array(rewards_)[:, self.agent_id]

        # Regret
        self.alpha[self.chosen_arm] += reward
        self.beta[self.chosen_arm] += 1 - reward
        self.cumulative_regret += float(max(self.alpha / (self.alpha + self.beta)) - self.alpha[self.chosen_arm] / (self.alpha[self.chosen_arm] + self.beta[self.chosen_arm]))

    def get_metrics(self) -> Dict[str, float]:
        metrics_dict = {"cumulative_regret": self.cumulative_regret}
        return metrics_dict