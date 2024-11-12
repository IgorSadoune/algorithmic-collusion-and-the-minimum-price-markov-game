#src/modules/mpmg.py

'''
Minimum Price Markov Game modular environment.
'''

# Dependencies 
import numpy as np
import random
import itertools
from typing import Tuple, List, Dict, Optional


class MPMGEnv(object):
    '''
    Default configuration: 
    ----------------------

        - 2-player homogeneous MPMG with common binary action space
        - collusive bids are 30% higher
    '''

    def __init__(
        self, 
        n_agents: int = 2, 
        action_size: int = 2, 
        sigma_beta: float = 0.0, 
        alpha: float = 1.3
    ):
        super(MPMGEnv, self).__init__()

        # Collusive potential parameters
        self.n_agents = n_agents
        self.sigma_beta = sigma_beta
        self.alpha = alpha

        # Internal state action variables
        self.action_size = action_size
        self.joint_action_size = action_size ** self.n_agents
        self.beta_size = self.n_agents
        self.state_size = self.n_agents + self.joint_action_size + self.beta_size
        self.state_space = {
            'action_frequencies': None,
            'joint_action_frequencies': None,
            'beta_parameters': None
        }

    @staticmethod
    def _get_joint_action_code(actions: List[int]) -> int:
        action_code = 0
        for action in actions:
            action_code = (action_code << 1) | action
        return action_code
    
    def _set_seed(self, seed: Optional[int]):
        '''
        Allow modular seeding for controlled experiments. Seeds built-in Python and NumPy random processes.
        '''
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_power_parameters(self) -> None:
        '''
        Generates market shares as power parameters based on market heterogeneity.
        '''
        # Homogeneous agents
        if self.sigma_beta == 0:
            self.beta_parameters = np.ones(self.n_agents) / self.n_agents
        # Heterogeneous agents
        else:
            beta = np.abs(np.random.normal(1 / self.n_agents, self.sigma_beta, self.n_agents))
            self.beta_parameters = beta / np.sum(beta)

    def _update_action_frequencies(self, actions: List[int]) -> None:
        for agent_id, action in enumerate(actions): 
            if action == 1:
                self.action_counts[agent_id] += 1
        self.action_frequencies = self.action_counts / self.iteration

    def _update_joint_action_frequencies(self, actions: List[int]) -> None:
        index = self._get_joint_action_code(actions)
        self.joint_action_counts[index] += 1
        self.joint_action_frequencies = self.joint_action_counts / self.iteration

    def _update_beta(self) -> None:
        '''
        Placeholder method for future beta parameter updates.
        '''
        pass

    def _get_immediate_rewards(self, actions: List[int]) -> np.ndarray:
        '''
        Follows Minimum Price Game (MPG) payoff structure.
        '''
        mask_defect = np.array([action == 0 for action in actions])  # Defection mask
        if mask_defect.sum() == 0:  # All cooperate
            rewards = ((1 - self.beta_parameters) * self.beta_parameters) * self.alpha
        else:
            beta_omega = mask_defect.dot(self.beta_parameters)
            rewards = ((1 - self.beta_parameters) * (self.beta_parameters / beta_omega)) * mask_defect
        return rewards

    def _get_state(self) -> Dict[str, np.ndarray]:
        '''
        Observation space can be incremented here.
        '''
        self.state_space['action_frequencies'] = self.action_frequencies
        self.state_space['joint_action_frequencies'] = self.joint_action_frequencies
        self.state_space['beta_parameters'] = self.beta_parameters
        # self.state_space['additional_variable'] = self.additional_variable
        return self.state_space

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        '''
        Reset env with unbiased frequencies. Returns initial state.
        '''
        self._set_seed(seed)
        self._get_power_parameters()
        self.iteration = 1  # not 0 because it's a counter
        self.action_counts = np.zeros(self.n_agents)
        self.joint_action_counts = np.zeros(self.joint_action_size)

        # Initialize state with unbiased frequencies (plays each joint action exactly once)
        joint_actions = list(itertools.product(range(self.action_size), repeat=self.n_agents))
        for actions in joint_actions:
            _, _, _ = self.step(list(actions))
        self._get_state()

        return self.state_space
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, dict, bool]:
        '''
        Executes a single step in the environment.
        '''
        # Update internal state variables
        self._update_action_frequencies(actions)
        self._update_joint_action_frequencies(actions)
        # self.beta_parameters = self._update_beta()

        # Get immediate rewards
        immediate_rewards = self._get_immediate_rewards(actions)

        # Next state
        next_state = self._get_state()

        # Update counters
        self.iteration += 1

        return immediate_rewards, next_state, True  # 'True' indicates the 'done' flag
