#src/modules/agents/d3qn_agent.py

import torch
from collections import deque
from typing import List, Dict, Any
import numpy as np
import random


class DuelingDQN(torch.nn.Module):
    '''
    '''
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int],
                 value_stream_hidden_dims: List[int],
                 advantage_stream_hidden_dims: List[int]
                 ):
        
        super(DuelingDQN, self).__init__()

        # Extraction layers
        feature_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(torch.nn.Linear(input_dim, hidden_dim))
            feature_layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        self.feature_layer = torch.nn.Sequential(*feature_layers)

        # Value stream
        value_layers = []
        value_input_dim = input_dim
        for hidden_dim in value_stream_hidden_dims:
            value_layers.append(torch.nn.Linear(value_input_dim, hidden_dim))
            value_layers.append(torch.nn.ReLU())
            value_input_dim = hidden_dim
        value_layers.append(torch.nn.Linear(value_input_dim, 1))
        self.value_stream = torch.nn.Sequential(*value_layers)

        # Advantage stream
        advantage_layers = []
        advantage_input_dim = input_dim
        for hidden_dim in advantage_stream_hidden_dims:
            advantage_layers.append(torch.nn.Linear(advantage_input_dim, hidden_dim))
            advantage_layers.append(torch.nn.ReLU())
            advantage_input_dim = hidden_dim
        advantage_layers.append(torch.nn.Linear(advantage_input_dim, action_dim))
        self.advantage_stream = torch.nn.Sequential(*advantage_layers)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class D3QNAgent:
    def __init__(self,
                 agent_id: int,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: Any
                 ): 

        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config['common']['gamma']
        self.lr = config['D3QN']['lr']
        self.epsilon = config['D3QN']['epsilon'] 
        self.epsilon_decay = config['D3QN']['epsilon_decay']  
        self.epsilon_min = config['D3QN']['epsilon_min'] 
        self.buffer_size = config['D3QN']['buffer_size']
        self.batch_size = config['D3QN']['batch_size']

        self.device = device
        self.value_stream_hidden_dims = config['D3QN']['value_stream_hidden_dims']
        self.advantage_stream_hidden_dims = config['D3QN']['advantage_stream_hidden_dims']
        self.hidden_dims = config['D3QN']['hidden_dims']
        self.q_network = DuelingDQN(state_dim, action_dim, self.hidden_dims, self.value_stream_hidden_dims, self.advantage_stream_hidden_dims).to(self.device)
        self.target_q_network = DuelingDQN(state_dim, action_dim, self.hidden_dims, self.value_stream_hidden_dims, self.advantage_stream_hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        self.memory = deque(maxlen=self.buffer_size)

        self.counter = 0

    def _update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def act(self, state: np.ndarray, exploit=False) -> bool:
        if not exploit and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state: np.ndarray, actions: List[bool], rewards: List[float], next_state: np.ndarray, dones: bool):
        self.memory.append((state, actions, rewards, next_state, dones))

    def learn(self):

        self.counter += 1

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions_, rewards_, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions_)[:, self.agent_id]).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards_)[:, self.agent_id]).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        actions = actions.view(-1, 1)

        q_values = self.q_network(states)
        _values = q_values.gather(1, actions)
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_q_network(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())

        loss = self.criterion(_values, target_q_values)

        self.loss = loss.item()
        self.q_value_0 = q_values.detach()[0,0].item()
        self.q_value_1 = q_values.detach()[0,1].item()

        # Underflow
        if loss == None:
            return 0

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # Update target network periodically (every few episodes)
        if self.counter % 2 == 0:
            self._update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_metrics(self) -> Dict[str, float]:
        metrics_dict = {"loss": self.loss,
                        "Q value defect": self.q_value_0,
                        "Q value cooperate": self.q_value_1}
        return metrics_dict