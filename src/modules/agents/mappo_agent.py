#src/modules/agents/mappo_agent.py

import torch
import numpy as np 
from typing import Dict, Any, List


class MLPNetwork(torch.nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int]
                 ):
        
        super(MLPNetwork, self).__init__()

        # Architecture
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, action_dim))
        
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MAPPOAgent:
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
        self.clip_param = config['MAPPO']['clip_param'] 
        self.c1 = config['MAPPO']['c1']  
        self.c2 = config['MAPPO']['c2'] 
        self.lr = config['MAPPO']['lr']
        self.tau = config['MAPPO']['tau']

        self.hidden_dims = config['MAPPO']['hidden_dims']
        self.actor = MLPNetwork(state_dim, action_dim, self.hidden_dims).to(device)
        self.critic = MLPNetwork(state_dim, 1, self.hidden_dims).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.memory = []

        self.device = device

    def _compute_gae(self, next_value, rewards, masks, values):
        values = torch.cat((values, next_value.unsqueeze(0)), dim=0)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    def act(self, state: np.ndarray, exploit=False) -> bool:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        if not exploit:
            action = torch.distributions.Categorical(probs).sample().item()
        else:
            action = torch.argmax(probs).item()
        return action

    def remember(self, state: np.ndarray, actions: List[bool], rewards: List[float], next_state: np.ndarray, done: bool):
        self.memory.append((state, actions, rewards, next_state, done))

    def learn(self):

        state, actions, rewards, next_state, done = zip(*self.memory)
        self.memory = []
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        actions = torch.LongTensor(np.array(actions)[:, self.agent_id]).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)[:, self.agent_id]).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).to(self.device)

        # Compute values and advantages
        values = self.critic(state)
        next_values = self.critic(next_state).detach()
        returns = self._compute_gae(next_values[-1], rewards, 1 - done, values)

        returns = torch.cat(returns).detach()
        advantages = returns - values

        # Update actor
        logits = self.actor(state)
        log_probs = torch.functional.F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        old_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1).detach()
        new_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        probs_0 = probs[0,0].item()
        probs_1 = probs[0,1].item()
        self.probs = [probs_0, probs_1]

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        actor_loss -= self.c2 * entropy

        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        self.entropy = entropy
        self.actor_loss = actor_loss.item()

        # Update critic
        critic_loss = self.c1 * (returns - values).pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.critic_loss = critic_loss.item()

    def get_metrics(self) -> Dict[str, float]:
        metrics_dict = {"actor_loss": self.actor_loss,
                        "critic_loss": self.critic_loss,
                        "policies": self.probs}
        return metrics_dict
