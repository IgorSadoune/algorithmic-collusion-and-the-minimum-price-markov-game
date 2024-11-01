#root/main.py

import yaml 
import torch 
import argparse 
from modules.agents.mappo_agent import MAPPOAgent
from modules.trainer import Trainer
from modules.utils import restricted_float

from modules.agents.mappo_agent import MAPPOAgent
from modules.agents.d3qn_agent import D3QNAgent
from modules.agents.d3qn_om_agent import D3QNOMAgent
from modules.agents.eg_agent import EpsilonGreedyAgent
from modules.agents.ts_agent import ThompsonSamplingAgent
from modules.agents.ucb_agent import UCBAgent

# Parsing arguments
parser = argparse.ArgumentParser(description='Experiment configuration')
parser.add_argument('--num_agents', type=int, default=None, help='Number of agents')
parser.add_argument('--sigma_beta', type=restricted_float, default=None, help='Sigma beta value (must be between 0 and 1)')
parser.add_argument('--agent_name', type=str, default=None, help='Agent name to call the right agents')
args = parser.parse_args()

# Load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize experiment
agent_dict = {
    'mappo': MAPPOAgent,
    'd3qn': D3QNAgent,
    'd3qn_om': D3QNOMAgent,
    'e_greedy': EpsilonGreedyAgent,
    'ts': ThompsonSamplingAgent,
    'ucb': UCBAgent
}

num_agents = args.num_agents
sigma_beta = args.sigma_beta
agent = {args.agent_name: agent_dict[args.agent_name]}

# Run experiment
trainer = Trainer(config=config, num_agents=num_agents, sigma_beta=sigma_beta, agent=agent, device=device)
trainer.train()