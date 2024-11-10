# Table of Contents
- [Paper Abstract](#abstract)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Virtual Environment](#virtual-environment-optional-but-recommended)
  - [Install the Required Dependencies](#install-the-required-dependencies)
- [Usage](#usage)
  - [Experiments](#all-experiments)
  - [Computational Considerations (with default hyperparameters)](#computational-considerations-with-default-configuration-of-models-hyperparameters)
  - [Hyperparameter Configuration](#models-hyperparameter-configuration)
  - [Specific Experiment](#specific-experiment)
  - [Experiment Parameter Description](#parameter-description)
- [License](#license)
- [Author](#author)

# Paper Abstract

This paper introduces the Minimum Price Markov Game (MPMG). The MPMG serves as a theoretical model and reasonable approximation of real-world first-price sealed-bid public auctions that follow the minimum price rule. The goal is to provide researchers and practitioners with a framework to study market fairness and regulation in both digitized and non-digitized public procurement processes, amidst growing concerns about algorithmic collusion in online markets. We demonstrate, using multi-agent reinforcement learning-driven artificial agents, that algorithmic tacit coordination is difficult to achieve in the MPMG when cooperation is not explicitly engineered. Paradoxically, our results highlight the robustness of the minimum price rule in an auction environment, but also show that it is not impervious to full-scale algorithmic collusion. These findings contribute to the ongoing debate about algorithmic pricing and its implications.


# Project Structure

```
run_all.sh                      # Run all experiments and plots script for Linux/Mac
run_all.bat                     # Run all experiments and plots script for Windows
LICENSE                         # MIT License
.gitignore                      # Ignore directories and files for Git
README.md                       # Description and usage guide
requirements.txt                # Dependencies list
src/                            # Source directory
├── config.yaml                 # Configuration file for experiments and models' hyperparameters
├── logs/                       # Contains experiments' .log files
├── metrics/                    # Contains training and evaluation metrics JSON files
├── plots/                      # Contains figures
├── modules/                    # Module directory
│   ├── agents/                 # MARL agents
│   │   ├── mappo_agent.py      # Multi-Agent Proximal Policy Optimization
│   │   ├── d3qn_agent.py       # Double Deep Q-network
│   │   ├── d3qn_om_agent.py    # Double Deep Q-network with opponent modeling
│   │   ├── eg_agent.py         # Epsilon Greedy bandit
│   │   ├── ts_agent.py         # Thompson Sampling bandit
│   │   └── ucb_agent.py        # Upper Confidence Bound bandit
│   ├── mpmg.py                 # Minimum Price Markov Game environment
│   ├── trainer.py              # General trainer class
|   ├── plotter.py              # Plotting module
│   └── utils.py                # Utility methods
└── scripts/                    # Scripts directory
    ├── plot.py                 # Plotting control script
    └── main.py                 # Experiment control script
```

# Requirements

- Python: from v3.8 to v3.10 ([intsall Python](https://www.python.org/downloads/))
- pip package installer (usually installed automatically with Python)
- 32GB RAM
- GPU access (optional but recommended)
- Mac OS, Linux distribution or Windows

# Installation

(Via command line)

## Download or Clone the Repository

Git clone or manual download via `https://github.com/IgorSadoune/Algorithmic-Collusion-and-the-Minimum-Price-Markov-Game.git`.

## Virtual Environment (optional but recommended)

1. Create a virtual environment from the root directory:

- On Mac/Linux, execute:
`python3 -m venv venv`

- On Windows, execute:
`python -m venv venv`

2. Activate the virtual environment using:

- On Mac/Linux, execute:
`source venv/bin/activate`

- On Windows, execute:
`.\venv\Scripts\activate`

**The virtual environment always needs to be activated when executing files from this repository.**

## Install the Required Dependencies:

The required python libraries are listed in the "requirements.txt" file. Those can directly be downloaded to your virtual environment (or root system if venv not setup) by executing

`pip install -r requirements.txt`

**If for some reason an error occurs with one package, the following commands will allow you to install the subsequent packages in the list:**

- On Mac/Linux:
  `while read package; do
    pip install "$package" || echo "Failed to install $package" >&2
done < requirements.txt`

- On Windows:
   - if using PowerShell 7:
     `Get-Content requirements.txt | ForEach-Object {
       pip install $_ || Write-Error "Failed to install $_"}`
   - or, if using command prompt:
     `for /f %i in (requirements.txt) do pip install %i`
     
**I do not provide support for Windows users utilizing PowerShell 5.**

# Usage

## Experiments

To run the experiments, for Linux/Mac users, execute in terminal from the root directory:

```sh
bash run_all.sh
```

and for Windows users:

```powershell
run_all.bat
```

This command will produce the `log.log`, `metrics.json`, and `plot.png` files.

## Computational Considerations (with default hyperparamters) 
There are 24 experiments as each of the 6 agent classes used in the paper are tested on 4 configurations of the MPMG. Among the 24 experiments, 12 are conducted on the 2-player MPMG and 12 on the 5-player MPMG, that is, 84 agents are trained over 100 replications of 100 training episodes, for a total 840,000 training iterations (optimization always uses stochastic gradient ascent). 

## Hyperparameter Configuration
Default configuration is stored in `src/config.yaml`, where the file can be accessed and modified. 

## Specific Experiment

To run a specific experiment, for example, MAPPO agents playing the 5-player heterogeneous MPMG, use the following command:

```sh
python3 src/scripts/main.py --num_agents 5 --sigma_beta 0.5 --agent_name "mappo"
```

for Linux/MAC users and 

```sh
python src/scripts/main.py --num_agents 5 --sigma_beta 0.5 --agent_name "mappo"
```

for Windows users. 

## Experiment Parameter Description
- **`--num_agents`**: Specifies the number of agents (e.g., `5`). Note that increasing the number of agents may significantly increase computation time.
- **`--sigma_beta`**: Controls the level of agent heterogeneity, with values ranging between `0.0` and `0.5`. A higher value implies greater heterogeneity among agents.
- **`--agent_name`**: Specifies the type of agent to be used. Options include:
  - `"mappo"`: MAPPO agents
  - `"d3qn"`: Dueling Double Deep Q-network agents
  - `"d3qnom"`: Dueling Double Deep Q-network with Opponent Modeling agents
  - `"eg"`: Epsilon Greedy agents
  - `"ts"`: Thompson Sampling agents
  - `"ucb"`: Upper Confidence Bound agents


# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# Author

Igor Sadoune - igor.sadoune@polymtl.ca

