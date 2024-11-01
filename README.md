# Abstract

This paper introduces the Minimum Price Markov Game (MPMG), a dynamic variant of the Prisoner's Dilemma. The MPMG serves as a theoretical model and reasonable approximation of real-world first-price sealed-bid public auctions that follow the minimum price rule. The goal is to provide researchers and practitioners with a framework to study market fairness and regulation in both digitized and non-digitized public procurement processes, amidst growing concerns about algorithmic collusion in online markets. Using multi-agent reinforcement learning-driven artificial agents, we demonstrate that algorithmic tacit coordination is difficult to achieve in the MPMG when cooperation is not explicitly engineered. Paradoxically, our results highlight the robustness of the minimum price rule in an auction environment, but also show that it is not impervious to full-scale algorithmic collusion. These findings contribute to the ongoing debates about algorithmic pricing and its implications.

# Project Structure

```
root/
├── modules/                        # Module directory
│   ├── agents/                     # MARL agents
│   │   ├── mappo_agent.py          # Multi-Agent Proximal Policy Optimization
|   |   ├── d3qn_agent.py           # Double Deep Q-network
│   │   ├── d3qn_om_agent.py        # Double Deep Q-network with opponent modeling
|   |   ├── eg_agent.py             # Epsilon Greedy bandit
|   |   ├── ts_agent.py             # Thompson Sampling bandit
|   |   └── ucb_agent.py            # Upper Confidence Bound bandit
│   ├── mpmg.py                     # Minimum Price Markov Game environment
|   ├── trainer.py                  # General trainer class
│   └── utils.py                    # Utility methods
├── logs/                           # Experiments' logs directory
├── metrics/                        # Training and evaluation metrics
├── plots/                          # Figures
├── main.py                         # Experiment control
├── run_all.sh                      # Run experiments Linux/Mac
├── run_all.bat                     # Run experiments Windows
├── config.yaml                     # Configuration file for training hyperparameters
├── .gitignore                      # Ignore directories and files for Git
├── requirements.txt                # Dependencies list
├── README.md                       # Description and usage guide
└── LICENSE                         # MIT License
```

# Requirements

- Git ([install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- Python: from v3.8 to v3.10 ([intsall Python](https://www.python.org/downloads/))
- pip package installer (usually installed automatically with Python)
- 32GB RAM
- GPU access (optional but recommended)
- Mac OS, Linux distribution or Windows

# Installation

(Via command line)

## Clone the Repository

`git clone https://github.com/IgorSadoune/multi-level-auction-generator.git`

## Virtual Environment (optional but recommended)

1. Create a virtual environment inside the downloaded repository. Go to the root of the folder "multi-level-auction-generator" and execute:

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

## All experiments

To run all the experiments, for Linux/Mac users, execute in terminal from the root directory:

```sh
bash run_all.sh
```

and for Windows users:

```powershell
run_all.bat
```

## Specific experiment

To run a specific experiment, for example, MAPPO agents playing the 5-player heterogeneous MPMG, use the following command:

```sh
python3 main.py --num_agents 5 --sigma_beta 0.5 --agent_name "mappo"
```

for Linux/MAC users and 

```sh
python main.py --num_agents 5 --sigma_beta 0.5 --agent_name "mappo"
```

for Windows users.

### Parameter description

- **`--num_agents`**: Specifies the number of agents (e.g., `5`). Note that increasing the number of agents may significantly increase computation time.
- **`--sigma_beta`**: Controls the level of agent heterogeneity, with values ranging between `0` and `0.5`. A higher value implies greater heterogeneity among agents.
- **`--agent_name`**: Specifies the type of agent to be used. Options include:
  - `"mappo"`: MAPPO agents
  - `"d3qn"`: Dueling Double Deep Q-network agents
  - `"d3qn_om"`: Dueling Double Deep Q-network with Opponent Modeling agents
  - `"e_greedy"`: Epsilon Greedy agents
  - `"ts"`: Thompson Sampling agents
  - `"ucb"`: Upper Confidence Bound agents

## Logs and Metrics

Upon execution of the experiments, log files will be created in the `logs/` directory for debugging support. The file naming convention follows:

```
mappo_5_05.log
```

where `"mappo"` indicates the agent class, `"5"` represents the number of agents (the value used for `--num_agents`), and `"05"` refers to the value `0.5` for the level of heterogeneity. Similarly, JSON files storing the various training and evaluation metrics will be created under the `metrics/` directory, using a similar naming convention:

```
mappo_5_05.json
```

If not existing, the `logs/` and `metrics/` directories will be created automatically. If a `.log` or `.json` file already exists before the execution of its associated experiment, it will be replaced if the experiment is executed again. When all experiments are executed at once, all the corresponding `.log` and `.json` files are created.

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# Author

Igor Sadoune - igor.sadoune@polymtl.ca

