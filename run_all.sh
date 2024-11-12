#!/bin/bash

# Run experiments with all configurations
declare -a AGENTS=("mappo" "d3qn" "eg" "ts" "ucb")
declare -a NUM_AGENTS=(2 5)
declare -a SIGMA_BETA=(0.0 0.5)

for agent in "${AGENTS[@]}"; do
  for num_agents in "${NUM_AGENTS[@]}"; do
    for sigma in "${SIGMA_BETA[@]}"; do
      echo "Running experiment with agent: $agent, num_agents: $num_agents, sigma_beta: $sigma"
      python3 src/scripts/main.py --agent_name "$agent" --num_agents "$num_agents" --sigma_beta "$sigma"
    done
  done
done

# Plot all figures
echo "Plotting metrics and heatmap"
python3 src/scripts/plot.py
