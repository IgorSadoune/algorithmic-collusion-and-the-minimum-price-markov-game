@echo off

REM Run experiments with all configurations
setlocal enableDelayedExpansion

set AGENTS=mappo d3qn d3qnom eg ts ucb
set NUM_AGENTS="2" "5"
set SIGMA_BETA="0.0" "0.5"

for %%a in (%AGENTS%) do (
  for %%n in (%NUM_AGENTS%) do (
    for %%s in (%SIGMA_BETA%) do (
      echo Running experiment with agent: %%a, num_agents: %%~n, sigma_beta: %%~s
      python src\scripts\main.py --agent_name %%a --num_agents %%~n --sigma_beta %%~s
    )
  )
)

echo Plotting training and evaluation metrics
python src\scripts\plot.py --file "all"
echo Plotting heatmap
python src/scripts/heatmap.py

endlocal
