# Multi-Agent Reinforcement Learning for Power Grid Topology Optimization
This repository contains the code used for the experiments of the paper 
[Multi-Agent Reinforcement Learning for Power Grid Topology Optimization](https://doi.org/10.48550/arXiv.2310.02605).

## Environment setting
- python >= 3.11  
- grid2op >= 1.9.1 
- lightsim2grid == 0.7.3 

### Create conda environment
```sh
conda env create -f environment.yml
conda activate MARL2023paper_env
```

### lightsim2grid installation
In case you are experiencing the following problem:\
https://github.com/BDonnot/lightsim2grid/issues/55 \
Follow the steps from https://lightsim2grid.readthedocs.io/en/latest/install_from_source.html#install-python
```sh
git clone https://github.com/BDonnot/lightsim2grid.git
cd lightsim2grid
git checkout v0.7.3
git submodule init
git submodule update
make
pip install -U pybind11
pip install -U .
```

## Scripts
### Train
To train an agent run `test.py` with the appropriate arguments.
An overview of all arguments that can be used for training the agent is provided in `test.py`. 
See the snippit below to give you an idea.
```sh
python test.py -n=[experiment_name] -a=[agent] 
                                -s=[seed] -c=[environment_name (5, 14)]

# Example
python test.py -n=case_5 -a='ppo' -s=0 -c=5 
```

### Evaluate
With `evaluate.py` you can evaluate the trained agent on a different set of chronics.
The detail of arguments is provided in `evaluate.py`. The snippit below shows an example how to use this script.
```sh
python evaluate.py -rd=[results_directory] -mn=[model_name]

# Example
python evaluate.py -rd=result -mn=case_5_ppo
```

## Visualization of code
![FlowChart of Code](/images/MARL_flowchart.png?raw=true "Flow chart of Code")

## Agent algorithms implemented
Below the agents that are implemented are listed as: 

'agent_name': _Name or description of the algorithm used (abbreviation)_ (link to paper)

Where 'agent_name' is what is used when training a specific agent. As described in the Train section.

1. 'isacd': _Multi agent Independend SACD (ISACD)_, <sub><sup>([E. van der Sar et al 2023](https://doi.org/10.48550/arXiv.1910.07207)) </sub></sup>
1. 'ippo': _Multi agent Independend PPO (IPPO)_, <sub><sup>([E. van der Sar et al 2023](https://doi.org/10.48550/arXiv.1910.07207)) </sub></sup>
1. 'dsacd': _Multi agent Dependend SACD (DSACD)_, <sub><sup>([E. van der Sar et al 2023](https://doi.org/10.48550/arXiv.1910.07207)) </sub></sup>
1. 'dppo': _Multi agent Dependend PPO (DPPO)_, <sub><sup>([E. van der Sar et al 2023](https://doi.org/10.48550/arXiv.1910.07207)) </sub></sup>
1. 'sacd': _Single agent Soft Actor Critic Discrete (SACD)_, <sub><sup>([P. Christodoulou 2019](https://doi.org/10.48550/arXiv.1910.07207)) </sub></sup>
1. 'ppo': _Single agent Proximal Policy Optimization (PPO)_,<sub><sup>([J. Schulman et al 2017](https://doi.org/10.48550/arXiv.1707.06347)) </sub></sup>
1. 'sac': _Semi-Markov Afterstate Actor-Critic (SMAAC)_ <sub><sup>([D.Yoon et al 2021](https://openreview.net/pdf?id=LmUJqB1Cz8)) </sub></sup>
1. 'sac2': _Semi-Markov Afterstate Actor-Critic (SMAAC)_  (adjusted version, to how it was actually described in the paper.)
1. 'sacd2': _SACD with a goal / list of actions set out (TEST)_
1. 'dqn': Deep Q-learning (DQN), <sub><sup>([V. Mnih et al 2013](https://doi.org/10.48550/arXiv.1312.5602)) </sub></sup>
1. 'dqn2': _DQN2, DQN with an adjusted GNN network (TEST)_


## References
```bibtex
@article{vandersar2023multiagent,
         title={Multi-Agent Reinforcement Learning for Power Grid Topology Optimization}, 
         author={Erica van der Sar and Alessandro Zocca and Sandjai Bhulai},
         year={2023},
         eprint={2310.02605},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
}
```

## Credit
The environment used for this code is rte-france's Grid2Op (https://github.com/rte-france/Grid2Op).

Furthermore, the code of https://github.com/KAIST-AILab/SMAAC (Copyright (c) 2020 KAIST-AILab) was used as a basis for this code base.