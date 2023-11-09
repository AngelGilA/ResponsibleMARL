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

## Agent algorithms implemented



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
The environment used for this code is rte-france's Grid2Op (https://github.com/rte-france/Grid2Op)
Furthermore, the code of https://github.com/KAIST-AILab/SMAAC (Copyright (c) 2020 KAIST-AILab) was used as a basis for this code base.