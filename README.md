# rllabsharp

Rllabsharp contains the code of our [paper](https://arxiv.org/abs/1710.11198) for TRPO experiments based on [rllab++](https://github.com/shaneshixiang/rllabplusplus).

The codes are experimental, and may require tuning or modifications to reach the best reported performances.

## Installation

Please follow the basic installation instructions in [rllab documentation](https://rllab.readthedocs.io/en/latest/).

## Dependency
- `Python 3.5`
- `Anaconda`
- [`rllab`](https://github.com/rll/rllab)
- `MuJoCo`

## Running Experiments
We provide three different Phi structures(linear, quadratic, mlp) and two different Phi optimization methods(FitQ and MinVar) in the repository. Optional flags are defined in [launcher_utils.py](sandbox/rocky/tf/launchers) and here are some running examples:

```sh
cd sandbox/rocky/tf/launchers
# Hopper-v1 with linear Phi and FitQ optimization
python algo_gym_stub.py --env_name Hopper-v1 --algo_name cfpo --pf_cls linear --use_gradient_vr False --pf_learning_rate 1e-4 --pf_iters 400

# Hopper-v1 with Linear Phi and MinVar optimization
python algo_gym_stub.py --env_name Hopper-v1 --algo_name cfpo --pf_cls linear --use_gradient_vr True --pf_learning_rate 1e-3 --pf_iters 800

# Hopper-v1 baseline qprop
python algo_gym_stub.py --env_name Hopper-v1 --algo_name qprop --qprop_eta_option=adapt1

```

## Feedbacks
Currently the code is a little messy, we will clean it and make it easier for test soon. If you have any questions about the code or the paper, please feel free to contact [Yihao Feng](yihaof95@gmail.com) or [Hao Liu](uestcliuhao@gmail.com).

