# Abstract RL

## Description

A modular python implementation of various reinforcement learning algorithms
for use in control problems. There is a distinction between discrete and 
continuous action spaces. For example Deep Q Network is discrete. Nevertheless
automatic discretization is supported by the framework already. So it can be 
used for control problems as well.

The following algorithmic papers were used:

- Maximum A Posteriori Policy Optimization (https://arxiv.org/abs/1806.06920)
- Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)
- Relative Entropy Regularized Policy Iteration (https://arxiv.org/abs/1812.02256)

Usually policy gradients algorithm can be successfully combined with value
function methods into a actor critic architecture. Additionally the following
methods are used for estimating the v and q function:

- Retrace (https://arxiv.org/abs/1606.02647)
- Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

## Installation

To install the abstract_rl package, start by checking out the repository and 
install it locally.

```
cd abstract_rl
pip install -e .
```

## Usage 

To create a new script utilizing TRPO or MPO simply include:

```
git clone git@gitlab.com:kosmitive/abstract_rl.git
cd abstract_rl
pip install -r requirements.txt
pip install -e .
```