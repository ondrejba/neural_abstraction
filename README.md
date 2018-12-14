

## Environments ##

* **envs/env_2d.py**: a simple environment with a continuous 2D state space and a discrete, arbitrarily large,
action space. The behavior of the environment is defined using JSON files in *resources/envs*.

## Networks ##

### Bisimulation ###

All networks for bisimulation learning must follow the class structure defined in **nets/abstract.py**.

* **nets/fully_connected.py**: a fully-connected network for experiments in simple environments.