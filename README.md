# marl-delivery
MARL samples code for Package Delivery.
You has to run and test against BFS agents for the following 5 configs.
The seeds are given at later time.

- Version v1.1: Fix a small logic of `start_time` should less than `dead_line`

# Testing scripts
```python main.py --seed 10 --max_time_steps 1000 --map map1.txt --num_agents 5 --n_packages 100``` 29.66 vs 129.5 vs 260.27

```python main.py --seed 10 --max_time_steps 1000 --map map2.txt --num_agents 5 --n_packages 100``` 38.7 vs 29.48 vs 358.33

```python main.py --seed 10 --max_time_steps 1000 --map map3.txt --num_agents 5 --n_packages 500``` 29.37 vs 40.519 vs 313.87

```python main.py --seed 10 --max_time_steps 1000 --map map4.txt --num_agents 10 --n_packages 500``` 29.2 vs 150.64 vs 783.78

```python main.py --seed 10 --max_time_steps 1000 --map map5.txt --num_agents 10 --n_packages 1000``` 8.66 vs 28 vs 58.18

# For RL testing
- You can use `simple_PPO.ipynb` as the starting point.
- Avoid modify the class `Env`, you can try to modify the `convert_state` function or `reward_shaping`
- You can choose to use or change the standard `PPO`. Note that: It is not easy to match the greedy agent, using RL.


# TODO:
- [x]: Add BFS agents
- [x]: Add test scripts
- [x]: Add RL agents
