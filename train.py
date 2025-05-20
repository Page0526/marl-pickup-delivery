from env import Environment
from agent import train_agents  # phải chắc bạn lưu agent.py với tên này
import os

if __name__ == "__main__":
    env = Environment(map_file="map.txt", max_time_steps=1000, n_robots=5, n_packages=20, seed=2025)
    agents = train_agents(env, n_episodes=50, save_dir="models")
