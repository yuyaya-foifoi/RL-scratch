import argparse
import os
import sys

import gym
import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import matplotlib.pyplot as plt
import pandas as pd

from REINFORCE.agent.REINFORCEAgent import REINFORCEAgent
from util.get_agent import get_agent
from util.save_dir import get_REINFORCEdir


def REINFORCEtrainer(cfg: dict) -> None:

    episodes = cfg["REINFORCE"]["episodes"]
    env = gym.make(cfg["Env"]["name"])

    agent = REINFORCEAgent(cfg)
    reward_log = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        sum_reward = 0

        while not done:

            action, derivative = agent.get_action(state)
            next_state, reward, done, info = env.step(action.item())

            agent.add(reward, derivative)
            state = next_state
            sum_reward += reward

        agent.update()

        reward_log.append(sum_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, sum_reward))

    df = pd.DataFrame(reward_log, columns=["reward_log"])

    base_path = get_REINFORCEdir(cfg)
    csv_path = os.path.join(base_path, "log.csv")
    png_path = os.path.join(base_path, "log.png")
    df.to_csv(csv_path)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(len(reward_log)), reward_log)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./config/config.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    REINFORCEtrainer(cfg)
