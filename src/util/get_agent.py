import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from DQN.agent.DQNAgent import DQNAgent
from REINFORCE.agent.REINFORCEAgent import REINFORCEAgent


def get_agent(cfg: dict):
    if cfg["Agent"] == "DQN":
        return DQNAgent(cfg)

    if cfg["Agent"] == "REINFORCE":
        return REINFORCEAgent(cfg)
