import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ActorCritic.model.PolicyNet import PolicyNet
from ActorCritic.model.ValueNet import ValueNet
from util.loss_function import get_loss_function


class ActorCriticAgent:
    def __init__(self, cfg: dict) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 環境に関する情報
        self.action_size = cfg["Env"]["action_size"]
        self.state_size = cfg["Env"]["state_size"]

        self.gamma = cfg["ActorCritic"]["gamma"]
        self.lr_pl_net = cfg["ActorCritic"]["pl_net_lr"]
        self.lr_v_net = cfg["ActorCritic"]["v_net_lr"]

        self.pl_net = PolicyNet(self.state_size, self.action_size, self.device).to(
            self.device
        )
        self.v_net = ValueNet(self.state_size, self.action_size, self.device).to(
            self.device
        )

        self.optimizer_pl = torch.optim.Adam(
            self.pl_net.parameters(), lr=self.lr_pl_net
        )
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=self.lr_v_net)

        self.loss_function = get_loss_function(cfg["ActorCritic"]["loss_function"])

    def get_action(self, state):

        probs = self.pl_net(state)
        m = Categorical(probs)

        # サンプリング
        action = m.sample()

        # log pi_{theta}(a|s) の theta に関する勾配を求める
        derivative = -m.log_prob(action)

        return action, derivative

    def update(self, state, derivative, reward, next_state, done):

        self.optimizer_v.zero_grad()
        self.optimizer_pl.zero_grad()

        td_target = reward + self.gamma * self.v_net(next_state) * (1 - done)
        v = self.v_net(state)
        loss_v = self.loss_function(v, td_target)

        delta = td_target - v
        loss_pl = derivative * delta

        loss_v.backward(retain_graph=True)
        loss_pl.backward(retain_graph=True)
        self.optimizer_v.step()
        self.optimizer_pl.step()
