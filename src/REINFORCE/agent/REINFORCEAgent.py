import os
import sys

import numpy as np
import torch
from torch.distributions.categorical import Categorical

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from REINFORCE.model.PolicyNet import PolicyNet


class REINFORCEAgent:
    def __init__(self, cfg: dict) -> None:

        """
        方策勾配法 : エージェントの行動確率をNNで表現
        theta_{t+1} = theta_{t} + alpha * J'(theta)
        J(theta) : エピソードあたりの報酬の期待値

        1) 方策に従って行動の決定
        2) 環境から報酬と次の状態を得る
        3) 状態(t), 行動(t), 報酬(t), 状態(t+1)を得る
        4) エピソード終了後, 報酬の期待値を計算 <- つまりはモンテカルロ
        5) エピソードで集まった全てのデータを用いてNNを学習
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 環境に関する情報
        self.action_size = cfg["Env"]["action_size"]
        self.state_size = cfg["Env"]["state_size"]

        self.gamma = cfg["REINFORCE"]["gamma"]
        self.lr = cfg["REINFORCE"]["lr"]

        self.memory = []
        self.pl_net = PolicyNet(self.state_size, self.action_size, self.device).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.pl_net.parameters(), lr=self.lr)

    def get_action(self, state: np.array):

        probs = self.pl_net(state)
        m = Categorical(probs)

        # サンプリング
        action = m.sample()

        # log pi_{theta}(a|s) の theta に関する勾配を求める
        derivative = -m.log_prob(action)

        return action, derivative

    def add(self, reward, derivative: torch.Tensor) -> None:

        data = (reward, derivative)
        self.memory.append(data)

    def update(self):

        self.optimizer.zero_grad()

        Q, loss = 0, 0
        for idx in np.arange(len(self.memory)):

            idx = len(self.memory) - idx - 1

            # 直近の報酬が重視されるように重みづけする
            Q += self.memory[idx][0] * self.gamma**idx

            # log pi_{theta}(a|s) の theta に関する勾配
            derivative = self.memory[idx][1]
            loss += derivative * Q

        # 期待値を単純に算術平均で近似する -> うまくいかない(to do)
        # loss /= len(self.memory)

        loss.backward()
        self.optimizer.step()
        self.memory = []
