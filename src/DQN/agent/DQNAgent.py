import copy
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from DQN.model.DQN import QNet
from DQN.model.ReplayBuffer import ReplayBuffer
from util.loss_function import get_loss_function


class DQNAgent:
    def __init__(self, cfg: dict) -> None:

        """
        # 1) 最適行動価値関数を知りたい
        # 2) 行動価値関数を知るには方策と遷移状態確率 P(s_{t+1}| s{t}, a_{t}) が必要
        # 3) サンプリングと逐次の更新によって行動価値関数を得ることが目的
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 環境に関する情報
        self.action_size = cfg["Env"]["action_size"]
        self.state_size = cfg["Env"]["state_size"]

        self.gamma = cfg["DQN"]["gamma"]
        self.lr = cfg["DQN"]["lr"]
        self.epsilon = cfg["DQN"]["epsilon"]
        self.buffer_size = cfg["DQN"]["buffer_size"]
        self.batch_size = cfg["DQN"]["batch_size"]
        self.loss_function = get_loss_function(cfg["DQN"]["loss_function"])

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.state_size, self.action_size, self.device).to(self.device)
        self.qnet_target = QNet(self.state_size, self.action_size, self.device).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: np.array) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax().item()

    def update(
        self,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        """
        Experience Replay
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # qnet(行動価値観数) で現在の状態から行動を推定する
        self.qnet.eval()
        qs = self.qnet(state)

        # action で選択した方の行動がq
        q = qs[np.arange(self.batch_size), action]

        # qnet_target でt+1時点の状態からt+1時点の行動を推定する
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)[0].to("cpu").detach().numpy()

        # t+1時点の行動価値
        td_target = reward + (1 - done) * self.gamma * next_q
        td_target = torch.from_numpy(td_target.astype(np.float32)).to(self.device)

        # t+1時点の行動価値とt時点の行動価値の誤差をとる
        loss = self.loss_function(q, td_target)

        self.qnet.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        """
        Fixed Target Q-Network
        """
        self.qnet_target = copy.deepcopy(self.qnet)
