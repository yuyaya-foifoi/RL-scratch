import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, device: torch.device) -> None:
        super(ValueNet, self).__init__()
        self.device = device
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x: np.array) -> torch.tensor:
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
