import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, device: torch.device) -> None:
        super(PolicyNet, self).__init__()
        self.device = device
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x: np.array) -> torch.tensor:
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x), dim=0)
        return x
