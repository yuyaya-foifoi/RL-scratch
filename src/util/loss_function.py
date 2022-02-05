import torch.nn as nn


def get_loss_function(loss_function_name: str) -> nn.modules.loss:
    if loss_function_name == "L1smooth":
        return nn.SmoothL1Loss()
