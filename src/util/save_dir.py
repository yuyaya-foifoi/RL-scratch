import os


def get_DQNdir(cfg: dict) -> str:
    lr = str(cfg["DQN"]["lr"])
    sync = str(cfg["DQN"]["sync_interval"])
    gamma = str(cfg["DQN"]["gamma"])
    batch = str(cfg["DQN"]["batch_size"])
    buffer_size = str(cfg["DQN"]["buffer_size"])
    eps = str(cfg["DQN"]["epsilon"])

    path = "./logs/DQN/lr_{}__sync_{}__gamma_{}__batch_{}__bfsize_{}__eps_{}".format(
        lr, sync, gamma, batch, buffer_size, eps
    )

    os.makedirs(path, exist_ok=True)

    return path


def get_REINFORCEdir(cfg: dict) -> str:
    lr = str(cfg["REINFORCE"]["lr"])
    gamma = str(cfg["REINFORCE"]["gamma"])

    path = "./logs/REINFORCE/lr_{}__gamma_{}".format(lr, gamma)

    os.makedirs(path, exist_ok=True)

    return path


def get_ActorCriticdir(cfg: dict) -> str:

    gamma = str(cfg["ActorCritic"]["gamma"])
    lr_pl_net = str(cfg["ActorCritic"]["pl_net_lr"])
    lr_v_net = str(cfg["ActorCritic"]["v_net_lr"])

    path = "./logs/ActorCritic/lrPL_{}__lrV_{}__gamma_{}".format(
        lr_pl_net, lr_v_net, gamma
    )

    os.makedirs(path, exist_ok=True)

    return path
