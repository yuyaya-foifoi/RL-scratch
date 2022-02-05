# RL-scratch

# Directory Configuration
```bash
.
├── Makefile
├── README.md
├── config
│   └── config.yml
├── logs
│   ├── ActorCritic
│   ├── DQN
│   └── REINFORCE
├── notebooks
│   └── dqn.ipynb
├── requirements.txt
└── src
    ├── ActorCritic
    │   ├── agent
    │   │   ├── ActorCriticAgent.py
    │   │   └──  __init__.py
    │   ├── model
    │   │   ├── PolicyNet.py
    │   │   ├── ValueNet.py
    │   │   └── __init__.py
    │   └── trainer
    │       ├── __init__.py
    │       └── trainer.py
    ├── DQN
    │   ├── agent
    │   │   ├── DQNAgent.py
    │   │   └── __init__.py
    │   ├── model
    │   │   ├── DQN.py
    │   │   ├── ReplayBuffer.py
    │   │   └── __init__.py
    │   └── trainer
    │       ├── __init__.py
    │       └── trainer.py
    ├── REINFORCE
    │   ├── agent
    │   │   ├── REINFORCEAgent.py
    │   │   └── __init__.py
    │   ├── model
    │   │   ├── PolicyNet.py
    │   │   └── __init__.py
    │   └── trainer
    │       ├── __init__.py
    │       └── trainer.py
    └── util
        ├── __init__.py
        ├── get_agent.py
        ├── loss_function.py
        └── save_dir.py
```

# Tutorial
```bash
python ./src/dqn.ipynb
```

# Run
### DQN
```bash
python ./src/DQN/trainer/trainer.py
```

### REINFORCE
```bash
python ./src/REINFORCE/trainer/trainer.py
```

### ActorCritic
```bash
./notebooks/ActorCritic/trainer/trainer.py
```
# Code Formatter
```bash
make style
```

# References
```bash
[1] Saito K., “『ゼロから作る Deep Learning ❹』 公開レビューのお知らせ,” note, 03-Dec-2021. [Online]. Available: https://note.com/koki0702/n/nd5147f1fcb89. [Accessed: 05-Feb-2022].
[2] A. Paszke, “Reinforcement learning (DQN) tutorial — PyTorch tutorials 1.10.1+cu102 documentation,” Pytorch.org. [Online]. Available: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html. [Accessed: 05-Feb-2022].
```


