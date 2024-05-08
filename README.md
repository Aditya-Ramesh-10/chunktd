# Chunked-TD

The code in this repository supplements the paper ["Sequence Compression Speeds Up Credit Assignment in Reinforcement Learning"](https://arxiv.org/abs/2405.03878).

The current implementation of algorithms heavily favor clarity and flexibility over computational efficiency.


## Instructions

Run an individual experiment from the root directory:

```bash
# For chain and split environment
python3 -m scripts.evaluate_cns

# Or accumulated-charge environment
python3 -m scripts.train_acc_charge

# Or key-to-door environment
python3 -m scripts.train_keytodoor

```

The settings for environment and algorithm can be modified in the `config_defaults` dictionary in each training file, or directly through a [Weights & Biases (wandb)](https://docs.wandb.ai/) sweep.

In case you don't want to use Weights & Biases (for logging):
```bash
export WANDB_MODE=disabled
```

## Dependencies

- gymnasium
- numpy
- torch
- wandb
