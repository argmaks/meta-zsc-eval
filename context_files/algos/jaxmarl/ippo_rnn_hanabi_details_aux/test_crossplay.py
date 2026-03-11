import flax.linen as nn
import flax.serialization as fs
import jax.numpy as jnp
from typing import Sequence, Dict
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
import hydra
import jaxmarl
import jax
import os
from functools import partial
from omegaconf import OmegaConf
import warnings
from networks import batchify, unbatchify, ActorCriticRNN
from networks import ScannedRNN
from jaxmarl.wrappers.baselines import LogWrapper
from evaluate_crossplay_gfppoy_best_of_k import evaluate_crossplay
# assert that the trained models exist
assert os.path.exists(os.path.join(os.path.dirname(__file__), "models", "100000.0", "model_0.pkl")), "Saved parameters for model 0 do not exist"
assert os.path.exists(os.path.join(os.path.dirname(__file__), "models", "100000.0", "model_1.pkl")), "Saved parameters for model 1 do not exist"



if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)
    algo_config = config['runner']['algo_config']
    rng = jax.random.PRNGKey(0)
    algo_config["SEEDS"] = [0,1]
    algo_config["SAVE_MODELS"] = True
    algo_config["TOTAL_TIMESTEPS"] = 1e5
    algo_config["WANDB_MODE"] = "disabled"
    algo_config["CHECKPOINT_TIMESTEPS"] = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        xp = evaluate_crossplay(algo_config, rng, rollouts=10)

    print(xp)