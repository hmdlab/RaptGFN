"""Model-training command."""

import logging
import os
import random
import sys
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from raptgfn.utils import flatten_config
from raptgfn.utils.common import set_seed

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be limited")


def init_run(cfg):
    """Initialize the run configuration and random state."""
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = f"aptamer_mogfn_{trial_id}"

    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)

    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve configuration interpolations.
    cfg = DictConfig(cfg)

    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    # Save the resolved configuration.
    with open("hydra_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    return cfg


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def run(config):
    """Run model training."""
    random.seed(None)  # Reset the seed selection before initializing a multirun job.

    # Prepare the logging configuration.
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    # Initialize optional W&B logging.
    if WANDB_AVAILABLE and config.wandb_mode != "disabled":
        wandb.init(
            project=config.project_name,
            config=log_config,
            mode=config.wandb_mode,
            group=config.group_name,
            name=config.exp_name,
            tags=config.exp_tags,
        )
        config["job_name"] = wandb.run.name

    config = init_run(config)  # Fix the run's random seed here.

    ret_val = float("NaN")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings within the training block.

        try:
            logging.info("Initializing the tokenizer...")
            tokenizer = hydra.utils.instantiate(config.tokenizer)

            logging.info("Initializing the task...")
            task = hydra.utils.instantiate(config.task, tokenizer=tokenizer)

            # Use the current working directory as the project root.
            project_root = Path(os.getcwd()).resolve()
            all_seqs, all_targets = task.task_setup(config, project_root=project_root)

            logging.info("Initializing the algorithm...")
            algorithm = hydra.utils.instantiate(
                config.algorithm, task=task, tokenizer=tokenizer, cfg=config.algorithm
            )

            logging.info("Starting optimization...")
            print(f"\n=== Training started ===")
            print(f"Objectives: {task.objectives}")
            print(f"Sequence length: {task.min_len}-{task.max_len}")
            print(f"Training steps: {config.algorithm.train_steps}")
            print(f"Device: {algorithm.device}")
            print("============================\n")

            metrics = algorithm.optimize(task, init_data=None)

            # Remove metric-name prefixes.
            metrics = {key.split("/")[-1]: val for key, val in metrics.items()}
            ret_val = metrics.get("hypervol_rel", float("NaN"))

            print(f"\n=== Optimization complete ===")
            print(f"Final hypervolume metric: {ret_val:.4f}")
            print("===================\n")

        except Exception as err:
            logging.exception(f"An error occurred: {err}")
            ret_val = float("NaN")

    if WANDB_AVAILABLE and config.wandb_mode != "disabled":
        wandb.finish()

    return ret_val


if __name__ == "__main__":
    run()
    sys.exit()
