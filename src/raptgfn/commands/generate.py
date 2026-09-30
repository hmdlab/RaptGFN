"""Sequence-generation command."""

import logging
import os
import random
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import MISSING, DictConfig, OmegaConf

from raptgfn.utils.common import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def run(config: DictConfig):
    """Generate and score sequences from a trained checkpoint."""
    print("=== Configuration ===")
    print(OmegaConf.to_yaml(config))

    if config.checkpoint_path is MISSING:
        print("Error: checkpoint_path is required.")
        sys.exit(1)

    # Initialize the random seed.
    config.seed = random.randint(0, 100000) if config.seed is None else config.seed
    set_seed(config.seed)

    ret_val = float("NaN")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            # Initialize the tokenizer.
            logging.info("Initializing the tokenizer...")
            tokenizer = hydra.utils.instantiate(config.tokenizer)

            # Initialize the task.
            logging.info("Initializing the task...")
            task = hydra.utils.instantiate(config.task, tokenizer=tokenizer)
            project_root = Path(os.getcwd()).resolve()
            task.task_setup(config, project_root=project_root)

            # Initialize the GFlowNet algorithm.
            # Pass checkpoint_path to the algorithm configuration.
            config.algorithm.checkpoint_path = config.checkpoint_path

            logging.info(
                f"Initializing the algorithm and loading the model from {config.checkpoint_path}..."
            )
            algorithm = hydra.utils.instantiate(
                config.algorithm,
                task=task,
                tokenizer=tokenizer,
                cfg=config.algorithm,
                task_cfg=config.task,
            )

            # Generate sequences and calculate their objective scores.
            print(f"\n=== Sequence generation started ===")
            print(f"Objectives: {task.objectives}")
            print(f"Requested sequences: {config.num_samples}")
            print(f"Device: {algorithm.device}")
            print("============================\n")

            # Set the requested sample count and optional evaluation preference.
            algorithm.num_samples = config.num_samples
            if config.pref is not None:
                algorithm.eval_pref = torch.tensor(config.pref).float()

            with torch.no_grad():
                samples, all_rews, _, _, _, _ = algorithm.evaluation(task, plot=False)

            # Write generated sequences and scores to CSV.
            scalar_rewards = np.sum(all_rews, axis=1)

            output_df = pd.DataFrame({"sequence": samples, "reward": scalar_rewards.tolist()})

            # Include the individual objective-specific score columns.
            for i, obj_name in enumerate(task.objectives):
                output_df[f"{obj_name}_score"] = all_rews[:, i].tolist()

            output_df.to_csv(config.output_file, index=False)

            print(f"\n=== Generation complete ===")
            print(f"Saved generated sequences and scores to {config.output_file}.")
            print(f"Mean evaluation score: {scalar_rewards.mean():.4f}")
            print("===================\n")

            ret_val = scalar_rewards.mean()

        except FileNotFoundError:
            logging.error(
                f"Error: checkpoint file not found: {config.checkpoint_path}"
            )
            sys.exit(1)
        except Exception as err:
            logging.exception(f"An unexpected error occurred: {err}")
            ret_val = float("NaN")
            sys.exit(1)

    return ret_val


if __name__ == "__main__":
    run()
    sys.exit()
