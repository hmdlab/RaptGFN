"""Base class for optimization algorithms."""

import gzip
import os
import pickle

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseAlgorithm:
    """Base interface for an optimization algorithm."""

    def __init__(self, cfg, task, tokenizer, **kwargs):
        self.cfg = cfg
        self.task = task
        self.tokenizer = tokenizer
        self.state = {}
        self.state_save_path = getattr(cfg, "state_save_path", "./algorithm_state.pkl.gz")

    def optimize(self, task, init_data=None):
        """Run optimization; implemented by subclasses."""
        raise NotImplementedError("Override this method in your class")

    def log(self, metrics, commit=True):
        """Log metrics."""
        if WANDB_AVAILABLE:
            try:
                wandb.log(metrics, commit=commit)
            except wandb.errors.Error:
                # Ignore logging errors when W&B has not been initialized.
                print(f"Metrics: {metrics}")
        else:
            print(f"Metrics: {metrics}")

    def update_state(self, metrics):
        """Update the algorithm state."""
        for k, v in metrics.items():
            if k in self.state.keys():
                self.state[k].append(v)
            else:
                self.state[k] = [v]

    def save_state(self):
        """Save the algorithm state."""
        os.makedirs(os.path.dirname(self.state_save_path), exist_ok=True)
        with gzip.open(self.state_save_path, "wb+") as f:
            pickle.dump(self.state, f)

    def load_state(self, path=None):
        """Load a saved algorithm state."""
        path = path or self.state_save_path
        if os.path.exists(path):
            with gzip.open(path, "rb") as f:
                self.state = pickle.load(f)
        else:
            print(f"State file not found: {path}")
