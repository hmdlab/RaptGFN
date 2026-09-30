"""Utility functions for multi-objective GFlowNets."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from polyleven import levenshtein

    POLYLEVEN_AVAILABLE = True
except ImportError:
    POLYLEVEN_AVAILABLE = False

try:
    from botorch.utils.multi_objective import pareto

    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def mean_pairwise_distances(seqs):
    """Compute the mean pairwise sequence distance."""
    if not POLYLEVEN_AVAILABLE:
        print("Warning: polyleven not available, using basic distance")
        return _basic_pairwise_distances(seqs)

    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(levenshtein(*pair))
    return np.mean(dists)


def _basic_pairwise_distances(seqs):
    """Compute fallback pairwise distances when polyleven is unavailable."""
    dists = []
    for i, seq1 in enumerate(seqs):
        for j, seq2 in enumerate(seqs[i + 1 :], i + 1):
            # Use Hamming distance for equal-length sequences.
            if len(seq1) == len(seq2):
                dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
            else:
                dist = abs(len(seq1) - len(seq2)) + min(len(seq1), len(seq2))
            dists.append(dist)
    return np.mean(dists) if dists else 0.0


def generate_simplex(dims, n_per_dim):
    """Construct a grid of points on the simplex."""
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in itertools.product(*spaces) if np.allclose(sum(comb), 1.0)])


def thermometer(v, n_bins=50, vmin=0, vmax=32):
    """Encode scalar values using thermometer features."""
    # Create the bins on the same device as v.
    device = v.device if hasattr(v, "device") else "cpu"
    bins = torch.linspace(vmin, vmax, n_bins, device=device)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap


def plot_pareto(pareto_rewards, all_rewards, pareto_only=False, objective_names=None):
    """Plot a Pareto-front diagnostic."""
    if pareto_rewards.shape[-1] < 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if not pareto_only:
            ax.scatter(
                *np.hsplit(all_rewards, all_rewards.shape[-1]),
                color="grey",
                alpha=0.6,
                label="All Samples",
            )
        ax.scatter(
            *np.hsplit(pareto_rewards, pareto_rewards.shape[-1]),
            color="red",
            s=60,
            label="Pareto Front",
        )

        if objective_names is not None:
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
        else:
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")

        ax.legend()
        ax.grid(True, alpha=0.3)

        if WANDB_AVAILABLE:
            return wandb.Image(fig)
        else:
            return fig

    elif pareto_rewards.shape[-1] == 3 and PLOTLY_AVAILABLE:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=all_rewards[:, 0],
                    y=all_rewards[:, 1],
                    z=all_rewards[:, 2],
                    mode="markers",
                    marker=dict(color="grey", size=4),
                    name="All Samples",
                ),
                go.Scatter3d(
                    x=pareto_rewards[:, 0],
                    y=pareto_rewards[:, 1],
                    z=pareto_rewards[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Pareto Front",
                ),
            ]
        )

        if objective_names is not None:
            fig.update_layout(
                scene=dict(
                    xaxis_title=objective_names[0],
                    yaxis_title=objective_names[1],
                    zaxis_title=objective_names[2],
                )
            )

        return fig
    else:
        print(f"Plotting not supported for {pareto_rewards.shape[-1]} objectives")
        return None


def pareto_frontier(solutions, rewards, maximize=True):
    """Extract the non-dominated samples and scores."""
    if not BOTORCH_AVAILABLE:
        print("Warning: botorch not available, using simple pareto calculation")
        return _simple_pareto_frontier(solutions, rewards, maximize)

    pareto_mask = pareto.is_non_dominated(
        torch.tensor(rewards) if maximize else -torch.tensor(rewards)
    )
    pareto_front = solutions[pareto_mask]
    pareto_rewards = rewards[pareto_mask]
    return pareto_front, pareto_rewards


def _simple_pareto_frontier(solutions, rewards, maximize=True):
    """Compute a fallback Pareto frontier when botorch is unavailable."""
    rewards = np.array(rewards)
    if not maximize:
        rewards = -rewards

    pareto_mask = np.zeros(len(rewards), dtype=bool)

    for i, point in enumerate(rewards):
        is_dominated = False
        for j, other_point in enumerate(rewards):
            if i != j and np.all(other_point >= point) and np.any(other_point > point):
                is_dominated = True
                break
        if not is_dominated:
            pareto_mask[i] = True

    pareto_front = solutions[pareto_mask]
    pareto_rewards = rewards[pareto_mask]
    if not maximize:
        pareto_rewards = -pareto_rewards

    return pareto_front, pareto_rewards


def compute_hypervolume(pareto_rewards, reference_point=None):
    """Compute the internal hypervolume metric.
    
    For more than two objectives, this fallback sums box volumes without subtracting overlap.
    """
    if reference_point is None:
        reference_point = np.min(pareto_rewards, axis=0) - 0.1

    # Use a two-dimensional calculation or a higher-dimensional volume-sum approximation.
    if pareto_rewards.shape[1] == 2:
        # Two-objective case.
        sorted_indices = np.argsort(pareto_rewards[:, 0])
        sorted_rewards = pareto_rewards[sorted_indices]

        hypervolume = 0.0
        for i, point in enumerate(sorted_rewards):
            if i == 0:
                width = point[0] - reference_point[0]
            else:
                width = point[0] - sorted_rewards[i - 1][0]
            height = point[1] - reference_point[1]
            hypervolume += width * height

        return hypervolume
    else:
        # For higher dimensions, sum reference-relative box volumes without overlap correction.
        volumes = np.prod(pareto_rewards - reference_point[None, :], axis=1)
        return np.sum(np.maximum(volumes, 0))
