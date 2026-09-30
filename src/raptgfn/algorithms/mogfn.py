"""Multi-objective GFlowNet for aptamer sequence generation."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from torch.distributions import Categorical
from tqdm import tqdm

from raptgfn.algorithms.base import BaseAlgorithm
from raptgfn.algorithms.conditional_transformer import CondGFNTransformer
from raptgfn.algorithms.mogfn_utils import (
    generate_simplex,
    mean_pairwise_distances,
    pareto_frontier,
    plot_pareto,
    thermometer,
)
from raptgfn.metrics import get_all_metrics
from raptgfn.utils import str_to_tokens


class MOGFN(BaseAlgorithm):
    """Multi-Objective GFlowNet for DNA Aptamer Generation"""

    def __init__(self, cfg, task, tokenizer, **kwargs):
        super(MOGFN, self).__init__(cfg, task, tokenizer, **kwargs)
        self.checkpoint_path = cfg.get("checkpoint_path")
        self.setup_vars()
        self.init_policy()

    def setup_vars(self):
        """Initialize algorithm state and configuration."""
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Task configuration.
        self.max_len = self.task.max_len
        self.min_len = self.task.min_len
        self.obj_dim = self.task.obj_dim

        # GFlowNet configuration.
        self.train_steps = cfg.train_steps
        # Target total steps for resumed training, or None if unspecified.
        self.total_train_steps = getattr(cfg, "total_train_steps", None)
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.therm_n_bins = cfg.therm_n_bins
        self.beta_use_therm = cfg.beta_use_therm
        self.pref_use_therm = cfg.pref_use_therm
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.beta_cond = cfg.beta_cond
        self.pref_cond = cfg.pref_cond
        self.beta_scale = cfg.beta_scale
        self.beta_shape = cfg.beta_shape
        self.pref_alpha = cfg.pref_alpha
        self.beta_max = cfg.beta_max
        self.reward_type = cfg.reward_type

        # Hybrid-learning configuration.
        self.offline_dataset_path = cfg.get("offline_dataset_path", None)
        self.mle_coeff = cfg.get("mle_coeff", 0.0)
        self.gfn_loss_coeff = cfg.get("gfn_loss_coeff", 1.0)
        self.offline_batch_size_ratio = cfg.get("offline_batch_size_ratio", 0.0)
        # Schedules for loss coefficients and the offline mixing ratio.
        self.mle_coeff_sched = getattr(cfg, "mle_coeff_schedule", {"enabled": False})
        self.gfn_coeff_sched = getattr(cfg, "gfn_loss_coeff_schedule", {"enabled": False})
        self.offline_ratio_sched = getattr(
            cfg, "offline_batch_size_ratio_schedule", {"enabled": False}
        )

        # Optimization parameters.
        self.pi_lr = cfg.pi_lr
        self.z_lr = cfg.z_lr
        self.wd = cfg.wd

        # Training-time evaluation settings.
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.num_samples = cfg.num_samples
        self.k = cfg.k
        self.simplex_bins = cfg.simplex_bins
        self.use_eval_pref = cfg.use_eval_pref
        self.num_pareto_points = cfg.num_pareto_points
        self.pareto_freq = cfg.pareto_freq
        self.unnormalize_rewards = cfg.unnormalize_rewards
        # Early stopping.
        self.early_stop_patience = getattr(cfg, "early_stop_patience", None)
        self.early_stop_min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
        # Output artifacts.
        self.artifacts_cfg = getattr(cfg, "artifacts", {"enabled": True, "dir": ".", "topk": 50})
        # Scheduler configuration.
        self.beta_sched = getattr(cfg, "beta_scheduler", {"enabled": False})
        self.pref_alpha_sched = getattr(cfg, "pref_alpha_schedule", {"enabled": False})
        self.ema_abs_logr = 0.0
        # Sampling strategy.
        self.sampling_cfg = getattr(
            cfg, "sampling", {"top_k": None, "top_p": None, "temp_schedule": {"enabled": False}}
        )
        # State tracked for logging.
        self.last_beta_scalar = None
        self.last_sampling_temp = self.sampling_temp
        self.last_subtb_loss = 0.0
        self.last_logZ_mean = 0.0
        self.last_logZ_std = 0.0
        self.last_tb_resid_abs_mean = 0.0
        self.last_tb_resid_abs_std = 0.0

        # Model configuration.
        self.model_cfg = cfg.model

    def init_policy(self):
        """Initialize the policy network and optimizers."""
        # Construct the model.
        self.model = CondGFNTransformer(
            max_len=self.max_len,
            vocab_size=len(self.tokenizer.full_vocab),
            num_actions=len(self.tokenizer.non_special_vocab) + 1,  # +1 for stop action
            num_hid=self.model_cfg.num_hid,
            num_layers=self.model_cfg.num_layers,
            num_head=self.model_cfg.num_head,
            dropout=self.model_cfg.dropout,
            batch_size=self.model_cfg.batch_size,
            objectives=self.task.objectives,
            therm_n_bins=self.therm_n_bins,
            beta_use_therm=self.beta_use_therm,
            pref_use_therm=self.pref_use_therm,
            enable_backward=getattr(self.cfg, "learn_backward", {}).get("enabled", False),
        ).to(self.device)

        # Construct the optimizers.
        self.opt = torch.optim.Adam(self.model.model_params(), lr=self.pi_lr, weight_decay=self.wd)
        self.opt_Z = torch.optim.Adam(
            self.model.Z_param(),
            lr=self.z_lr * float(getattr(self.cfg, "z_lr_scale", 1.0)),
            weight_decay=self.wd,
        )
        # Optionally scale the backward-policy learning rate.
        if getattr(self.cfg, "learn_backward", {}).get("enabled", False):
            for g in self.opt.param_groups:
                g["lr"] = g["lr"] * float(
                    getattr(self.cfg, "learn_backward", {}).get("lr_scale", 1.0)
                )

        # Restore a checkpoint after creating the optimizers.
        if self.checkpoint_path:
            print(f"Loading model from {self.checkpoint_path}")
            # Account for the changed torch.load default in PyTorch 2.6 and later.
            # Try weights_only=True first, allowing the configuration types used by checkpoints.
            checkpoint = None
            try:
                from torch.serialization import add_safe_globals

                try:
                    # Allow OmegaConf DictConfig when present in a trusted checkpoint.
                    import omegaconf

                    add_safe_globals([omegaconf.dictconfig.DictConfig])
                except Exception:
                    pass
                checkpoint = torch.load(
                    self.checkpoint_path, map_location=self.device, weights_only=True
                )
            except Exception:
                # Fallback to weights_only=False only for trusted checkpoints.
                # This fallback can execute arbitrary code; never use untrusted checkpoints.
                checkpoint = torch.load(
                    self.checkpoint_path, map_location=self.device, weights_only=False
                )
            # Support both checkpoint dictionaries and legacy state dictionaries.
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
                if "optimizer_Z_state_dict" in checkpoint:
                    self.opt_Z.load_state_dict(checkpoint["optimizer_Z_state_dict"])
                if "step" in checkpoint:
                    self.resume_step = checkpoint["step"]
            else:
                # Legacy format: the checkpoint itself is a state dictionary.
                self.model.load_state_dict(checkpoint)
            print(f"Model and optimizer loaded from {self.checkpoint_path}")

        # Construct simplex points for multi-objective evaluation.
        self.simplex = generate_simplex(self.obj_dim, self.simplex_bins)

        # Evaluation preferences.
        if hasattr(self.task, "eval_pref") and self.task.eval_pref is not None:
            self.eval_pref = torch.tensor(self.task.eval_pref).float()
        else:
            self.eval_pref = torch.ones(self.obj_dim).float() / self.obj_dim

    def optimize(self, task, init_data=None):
        """Run the optimization loop."""
        # Enable task training mode.
        if hasattr(task, "set_training_mode"):
            task.set_training_mode(True)
        losses = []
        rewards = []

        desc_str = "rew {:.2f}, hv {:.2f}, r2 {:.2f}, hsri {:.2f}, loss {:.2f}, rs {:.2f}"
        # When resuming, use the remaining target steps; otherwise use train_steps.
        start_step = getattr(self, "resume_step", 0)
        if self.total_train_steps is not None:
            remaining = max(0, int(self.total_train_steps) - int(start_step))
        else:
            remaining = self.train_steps
        pb = tqdm(range(remaining), desc=desc_str.format(0, 0, 0, 0, 0, 0))

        # Initialize evaluation bookkeeping.
        hv, r2, hsri = 0, 0, 0
        best_hv = getattr(self, "best_hv", None)
        no_improve_count = 0
        # Optional initial log-Z bias adjustment based on the mean log reward.
        z_bias_inited = False

        for i in pb:
            # Update coefficient schedules.
            def _interp(start, end, t, mode="linear"):
                if mode == "cosine":
                    import math

                    w = 0.5 - 0.5 * math.cos(math.pi * t)
                else:
                    w = t
                return start * (1 - w) + end * w

            # Use remaining steps when a schedule has no explicit total_train_steps.
            def _t_from_sched(sched, step, default_total):
                total = sched.get("total_steps", default_total) or default_total
                return min(1.0, (step + 1) / float(total))

            if self.mle_coeff_sched.get("enabled", False):
                mode = self.mle_coeff_sched.get("mode", "linear")
                start_v = float(self.mle_coeff_sched.get("start", float(self.mle_coeff)))
                end_v = float(self.mle_coeff_sched.get("end", float(self.mle_coeff)))
                t = _t_from_sched(self.mle_coeff_sched, i, remaining)
                self.mle_coeff = float(_interp(start_v, end_v, t, mode))
            if self.gfn_coeff_sched.get("enabled", False):
                mode = self.gfn_coeff_sched.get("mode", "linear")
                start_v = float(self.gfn_coeff_sched.get("start", float(self.gfn_loss_coeff)))
                end_v = float(self.gfn_coeff_sched.get("end", float(self.gfn_loss_coeff)))
                t = _t_from_sched(self.gfn_coeff_sched, i, remaining)
                self.gfn_loss_coeff = float(_interp(start_v, end_v, t, mode))
            if self.offline_ratio_sched.get("enabled", False):
                mode = self.offline_ratio_sched.get("mode", "linear")
                start_v = float(
                    self.offline_ratio_sched.get("start", float(self.offline_batch_size_ratio))
                )
                end_v = float(
                    self.offline_ratio_sched.get("end", float(self.offline_batch_size_ratio))
                )
                t = _t_from_sched(self.offline_ratio_sched, i, remaining)
                self.offline_batch_size_ratio = float(_interp(start_v, end_v, t, mode))

            # Update the Dirichlet concentration schedule.
            if self.pref_alpha_sched.get("enabled", False):
                mode = self.pref_alpha_sched.get("mode", "linear")
                total = self.pref_alpha_sched.get("total_steps", remaining) or remaining
                start_a = float(self.pref_alpha_sched.get("start", 1.0))
                end_a = float(self.pref_alpha_sched.get("end", 10.0))
                t = min(1.0, (i + 1) / float(total))
                if mode == "cosine":
                    import math

                    w = 0.5 - 0.5 * math.cos(math.pi * t)
                else:
                    w = t
                self.pref_alpha = start_a * (1 - w) + end_a * w

            # Update the sampling-temperature schedule.
            if self.sampling_cfg.get("temp_schedule", {}).get("enabled", False):
                mode = self.sampling_cfg["temp_schedule"].get("mode", "linear")
                total = (
                    self.sampling_cfg["temp_schedule"].get("total_steps", remaining) or remaining
                )
                t = min(1.0, (i + 1) / float(total))
                start_t = float(self.sampling_cfg["temp_schedule"].get("start", 1.0))
                end_t = float(self.sampling_cfg["temp_schedule"].get("end", 1.0))
                if mode == "cosine":
                    import math

                    w = 0.5 - 0.5 * math.cos(math.pi * t)
                else:
                    w = t
                self.sampling_temp = start_t * (1 - w) + end_t * w
            # Store the effective sampling temperature.
            self.last_sampling_temp = self.sampling_temp

            res = self.train_step(task, self.batch_size)
            if len(res) == 4:
                total_loss, online_loss, mle_loss, r = res
                r_abs = abs(r)
            else:
                total_loss, online_loss, mle_loss, r, r_abs = res
            losses.append(total_loss)
            rewards.append(r)
            # Update the moving average used by the adaptive beta schedule.
            if self.beta_sched.get("enabled", False):
                decay = float(self.beta_sched.get("adaptive", {}).get("ema_decay", 0.99))
                self.ema_abs_logr = decay * self.ema_abs_logr + (1 - decay) * float(abs(r_abs))

            # Log training metrics at every step.
            metrics = {
                "total_train_loss": total_loss,
                "online_tb_loss": online_loss,
                "offline_mle_loss": mle_loss,
                "train_rewards": r,
                "step": i,
                "learning_rate": self.opt.param_groups[0]["lr"],
                "z_learning_rate": self.opt_Z.param_groups[0]["lr"],
                "ema_abs_logr": self.ema_abs_logr,
                "pref_alpha": float(self.pref_alpha) if hasattr(self, "pref_alpha") else None,
                "sampling_temp": float(self.last_sampling_temp),
                "z_logZ_mean": float(self.last_logZ_mean),
                "z_logZ_std": float(self.last_logZ_std),
                "subtb_loss": (
                    float(self.last_subtb_loss)
                    if getattr(self.cfg, "subtb", {}).get("enabled", False)
                    else 0.0
                ),
                "backward_enabled": int(
                    getattr(self.cfg, "learn_backward", {}).get("enabled", False)
                ),
                "tb_resid_abs_mean": float(self.last_tb_resid_abs_mean),
                "tb_resid_abs_std": float(self.last_tb_resid_abs_std),
                "sampling_top_p": (
                    float(self.sampling_cfg.get("top_p"))
                    if self.sampling_cfg.get("top_p") is not None
                    else None
                ),
                "sampling_top_k": (
                    int(self.sampling_cfg.get("top_k"))
                    if self.sampling_cfg.get("top_k") is not None
                    else None
                ),
                "mle_coeff_cur": float(self.mle_coeff),
                "gfn_loss_coeff_cur": float(self.gfn_loss_coeff),
                "offline_ratio_cur": float(self.offline_batch_size_ratio),
            }
            if self.last_beta_scalar is not None:
                metrics["beta_scheduled"] = float(self.last_beta_scalar)
            self.log(metrics)

            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    if hasattr(task, "set_training_mode"):
                        task.set_training_mode(False)
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(
                        task, plot=True
                    )
                    if hasattr(task, "set_training_mode"):
                        task.set_training_mode(True)

                hv = mo_metrics.get("hypervolume", 0)
                r2 = mo_metrics.get("r2", 0)
                hsri = mo_metrics.get("hsri", 0)

                # Save the best checkpoint when the logged hypervolume metric improves.
                if best_hv is None or hv > best_hv + self.early_stop_min_delta:
                    self.best_hv = hv
                    best_hv = hv
                    no_improve_count = 0
                    output_dir = HydraConfig.get().runtime.output_dir
                    best_path = os.path.join(output_dir, "model_checkpoint.best.pth")
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.opt.state_dict(),
                            "optimizer_Z_state_dict": self.opt_Z.state_dict(),
                            "cfg": self.cfg,
                            "step": getattr(self, "resume_step", 0) + i,
                            "best_hv": self.best_hv,
                        },
                        best_path,
                    )
                    print(f"Best model (HV={hv:.4f}) saved to {best_path}")
                else:
                    no_improve_count += 1

                # Log evaluation-time metrics.
                eval_metrics = {
                    "topk_rewards": np.mean(topk_metrics[0]) if len(topk_metrics[0]) > 0 else 0.0,
                    "topk_diversity": np.mean(topk_metrics[1]) if len(topk_metrics[1]) > 0 else 0.0,
                    "sample_r": np.mean(rs) if len(rs) > 0 else 0.0,
                    "rs": np.mean(rs) if len(rs) > 0 else 0.0,
                    "eval_step": i,
                    **mo_metrics,
                }

                # Record sampled sequences in a W&B table.
                if i % (self.eval_freq * 5) == 0:  # Once every five evaluation intervals.
                    sample_table = []
                    # Rank the scores returned by evaluation() by their unweighted sum.
                    if len(all_rews) > 0:
                        scalar = np.sum(all_rews, axis=1)
                        topk = min(10, len(scalar))
                        top_idx = np.argsort(scalar)[-topk:][::-1]
                        for rank, idx in enumerate(top_idx, start=1):
                            seq = samples[idx]
                            length = len(seq)
                            row = [rank, seq, length]
                            # Retain objective scores in the configured order.
                            row.extend([float(v) for v in all_rews[idx].tolist()])
                            sample_table.append(row)

                    try:
                        import wandb

                        if hasattr(wandb, "Table"):
                            # Construct table columns from objective names.
                            objective_columns = []
                            for obj in task.objectives:
                                # Format each objective name as a readable score-column label.
                                formatted_name = (
                                    obj.replace("_", " ").title().replace(" ", "_") + "_Score"
                                )
                                objective_columns.append(formatted_name)

                            columns = ["Rank", "Sequence", "Length"] + objective_columns
                            eval_metrics["sample_sequences"] = wandb.Table(
                                columns=columns, data=sample_table
                            )
                    except ImportError:
                        pass

                self.log(eval_metrics, commit=False)

                if fig is not None:
                    self.log({"pareto_front": fig}, commit=False)

                # Save the Pareto-front output.
                if i % self.pareto_freq == 0:
                    self._save_pareto_results(task)
                # Write CSV artifacts.
                if (
                    getattr(self.artifacts_cfg, "enabled", True)
                    if hasattr(self.artifacts_cfg, "enabled")
                    else self.artifacts_cfg.get("enabled", True)
                ):
                    out_dir = HydraConfig.get().runtime.output_dir
                    # Use all_rews returned by evaluation() as a NumPy array.
                    from raptgfn.algorithms.mogfn_utils import pareto_frontier

                    pareto_samples, pareto_rewards = pareto_frontier(
                        np.array(samples), np.array(all_rews), maximize=True
                    )
                    self._save_eval_csvs(
                        out_dir, samples, np.array(all_rews), np.array(pareto_rewards), task
                    )
                    eval_metrics["artifacts_saved"] = 1

                # Stop when the configured patience is exhausted.
                if (
                    self.early_stop_patience is not None
                    and no_improve_count >= self.early_stop_patience
                ):
                    print(
                        f"Early stopping at step {i} (no improvement for {no_improve_count} evals)"
                    )
                    break

                # Apply the optional log-Z bias initialization only during early steps.
                if (not z_bias_inited) and getattr(self.cfg, "z_bias_init", {}).get(
                    "enabled", False
                ):
                    max_steps = int(getattr(self.cfg, "z_bias_init", {}).get("max_steps", 200))
                    if i >= max_steps:
                        z_bias_inited = True
                        continue
                    try:
                        # Estimate the mean log reward from a small sampled batch.
                        cond_var_tmp, (prefs_tmp, beta_tmp) = self._get_condition_var(
                            train=True, bs=32
                        )
                        states_tmp, _ = self.sample(32, cond_var_tmp)
                        log_r_tmp, vm = self.process_reward(states_tmp, prefs_tmp, task)
                        mu = float(log_r_tmp[vm].mean().item()) if vm.any() else 0.0
                        # Shift the Z_mod bias uniformly rather than estimating an offset at zero conditions.
                        if hasattr(self.model.Z_mod, "bias") and self.model.Z_mod.bias is not None:
                            with torch.no_grad():
                                (
                                    self.model.Z_mod.bias.add_(mu / max(1, self.model.num_hid))
                                    if self.model.Z_type == "linear"
                                    else (
                                        self.model.Z_mod.model[-1].bias.add_(
                                            mu / max(1, self.model.num_hid)
                                        )
                                        if self.model.Z_type == "mlp"
                                        else self.model.Z_mod.bias.add_(mu)
                                    )
                                )
                        z_bias_inited = True
                    except Exception:
                        z_bias_inited = True

                pb.set_description(
                    desc_str.format(r, hv, r2, hsri, np.mean(losses[-10:]), np.mean(rewards[-10:]))
                )

        # Save the final model.
        output_dir = HydraConfig.get().runtime.output_dir
        model_save_path = os.path.join(output_dir, "model_checkpoint.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "optimizer_Z_state_dict": self.opt_Z.state_dict(),
                "cfg": self.cfg,
                "step": getattr(self, "resume_step", 0) + self.train_steps,
            },
            model_save_path,
        )
        print(f"Model saved to {model_save_path}")

        return {"losses": losses, "train_rs": rewards, "hypervol_rel": hv}

    def train_step(self, task, batch_size):
        """Perform one training step."""
        total_loss = torch.tensor(0.0, device=self.device)
        online_loss = torch.tensor(0.0, device=self.device)
        mle_loss = torch.tensor(0.0, device=self.device)
        log_r_mean = 0
        log_r_abs_mean = 0

        # Split the minibatch into online and offline portions.
        offline_bs = int(batch_size * self.offline_batch_size_ratio)
        online_bs = batch_size - offline_bs

        # Online trajectory-balance loss.
        if online_bs > 0:
            cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=online_bs)
            online_states, online_log_p_F = self.sample(online_bs, cond_var)

            log_r, valid_mask = self.process_reward(online_states, prefs, task)
            log_r = log_r.to(self.device)
            valid_mask = valid_mask.to(self.device)
            log_r_mean = log_r[valid_mask].mean().item() if valid_mask.any() else 0.0

            log_Z = self.model.Z(cond_var)
            # Partition-function statistics for logging.
            try:
                self.last_logZ_mean = log_Z.mean().item()
                self.last_logZ_std = log_Z.std().item() if log_Z.numel() > 1 else 0.0
            except Exception:
                self.last_logZ_mean = 0.0
                self.last_logZ_std = 0.0
            # Prepare learned backward probabilities for TB and prefix residuals.
            compute_true = (
                getattr(self.cfg, "learn_backward", {}).get("enabled", False)
                or getattr(self.cfg, "subtb", {}).get("true_enabled", False)
            ) and getattr(self.model, "enable_backward", False)
            if compute_true:
                tb_loss_true, subtb_loss_true, diffs_vec = self._compute_true_tb_and_subtb(
                    online_states, cond_var, prefs, beta, valid_mask, log_r
                )
            else:
                tb_loss_true = torch.tensor(0.0, device=self.device)
                subtb_loss_true = torch.tensor(0.0, device=self.device)
                diffs_vec = None

            # Fallback residual calculation.
            numerator = log_Z + online_log_p_F
            denominator = beta * log_r

            if valid_mask.any():
                if compute_true:
                    online_loss = tb_loss_true
                else:
                    diff = (numerator - denominator)[valid_mask]
                    online_loss = diff.pow(2).mean()
                    # Residual statistics.
                    with torch.no_grad():
                        absd = diff.abs()
                        self.last_tb_resid_abs_mean = float(absd.mean().item())
                        self.last_tb_resid_abs_std = (
                            float(absd.std().item()) if absd.numel() > 1 else 0.0
                        )
                # Add the prefix residual term when enabled.
                if getattr(self.cfg, "subtb", {}).get("true_enabled", False):
                    online_loss = online_loss + subtb_loss_true
                total_loss += self.gfn_loss_coeff * online_loss
                # Optional legacy squared-residual approximation.
                if getattr(self.cfg, "subtb", {}).get("enabled", False) and not getattr(
                    self.cfg, "subtb", {}
                ).get("true_enabled", False):
                    max_pref = int(getattr(self.cfg, "subtb", {}).get("max_prefixes", 4))
                    subtb_weight = float(getattr(self.cfg, "subtb", {}).get("weight", 0.05))
                    use_all = bool(getattr(self.cfg, "subtb", {}).get("use_all_prefixes", False))
                    seq_lens = torch.tensor([len(s) for s in online_states], device=self.device)
                    num_prefix = seq_lens if use_all else torch.clamp(seq_lens, max=max_pref)
                    # Scale the squared residual by the configured coefficient.
                    diff = (numerator - denominator)[valid_mask]
                    subtb_loss = (
                        diff.pow(2)
                        * (
                            num_prefix[valid_mask].float()
                            / torch.clamp(seq_lens[valid_mask].float(), min=1.0)
                        )
                    ).mean()
                    total_loss += subtb_weight * subtb_loss
                    try:
                        self.last_subtb_loss = float((subtb_weight * subtb_loss).item())
                    except Exception:
                        self.last_subtb_loss = 0.0
                else:
                    try:
                        self.last_subtb_loss = (
                            float(subtb_loss_true.item())
                            if isinstance(subtb_loss_true, torch.Tensor)
                            else float(subtb_loss_true)
                        )
                    except Exception:
                        self.last_subtb_loss = 0.0
                # Log the correlation between log Z and the TB residual.
                try:
                    if compute_true and diffs_vec is not None and diffs_vec.numel() > 1:
                        zv = log_Z[valid_mask].detach()
                        # diffs_vec contains valid samples only.
                        dv = diffs_vec.detach().abs()
                        if zv.numel() == dv.numel():
                            zv = (zv - zv.mean()) / (zv.std() + 1e-8)
                            dv = (dv - dv.mean()) / (dv.std() + 1e-8)
                            corr = float((zv * dv).mean().item())
                            self.log({"tb_resid_z_corr": corr})
                except Exception:
                    pass
            else:
                online_loss = torch.tensor(0.0, device=self.device)

        # Offline maximum-likelihood loss.
        if offline_bs > 0 and self.mle_coeff > 0 and self.task.offline_sequences:
            offline_states = self.task.sample_offline_sequences(offline_bs)
            if offline_states:
                # Keep only A/C/G/T sequences within the configured length bounds.
                filtered_offline = [
                    s
                    for s in offline_states
                    if isinstance(s, str)
                    and set(s.upper()) <= set("ATGC")
                    and (self.min_len <= len(s) <= self.max_len)
                ]
                if len(filtered_offline) > 0:
                    # Generate conditions for offline sequences.
                    cond_var, _ = self._get_condition_var(train=True, bs=len(filtered_offline))
                    # Compute teacher-forced cross entropy for the batch.
                    offline_nll = self._compute_ce_for_sequences(filtered_offline, cond_var)
                    mle_loss = offline_nll.mean()
                    total_loss += self.mle_coeff * mle_loss

        # Backpropagate and update parameters.
        if total_loss.requires_grad:
            # Optional log-Z anchor regularization.
            if getattr(self.cfg, "z_anchor", {}).get("enabled", False):
                weight = float(getattr(self.cfg, "z_anchor", {}).get("weight", 0.0))
                if weight > 0:
                    # Obtain the anchor target from the configured statistic, such as ema_abs_logr.
                    c = (
                        float(self.ema_abs_logr)
                        if getattr(self.cfg, "z_anchor", {}).get("source", "beta_logr_ema")
                        == "beta_logr_ema"
                        else 0.0
                    )
                    anchor = 0.0
                    # Use log Z at the mean condition as a lightweight approximation.
                    with torch.no_grad():
                        cond0, _ = self._get_condition_var(train=True, bs=16)
                    z_val = self.model.Z(cond0).mean()
                    anchor = weight * (z_val - c) ** 2
                    total_loss = total_loss + anchor
            self.opt.zero_grad()
            self.opt_Z.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
            self.opt.step()
            self.opt_Z.step()

        return total_loss.item(), online_loss.item(), mle_loss.item(), log_r_mean

    def _compute_true_tb_and_subtb(self, states, cond_var, prefs, beta, valid_mask, log_r):
        """Compute trajectory-balance and auxiliary prefix residual losses.
        
        TB combines trajectory sums of forward and backward log probabilities with beta times the terminal log reward. The auxiliary local prefix residual omits the terminal reward; it is not a standard reward-complete Subtrajectory Balance objective.
        
        Returns:
            Mean TB loss, the auxiliary prefix loss, and the residual vector.
        """
        B = len(states)
        device = self.device
        # Prepare sequence lengths and tokens.
        seq_lens = torch.tensor([len(s) for s in states], device=device)
        x_tokens = str_to_tokens(states, self.tokenizer).to(device)
        seq_len = x_tokens.size(1)
        ar = torch.arange(seq_len, device=device).unsqueeze(0)
        mask_full = ar >= (seq_lens + 1).unsqueeze(1)  # +1 for [CLS]

        # Compute differentiable forward and backward log probabilities at every position.
        f_logits_all = self.model(
            x_tokens, cond_var, mask_full, return_all=True, lens=seq_lens, logsoftmax=True
        )
        logp_f_all = f_logits_all  # [B, T, A]
        logp_b_all = None
        if getattr(self.cfg, "learn_backward", {}).get("enabled", False) and getattr(
            self.model, "enable_backward", False
        ):
            b_logits_all = self.model.backward_logits(
                x_tokens, cond_var, mask_full, return_all=True, lens=seq_lens, logsoftmax=True
            )
            logp_b_all = b_logits_all

        # Construct the action sequence.
        traj = getattr(self, "last_traj_info", None)
        tb_terms = []
        subtb_terms_all = []
        diffs = []
        max_pref = int(getattr(self.cfg, "subtb", {}).get("max_prefixes", 4))
        use_all = bool(getattr(self.cfg, "subtb", {}).get("use_all_prefixes", False))
        subtb_weight = float(getattr(self.cfg, "subtb", {}).get("weight", 0.05))

        for i in range(B):
            if not bool(valid_mask[i].item()):
                continue
            actions = traj[i]["actions"] if i < len(traj) else []
            if len(actions) == 0:
                continue
            T_i = len(actions)
            # forward log prob per step
            lp_f = [logp_f_all[i, t, actions[t]] for t in range(T_i)]
            lp_f = torch.stack(lp_f)
            # backward log prob per step (optional)
            if logp_b_all is not None:
                lp_b = [logp_b_all[i, t, actions[t]] for t in range(T_i)]
                lp_b = torch.stack(lp_b)
            else:
                lp_b = torch.zeros_like(lp_f)
            # Z and reward
            log_Z_i = self.model.Z(cond_var[i : i + 1]).squeeze(0)
            diff_i = (log_Z_i + lp_f.sum()) - (beta[i] * log_r[i] + lp_b.sum())
            tb_terms.append(diff_i**2)
            diffs.append(diff_i.detach())
            # Auxiliary local residual for selected prefixes; no terminal reward is included.
            if getattr(self.cfg, "subtb", {}).get("true_enabled", False):
                if use_all:
                    prefix_idx = list(range(1, T_i + 1))
                else:
                    step = max(1, (T_i + max_pref - 1) // max_pref)
                    prefix_idx = list(range(step, T_i + 1, step))
                for t_end in prefix_idx:
                    num = log_Z_i + lp_f[:t_end].sum()
                    den = lp_b[:t_end].sum()  # Use prefix backward log probabilities without a reward term.
                    subtb_terms_all.append((num - den) ** 2)

        tb_loss_mean = (
            torch.mean(torch.stack(tb_terms))
            if len(tb_terms) > 0
            else torch.tensor(0.0, device=device)
        )
        subtb_loss_term = (
            (subtb_weight * torch.mean(torch.stack(subtb_terms_all)))
            if len(subtb_terms_all) > 0
            else torch.tensor(0.0, device=device)
        )
        diffs_vec = torch.stack(diffs) if len(diffs) > 0 else torch.tensor([], device=device)
        return tb_loss_mean, subtb_loss_term, diffs_vec

    def _compute_ce_for_sequences(self, sequences: list[str], cond_var):
        """Compute teacher-forced sequence negative log likelihood.
        
        Sum cross entropy for the correct next nucleotide or STOP action at each step.
        """
        device = self.device
        batch_size = len(sequences)
        # Start with the empty prefix containing only the classification token.
        states = [""] * batch_size
        x = str_to_tokens(states, self.tokenizer).to(device)
        lens = torch.ones(batch_size, dtype=torch.long, device=device)
        total_nll = torch.zeros(batch_size, device=device)
        seq_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)

        # Precompute nucleotide-to-action identifiers.
        char_to_id = {ch: idx for idx, ch in enumerate(self.tokenizer.sampling_vocab)}
        max_T = int(seq_lengths.max().item()) if batch_size > 0 else 0

        for t in range(max_T + 1):
            # Construct the active-position mask.
            seq_len = x.size(1)
            ar = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = ar >= lens.unsqueeze(1)
            # Use the logits at the current position.
            logits_all = self.model(x, cond_var, mask, return_all=True, lens=lens)
            batch_idx = torch.arange(batch_size, device=device)
            pos_idx = (lens - 1).clamp(min=0)
            logits = logits_all[batch_idx, pos_idx, :]

            # Construct the target actions.
            # Before the end: next nucleotide; at the end: STOP(0); afterwards: inactive.
            active = (t <= seq_lengths).to(device)
            target = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Select positions before the sequence ends.
            idx_lt = torch.nonzero(t < seq_lengths, as_tuple=False).squeeze(1)
            if idx_lt.numel() > 0:
                # Look up the next nucleotide action.
                next_chars = [sequences[i][t] for i in idx_lt.tolist()]
                next_ids = torch.tensor(
                    [char_to_id.get(ch, -1) + 1 for ch in next_chars], device=device
                )
                target[idx_lt] = next_ids
            # Mark positions after the sequence end as inactive.
            idx_gt = torch.nonzero(t > seq_lengths, as_tuple=False).squeeze(1)
            if idx_gt.numel() > 0:
                active[idx_gt] = False

            if active.any():
                ce = F.cross_entropy(logits[active], target[active], reduction="none")
                total_nll[active] += ce

            # Extend active prefixes that have not reached their sequence end.
            idx_upd = torch.nonzero((t < seq_lengths) & active, as_tuple=False).squeeze(1)
            if idx_upd.numel() > 0:
                for i in idx_upd.tolist():
                    states[i] += sequences[i][t]
                lens[idx_upd] += 1
                x = str_to_tokens(states, self.tokenizer).to(device)

        return total_nll

    def get_log_probs_for_trajectories(self, sequences: list[str], cond_var):
        """
        Computes the log-probabilities of given trajectories (sequences) under the current model.
        This is used for calculating the MLE loss for offline data.
        """
        device = self.device
        batch_size = len(sequences)

        # Initial states are empty strings
        states = [""] * batch_size
        x = str_to_tokens(states, self.tokenizer).to(device)
        lens = torch.ones(batch_size, dtype=torch.long, device=device)

        # Total log-probabilities for each trajectory
        traj_log_probs = torch.zeros(batch_size, device=device)

        # Mask for sequences that are still being processed
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        seq_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)

        # Loop through each step of sequence generation
        for t in range(self.max_len + 1):
            if not active_mask.any():
                break

            # Get model predictions for the current states
            mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            for i, length in enumerate(lens):
                if length < x.size(1):
                    mask[i, length:] = True

            logits = self.model(x, cond_var, mask, lens=lens)
            policy_dist = Categorical(logits=logits)

            # Determine the target action for the current step `t`
            target_actions = torch.zeros(batch_size, dtype=torch.long, device=device)

            for i in range(batch_size):
                if not active_mask[i]:
                    continue

                if t < seq_lengths[i]:
                    # Action is the next character in the sequence
                    action_char = sequences[i][t]
                    target_actions[i] = self.tokenizer.sampling_vocab.index(action_char) + 1
                    states[i] += action_char
                    lens[i] += 1
                elif t == seq_lengths[i]:
                    # Action is the STOP action
                    target_actions[i] = 0
                    active_mask[i] = False  # Trajectory finished
                else:
                    # This case should not be hit if max_len is respected, but as a safeguard
                    active_mask[i] = False

            # Calculate the log-probability of the target actions
            step_log_probs = policy_dist.log_prob(target_actions)

            # Add the log-prob to the total for active trajectories
            traj_log_probs += step_log_probs * active_mask.float()

            # Update token tensor for the next iteration
            x = str_to_tokens(states, self.tokenizer).to(device)

        return traj_log_probs

    def _get_condition_var(self, train=True, bs=1):
        """Sample and encode conditioning variables."""
        # Sample preferences from a continuous Dirichlet distribution in training and evaluation.
        concentration = torch.ones(self.obj_dim, device=self.device) * self.pref_alpha
        prefs = torch.distributions.Dirichlet(concentration).sample((bs,))

        # Construct inverse-temperature values.
        if self.beta_cond:
            # When enabled, obtain a fixed batch value from the beta scheduler.
            use_sched = self.beta_sched.get("enabled", False)
            if use_sched:
                beta_scalar = self._compute_scheduled_beta()
                # Store the value for logging.
                self.last_beta_scalar = float(beta_scalar)
                beta_raw_scaled = torch.full((bs,), beta_scalar, device=self.device)
                if self.beta_use_therm:
                    beta = thermometer(beta_raw_scaled, self.therm_n_bins, 0, self.beta_max)
                else:
                    beta = beta_raw_scaled
            else:
                if self.beta_use_therm:
                    beta_raw = (
                        torch.distributions.Beta(self.beta_shape, 1).sample((bs,)).to(self.device)
                    )
                    beta_raw_scaled = beta_raw * self.beta_max
                    beta = thermometer(beta_raw_scaled, self.therm_n_bins, 0, self.beta_max)
                else:
                    beta_raw_scaled = (
                        torch.distributions.Beta(self.beta_shape, 1).sample((bs,)).to(self.device)
                        * self.beta_max
                    )
                    beta = beta_raw_scaled
        else:
            beta_raw_scaled = torch.ones(bs).to(self.device) * self.sample_beta
            beta = beta_raw_scaled

        # Concatenate the encoded conditions.
        if self.pref_use_therm and self.pref_cond:
            prefs_encoded = thermometer(prefs, self.therm_n_bins, 0, 1)
            prefs_encoded = prefs_encoded.view(bs, -1)
        else:
            prefs_encoded = prefs

        if self.beta_use_therm:
            cond_var = torch.cat([beta, prefs_encoded], dim=1)
        else:
            cond_var = torch.cat([beta.unsqueeze(1), prefs_encoded], dim=1)

        return cond_var, (prefs, beta_raw_scaled)

    def _compute_scheduled_beta(self) -> float:
        """Compute beta from the configured schedule."""
        min_b = float(self.beta_sched.get("min_beta", 0.5))
        max_b = float(self.beta_sched.get("max_beta", 32.0))
        mode = self.beta_sched.get("mode", "adaptive")
        warmup_steps = int(self.beta_sched.get("warmup_steps", 0))
        # Normalized progress ranges from zero to one.
        t = 0.0
        if self.total_train_steps:
            done = getattr(self, "resume_step", 0)
            t = max(0.0, min(1.0, done / float(self.total_train_steps)))
        if mode == "warmup" or mode == "linear":
            if warmup_steps > 0 and self.total_train_steps:
                t = min(1.0, getattr(self, "resume_step", 0) / float(warmup_steps))
            beta = min_b * (1 - t) + max_b * t
        elif mode == "cosine":
            import math

            w = 0.5 - 0.5 * math.cos(math.pi * t)
            beta = min_b * (1 - w) + max_b * w
        else:  # adaptive
            target = float(self.beta_sched.get("adaptive", {}).get("target_logr", 1.0))
            ema = max(1e-8, float(self.ema_abs_logr))
            beta = target / ema
            beta = max(min_b, min(max_b, beta))
        return float(beta)

    def _save_eval_csvs(
        self,
        out_dir: str,
        samples: list[str],
        all_rewards: np.ndarray,
        pareto_rewards: np.ndarray,
        task,
    ):
        """Save evaluation artifacts as CSV files."""
        try:
            import pandas as pd
        except ImportError:
            return
        os.makedirs(out_dir, exist_ok=True)
        # all_samples.csv
        scalar = np.sum(all_rewards, axis=1) if len(all_rewards) > 0 else np.array([])
        df_all = pd.DataFrame({"sequence": samples, "len": [len(s) for s in samples]})
        for i, obj in enumerate(task.objectives):
            df_all[f"{obj}"] = all_rewards[:, i] if len(all_rewards) > 0 else []
        df_all["scalar_reward"] = scalar
        df_all.to_csv(os.path.join(out_dir, "all_samples.csv"), index=False)
        # Write rank-one points to pareto_front.csv.
        # Select the corresponding sequences.
        pareto_set = set(map(tuple, pareto_rewards.tolist())) if len(pareto_rewards) > 0 else set()
        rows = []
        if pareto_set:
            for s, r in zip(samples, all_rewards):
                if tuple(r.tolist()) in pareto_set:
                    row = {"sequence": s, "len": len(s), "rank": 1}
                    for i, obj in enumerate(task.objectives):
                        row[f"{obj}"] = r[i]
                    rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "pareto_front.csv"), index=False)

    def sample(self, episodes, cond_var=None, train=True):
        """Sample sequences from the forward policy."""
        states = [""] * episodes
        # Optionally retain traces for backward-policy and prefix-residual training.
        capture_traj = getattr(self.cfg, "learn_backward", {}).get("enabled", False) or getattr(
            self.cfg, "subtb", {}
        ).get("true_enabled", False)
        if capture_traj:
            self.last_traj_info = [
                {"actions": [], "stop_step": None, "logp_f_steps": []} for _ in range(episodes)
            ]
        traj_logprob = torch.zeros(episodes, device=self.device)

        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)

        active_mask = torch.ones(episodes, dtype=torch.bool, device=self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device)
        lens = torch.ones(episodes, dtype=torch.long, device=self.device)  # Start at one to account for the initial classification token.

        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in range(self.max_len):
            if not active_mask.any():
                break

            # Construct the padding mask.
            seq_len = x.size(1)
            ar = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, S]
            mask = ar >= lens.unsqueeze(1)  # [B, S]

            logits = self.model(x, cond_var, mask, lens=lens)
            # Apply top-k and top-p sampling filters.
            top_k = (
                None
                if self.sampling_cfg.get("top_k", None) in [None, 0]
                else int(self.sampling_cfg.get("top_k"))
            )
            top_p = self.sampling_cfg.get("top_p", None)
            if top_k is not None or top_p is not None:
                # Apply softmax to the log probabilities.
                probs = torch.softmax(logits, dim=-1)
                if top_k is not None:
                    kth = torch.topk(probs, k=top_k, dim=-1).values[..., -1:]
                    mask_k = probs < kth
                    probs = probs.masked_fill(mask_k, 0.0)
                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask_p = cum > float(top_p)
                    # keep the first token above threshold
                    mask_p[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask_p, 0.0)
                    # Map the mask back to the original vocabulary order.
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                # Renormalize the probabilities.
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                logits = torch.log(probs + 1e-12)

            # Apply minimum-length constraints to a copy of the logits.
            if t < self.min_len:
                logits = logits.clone()
                logits[:, 0] = -1000  # Disable the STOP action.

            # Sample the next action.
            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()

            # Mix in random actions during training.
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                random_actions = torch.randint(
                    int(t < self.min_len), logits.shape[1], (episodes,)
                ).to(self.device)
                actions = torch.where(uniform_mix, random_actions, actions)

            # Accumulate sampled-action log probabilities.
            log_probs = policy_dist.log_prob(actions)
            traj_logprob = traj_logprob + log_probs * active_mask.float()
            if capture_traj:
                for idx in torch.nonzero(active_mask, as_tuple=False).squeeze(1).tolist():
                    self.last_traj_info[idx]["actions"].append(int(actions[idx].item()))
                    self.last_traj_info[idx]["logp_f_steps"].append(float(log_probs[idx].item()))

            # Update sequence states.
            # STOP(0) marks a trajectory inactive.
            stop_mask = (actions == 0) & active_mask
            active_mask = active_mask & (~stop_mask)
            # Select trajectories that remain active.
            if active_mask.any():
                active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
                # Append nucleotides with a bounded loop over active trajectories.
                for idx in active_idx.tolist():
                    act = actions[idx].item()
                    if act > 0:
                        token = self.tokenizer.sampling_vocab[act - 1]
                        states[idx] += token
                        lens[idx] += 1
            # Record the stopping step for completed trajectories.
            if capture_traj and stop_mask.any():
                stop_idx = torch.nonzero(stop_mask, as_tuple=False).squeeze(1)
                for idx in stop_idx.tolist():
                    if self.last_traj_info[idx]["stop_step"] is None:
                        self.last_traj_info[idx]["stop_step"] = int(lens[idx].item())

            # Update the tokenized input.
            x = str_to_tokens(states, self.tokenizer).to(self.device)

            # Handle trajectories that reach the maximum length.
            if lens.max() >= self.max_len:
                break

        return states, traj_logprob

    def process_reward(self, states, prefs, task):
        """Compute and transform the conditional reward."""
        # Filter invalid states.
        valid_states = [s for s in states if len(s) >= self.min_len]
        if not valid_states:
            return torch.zeros(len(states)).to(self.device)

        # Compute objective-specific scores.
        scores = task.score(valid_states)
        scores = torch.tensor(scores).float().to(self.device)

        # Combine scores using the sampled preferences.
        if self.reward_type == "convex":
            weighted_scores = torch.sum(scores * prefs[: len(scores)], dim=1)
        else:
            weighted_scores = torch.sum(scores, dim=1)

        # Transform and clip the reward.
        # Avoid taking logarithms of nonpositive rewards.
        # Clamp to the larger of the configured reward floor and 1e-8.
        # This keeps every logarithm argument positive.
        safe_min_reward = max(self.reward_min, 1e-8)
        rewards = torch.clamp(weighted_scores, min=safe_min_reward)
        log_rewards = torch.log(rewards)

        # Handle invalid states.
        if len(valid_states) < len(states):
            # Use masking and a positive floor instead of returning negative infinity.
            mask = torch.zeros(len(states), dtype=torch.bool, device=self.device)
            mask[: len(valid_states)] = True
            full_rewards = torch.full((len(states),), math.log(safe_min_reward), device=self.device)
            full_rewards[mask] = log_rewards
            return full_rewards, mask

        return log_rewards, torch.ones(len(states), dtype=torch.bool, device=self.device)

    def evaluation(self, task, plot=False):
        """Sample sequences and compute scores for internal training and generation."""
        # Generate candidate samples.
        all_samples = []
        all_rewards = []

        # Support requests smaller than a single batch.
        remaining = self.num_samples
        while remaining > 0:
            cur_bs = min(self.batch_size, remaining)
            samples, _ = self.sample(cur_bs, train=False)
            valid_samples = [s for s in samples if len(s) >= self.min_len]

            if valid_samples:
                rewards = task.score(valid_samples)
                all_samples.extend(valid_samples)
                all_rewards.extend(rewards)
            remaining -= cur_bs

        if not all_samples:
            return [], [], [], {}, [[], []], None

        all_rewards = np.array(all_rewards)

        # Compute non-dominated samples and scores.
        pareto_samples, pareto_rewards = pareto_frontier(
            np.array(all_samples), all_rewards, maximize=True
        )

        # Compute internal monitoring metrics.
        mo_metrics = get_all_metrics(
            pareto_rewards,
            self.eval_metrics,
            hv_ref=np.min(all_rewards, axis=0) - 0.1,
            num_obj=self.obj_dim,
            r2_prefs=self.simplex,  # Preference directions for the R2 metric.
        )

        # Summarize top-scoring candidates.
        topk_rewards, topk_diversity = self._analyze_topk(all_samples, all_rewards)

        # Optionally construct a diagnostic plot.
        fig = None
        if plot and len(pareto_rewards) > 0:
            # Build axis labels from objective names.
            objective_names = []
            for obj in task.objectives:
                # Replace underscores with spaces and use title case.
                formatted_name = obj.replace("_", " ").title()
                objective_names.append(formatted_name)

            fig = plot_pareto(pareto_rewards, all_rewards, objective_names=objective_names)

        return (
            all_samples,
            all_rewards,
            all_rewards.mean(axis=0),
            mo_metrics,
            [topk_rewards, topk_diversity],
            fig,
        )

    def _analyze_topk(self, samples, rewards):
        """Summarize top-k rewards and sequence diversity."""
        if len(samples) == 0:
            return np.array([]), np.array([])

        # Select the top-k candidates for each objective.
        topk_rewards = []
        topk_diversity = []

        for obj_idx in range(self.obj_dim):
            obj_rewards = rewards[:, obj_idx]
            topk_indices = np.argsort(obj_rewards)[-self.k :]
            topk_samples = [samples[i] for i in topk_indices]

            topk_rewards.append(np.mean(obj_rewards[topk_indices]))
            topk_diversity.append(mean_pairwise_distances(topk_samples))

        return np.array(topk_rewards), np.array(topk_diversity)

    def _save_pareto_results(self, task):
        """Save the Pareto-front results."""
        samples, rewards = self.sample(self.num_pareto_points, train=False)
        valid_samples = [s for s in samples if len(s) >= self.min_len]

        if valid_samples:
            scores = task.score(valid_samples)
            pareto_samples, pareto_rewards = pareto_frontier(
                np.array(valid_samples), scores, maximize=True
            )

            self.update_state(
                {
                    "pareto_samples": pareto_samples.tolist(),
                    "pareto_rewards": pareto_rewards.tolist(),
                    "all_samples": valid_samples,
                    "all_rewards": scores.tolist(),
                }
            )
            self.save_state()
