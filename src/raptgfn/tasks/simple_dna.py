"""Aptamer generation task.

Users supply preprocessed inputs separately. This class loads score dictionaries and offline sequences and computes objective scores.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import RNA

from raptgfn.tasks.base import BaseTask


class SimpleDNATask(BaseTask):
    """Compute configurable sequence objectives.
    
    Available objectives include length, GC content, base balance, dinucleotide diversity, sequence complexity, k-mer dictionary scores, and RNAfold MFE. The manuscript configuration uses four k-mer scores and transformed MFE; the other objectives are generic implementation capabilities. K-mer objectives require paths to precomputed JSON dictionaries. More negative raw MFE produces a higher transformed MFE score.
    """

    def __init__(
        self,
        regex_list,
        max_len,
        min_len,
        num_start_examples,
        tokenizer,
        objectives,
        transform=lambda x: x,
        **kwargs,
    ):
        obj_dim = len(objectives)
        super().__init__(tokenizer, obj_dim, max_len, transform, **kwargs)
        self.regex_list = None
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs["max_score_per_dim"]
        self.score_max = kwargs.get("score_max", [1.0] * obj_dim)
        self.objectives = objectives
        # Optional automatic normalization.
        self.auto_norm = kwargs.get(
            "auto_norm", {"enabled": False, "mode": "quantile", "q_low": 0.1, "q_high": 0.9}
        )

        # Retain the evaluation preference supplied through the configuration.
        self.eval_pref = kwargs.get("eval_pref", None)

        # Objective-specific parameters.
        self.ideal_length = kwargs.get("ideal_length", (min_len + max_len) // 2)
        self.ideal_gc_content = kwargs.get("ideal_gc_content", 0.5)  # 50% GC content

        # RNAfold MFE parameters.
        self.rnafold_norm_factor = kwargs.get("rnafold_norm_factor", 20.0)
        # Configure the optional training-time RNAfold bypass.
        self.disable_rnafold_in_training = kwargs.get("disable_rnafold_in_training", True)
        self._training_mode = False

        # K-mer reward parameters.
        self.kmer_objectives = kwargs.get("kmer_objectives", {})
        self.min_kmer_reward = kwargs.get("min_kmer_reward", 1.0e-8)
        self.kmer_rewards = {}  # Loaded during task_setup.

        # Offline training data.
        self.offline_dataset_path = kwargs.get("offline_dataset_path", None)
        self.offline_sequences = None
        self.offline_sampling_cfg = kwargs.get("offline_sampling", {"equalize_length": False})

    def task_setup(self, *args, **kwargs):
        """Load objective dictionaries and the optional offline training set."""

        # Initialize each configured k-mer objective.
        kmer_objectives_to_process = [obj for obj in self.objectives if obj.startswith("kmer_")]

        for obj_name in kmer_objectives_to_process:
            if obj_name in self.kmer_objectives:
                config = self.kmer_objectives[obj_name]
                path = config.get("path")
                if path:
                    rewards = self._load_kmer_rewards(path)
                    if rewards:
                        self.kmer_rewards[obj_name] = rewards
                        print(f"Loaded reward data for objective '{obj_name}'.")
                    else:
                        print(
                            f"Warning: could not load reward data for objective '{obj_name}'. Removing this objective."
                        )
                        self._remove_objective(obj_name)
                else:
                    print(
                        f"Warning: objective '{obj_name}' has no configured 'path'. Removing this objective."
                    )
                    self._remove_objective(obj_name)
            else:
                print(
                    f"Warning: objective '{obj_name}' is not configured in 'kmer_objectives'. Removing this objective."
                )
                self._remove_objective(obj_name)

        # Load the offline training set.
        if self.offline_dataset_path:
            self._load_offline_dataset()

        return [], []

    def _remove_objective(self, obj_name):
        """Remove an unavailable objective and its associated configuration."""
        if obj_name in self.objectives:
            try:
                obj_idx = self.objectives.index(obj_name)
                self.objectives.pop(obj_idx)
                if obj_idx < len(self.score_max):
                    self.score_max.pop(obj_idx)
                self.obj_dim = len(self.objectives)

                if self.obj_dim < 1:  # A GFlowNet requires at least one objective.
                    raise ValueError("No objectives remain. At least one objective is required.")
            except ValueError:
                # Ignore an objective that has already been removed.
                pass

    def _load_offline_dataset(self):
        """Load the offline training sequences from a CSV file."""
        if self.offline_dataset_path is None:
            print("Warning: offline_dataset_path is not configured.")
            return

        if not os.path.exists(self.offline_dataset_path):
            print(
                f"Warning: offline dataset file not found: {self.offline_dataset_path}"
            )
            return

        print(f"Loading the offline dataset: {self.offline_dataset_path}")
        try:
            # Derive the filtered-cache filename.
            cache_path = self.offline_dataset_path.replace(".csv", "_filtered.csv")
            if os.path.exists(cache_path):
                print(f"Loading cached offline data: {cache_path}")
                df = pd.read_csv(cache_path)
            else:
                df = pd.read_csv(self.offline_dataset_path)
                # Validate the input column before filtering.
                if "sequence" not in df.columns:
                    print(
                        "Error: the offline dataset has no 'sequence' column."
                    )
                    return
                seqs = df["sequence"].dropna().astype(str)
                seqs = seqs[seqs.apply(lambda s: set(s.upper()) <= set("ATGC"))]
                seqs = seqs[(seqs.str.len() >= self.min_len) & (seqs.str.len() <= self.max_len)]
                seqs = seqs.drop_duplicates()
                seqs.to_csv(cache_path, index=False, header=["sequence"])
                print(
                    f"Cached filtered offline data: {cache_path} ({len(seqs)} sequences)"
                )
                df = pd.DataFrame({"sequence": seqs})
            self.offline_sequences = df["sequence"].tolist()
            # Record the sequence-length histogram.
            lengths = pd.Series([len(s) for s in self.offline_sequences])
            self.offline_len_hist = lengths.value_counts().sort_index()

            if not self.offline_sequences:
                print(
                    "Warning: the offline dataset contains no valid sequences within the configured length range."
                )
            else:
                print(f"Loaded the offline dataset ({len(self.offline_sequences)} sequences)")

        except Exception as e:
            print(f"Error: failed to load the offline dataset: {e}")

    def sample_offline_sequences(self, batch_size):
        """Sample sequences from the loaded offline dataset.
        
        Args:
            batch_size: Number of sequences to sample.
        
        Returns:
            A list of sampled sequence strings.
        """
        if not self.offline_sequences:
            return []
        if self.offline_sampling_cfg.get("equalize_length", False):
            # Optionally weight by inverse length frequency to flatten the length distribution.
            lengths = [len(s) for s in self.offline_sequences]
            import collections

            freq = collections.Counter(lengths)
            weights = [1.0 / freq[l] for l in lengths]
            # Normalize the sampling weights.
            total = sum(weights)
            probs = [w / total for w in weights]
            import numpy as np

            idx = np.random.choice(
                len(self.offline_sequences), size=batch_size, replace=True, p=probs
            )
            return [self.offline_sequences[i] for i in idx]
        else:
            return random.choices(self.offline_sequences, k=batch_size)

    def _load_kmer_rewards(self, path: str):
        """Load a precomputed k-mer reward dictionary.
        
        Args:
            path: Input JSON path.
        
        Returns:
            A mapping from k-mers to scores, or None on failure.
        """
        if not path:
            print("Warning: no k-mer reward file path was specified.")
            return None

        if not os.path.exists(path):
            print(f"Warning: k-mer reward file not found: {path}")
            return None

        print(f"Loading k-mer reward data: {path}")

        try:
            with open(path, "r") as f:
                rewards = json.load(f)

            if not rewards:
                print(f"Warning: the k-mer reward file is empty: {path}")
                return None

            print(f"Loaded k-mer reward data ({len(rewards)} distinct k-mers)")
            return rewards

        except json.JSONDecodeError:
            print(f"Error: the k-mer reward file is not valid JSON: {path}")
            return None
        except Exception as e:
            print(f"Error: failed to load the k-mer reward file: {e}")
            return None

    def score(self, candidates):
        """Compute objective-specific scores for candidate sequences.
        
        Args:
            candidates: Sequence strings in computational A/C/G/T notation.
        
        Returns:
            A NumPy array with shape [n_candidates, n_objectives].
        """
        scores = self.compute_scores(candidates, objectives=self.objectives)
        # Optional quantile-based normalization.
        if self.auto_norm.get("enabled", False):
            mode = self.auto_norm.get("mode", "quantile")
            if mode == "quantile":
                ql = float(self.auto_norm.get("q_low", 0.1))
                qh = float(self.auto_norm.get("q_high", 0.9))
                # Estimate quantiles for each objective.
                qs = []
                for j in range(scores.shape[1]):
                    lo = np.quantile(scores[:, j], ql)
                    hi = np.quantile(scores[:, j], qh)
                    if hi <= lo:
                        lo, hi = 0.0, max(1.0, hi)
                    qs.append((lo, hi))
                # Apply linear normalization.
                for j, (lo, hi) in enumerate(qs):
                    scores[:, j] = np.clip((scores[:, j] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        else:
            # Use the configured objective-specific normalization.
            scores = scores / np.array(self.score_max)
        return scores

    def compute_scores(self, sequences, objectives="length"):
        """Compute the requested objective scores.
        
        Args:
            sequences: Candidate sequence strings.
            objectives: A single objective name or a list of names.
        
        Returns:
            A NumPy array of objective scores for each sequence.
        """
        if isinstance(objectives, str):
            objectives = [objectives]

        all_scores = []
        for obj in objectives:
            if obj == "length":
                scores = self._compute_length_score(sequences)
            elif obj == "gc_content":
                scores = self._compute_gc_content_score(sequences)
            elif obj == "base_balance":
                scores = self._compute_base_balance_score(sequences)
            elif obj == "dinucleotide_diversity":
                scores = self._compute_dinucleotide_diversity_score(sequences)
            elif obj == "complexity":
                scores = self._compute_complexity_score(sequences)
            elif obj.startswith("kmer_"):
                scores = self._compute_kmer_reward_score(sequences, obj)
            elif obj == "rnafold_mfe":
                scores = self._compute_rnafold_mfe_score(sequences)
            else:
                raise ValueError(f"Unknown objective: {obj}")
            all_scores.append(scores)

        final_scores = np.vstack(all_scores).T

        # Normalize scores to the configured range.
        # Raw MFE is negative and uses its own score transformation.
        # final_scores /= np.array(self.score_max)

        return final_scores

    def set_training_mode(self, is_training: bool):
        """Switch between training and evaluation modes."""
        self._training_mode = bool(is_training)

    def _compute_length_score(self, sequences):
        """Score proximity to the configured ideal sequence length."""
        length_scores = []
        for seq in sequences:
            length_diff = abs(len(seq) - self.ideal_length)
            # Assign higher values to lengths closer to the target.
            score = 1.0 / (1.0 + length_diff / self.ideal_length)
            length_scores.append(score)
        return np.array(length_scores)

    def _compute_gc_content_score(self, sequences):
        """Score proximity to the configured ideal GC fraction."""
        gc_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                gc_scores.append(0.0)
                continue

            seq_upper = seq.upper()
            gc_count = seq_upper.count("G") + seq_upper.count("C")
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0.0

            # Assign higher values to GC fractions closer to the target.
            gc_diff = abs(gc_content - self.ideal_gc_content)
            score = 1.0 / (1.0 + gc_diff * 4)  # Scale the GC-deviation penalty by four.
            gc_scores.append(score)
        return np.array(gc_scores)

    def _compute_base_balance_score(self, sequences):
        """Score the uniformity of nucleotide composition."""
        balance_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                balance_scores.append(0.0)
                continue

            seq_upper = seq.upper()
            if len(seq_upper) == 0:
                balance_scores.append(0.0)
                continue

            # Calculate the frequency of each nucleotide.
            base_counts = {
                "A": seq_upper.count("A"),
                "T": seq_upper.count("T"),
                "G": seq_upper.count("G"),
                "C": seq_upper.count("C"),
            }

            # The uniform-composition target assigns one quarter to each nucleotide.
            ideal_count = len(seq_upper) / 4
            variance = sum((count - ideal_count) ** 2 for count in base_counts.values())

            # Lower composition variance yields a higher score.
            score = 1.0 / (1.0 + variance / (ideal_count**2))
            balance_scores.append(score)
        return np.array(balance_scores)

    def _compute_dinucleotide_diversity_score(self, sequences):
        """Score dinucleotide diversity."""
        diversity_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq) or len(seq) < 2:
                diversity_scores.append(0.0)
                continue

            seq_upper = seq.upper()

            # Count distinct dinucleotides.
            dinucleotides = set()
            for i in range(len(seq_upper) - 1):
                dinucleotides.add(seq_upper[i : i + 2])

            # There are at most sixteen possible dinucleotides.
            diversity_ratio = len(dinucleotides) / 16.0
            diversity_scores.append(diversity_ratio)
        return np.array(diversity_scores)

    def _compute_complexity_score(self, sequences):
        """Score sequence complexity by penalizing repeated patterns."""
        complexity_scores = []
        for seq in sequences:
            if not self._validate_sequence(seq):
                complexity_scores.append(0.0)
                continue

            seq_upper = seq.upper()
            if len(seq_upper) < 4:
                complexity_scores.append(0.5)  # Use an intermediate score for very short sequences.
                continue

            # Count repeated four-nucleotide patterns.
            tetranucleotides = {}
            for i in range(len(seq_upper) - 3):
                tetra = seq_upper[i : i + 4]
                tetranucleotides[tetra] = tetranucleotides.get(tetra, 0) + 1

            # Calculate the frequency of the most common four-nucleotide pattern.
            max_repeat = max(tetranucleotides.values()) if tetranucleotides else 1
            total_tetras = len(seq_upper) - 3

            # Assign a higher score when repetition is less frequent.
            repeat_ratio = max_repeat / total_tetras if total_tetras > 0 else 1.0
            complexity_score = 1.0 - repeat_ratio
            complexity_scores.append(max(0.0, complexity_score))
        return np.array(complexity_scores)

    def _calculate_arithmetic_mean(self, sequence: str, reward_data: dict, k: int) -> float:
        """Compute the arithmetic mean of k-mer scores."""
        if len(sequence) < k:
            return 0.0
        n_kmers = len(sequence) - k + 1
        total_score = 0.0
        for i in range(n_kmers):
            kmer = sequence[i : i + k]
            score = reward_data.get(kmer, self.min_kmer_reward)
            total_score += float(score)
        return total_score / float(n_kmers)

    def _calculate_geometric_mean(self, sequence: str, reward_data: dict, k: int) -> float:
        """Compute the geometric mean of k-mer scores with numerical stabilization."""
        if len(sequence) < k:
            return 0.0
        epsilon = 1.0e-9
        n_kmers = len(sequence) - k + 1
        log_sum = 0.0
        for i in range(n_kmers):
            kmer = sequence[i : i + k]
            score = reward_data.get(kmer, self.min_kmer_reward)
            log_sum += float(np.log(float(score) + epsilon))
        return float(np.exp(log_sum / float(n_kmers)))

    def _calculate_harmonic_mean(self, sequence: str, reward_data: dict, k: int) -> float:
        """Compute the harmonic mean of k-mer scores with numerical stabilization."""
        if len(sequence) < k:
            return 0.0
        epsilon = 1.0e-9
        n_kmers = len(sequence) - k + 1
        inv_sum = 0.0
        for i in range(n_kmers):
            kmer = sequence[i : i + k]
            score = reward_data.get(kmer, self.min_kmer_reward)
            inv_sum += 1.0 / (float(score) + epsilon)
        if inv_sum <= 0.0:
            return 0.0
        return float(n_kmers) / inv_sum

    def _calculate_minimum_score(self, sequence: str, reward_data: dict, k: int) -> float:
        """Return the minimum k-mer score."""
        if len(sequence) < k:
            return 0.0
        n_kmers = len(sequence) - k + 1
        min_score = None
        for i in range(n_kmers):
            kmer = sequence[i : i + k]
            score = float(reward_data.get(kmer, self.min_kmer_reward))
            if (min_score is None) or (score < min_score):
                min_score = score
        return 0.0 if min_score is None else float(min_score)

    def _compute_kmer_reward_score(self, sequences, obj_name):
        """Score sequences using a k-mer reward dictionary."""
        if obj_name not in self.kmer_rewards or not self.kmer_objectives.get(obj_name):
            return np.zeros(len(sequences))

        reward_data = self.kmer_rewards[obj_name]
        k = self.kmer_objectives[obj_name].get("k")
        aggregation = self.kmer_objectives[obj_name].get("aggregation", "arithmetic").lower()

        if k is None:
            return np.zeros(len(sequences))

        # Dispatch to the requested aggregation method.
        aggregation_fn = None
        if aggregation == "arithmetic":
            aggregation_fn = self._calculate_arithmetic_mean
        elif aggregation == "geometric":
            aggregation_fn = self._calculate_geometric_mean
        elif aggregation == "harmonic":
            aggregation_fn = self._calculate_harmonic_mean
        elif aggregation in ("min", "minimum"):
            aggregation_fn = self._calculate_minimum_score
        else:
            # Fall back to arithmetic averaging for an unrecognized aggregation name.
            aggregation_fn = self._calculate_arithmetic_mean

        kmer_scores = np.array([aggregation_fn(seq, reward_data, k) for seq in sequences])
        return kmer_scores

    def _compute_rnafold_mfe_score(self, sequences):
        """Transform variable-region MFE into a score.
        
        Using ViennaRNA fold_compound, compute r = -MFE and return r / (r + C), where C is rnafold_norm_factor. For positive C, increasingly negative MFE approaches a score of one from below.
        """
        # Compute variable-region MFE with ViennaRNA in a thread pool.
        from concurrent.futures import ThreadPoolExecutor

        def score_one(seq: str) -> float:
            fc = RNA.fold_compound(seq)
            (structure, mfe) = fc.mfe()
            reward = -mfe
            if self.rnafold_norm_factor <= 0:
                return 0.0
            # Normalize as r / (r + C).
            return reward / (reward + self.rnafold_norm_factor)

        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(score_one, sequences))
        return np.array(scores)

    def _validate_sequence(self, sequence):
        """Check that the sequence contains only A, C, G, and T."""
        return all(base in "ATGC" for base in sequence.upper())

    def _generate_random_sequence(self, length):
        """Generate a random sequence of the requested length."""
        bases = ["A", "T", "G", "C"]
        return "".join(np.random.choice(bases) for _ in range(length))
