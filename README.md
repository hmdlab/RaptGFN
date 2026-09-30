# RaptGFN

Preference-conditioned RNA aptamer candidate generation using a multi-objective GFlowNet.

This repository contains **training and generation only**. Training combines trajectory balance on policy-generated sequences with offline maximum-likelihood learning on observed sequences. The supplied task template contains the five objectives used in the manuscript:

- 3-mer frequency and enrichment;
- 5-mer frequency and enrichment;
- a variable-region minimum-free-energy (MFE) score.

The training reward is a preference-weighted combination. The generated CSV column named reward is the **unweighted sum** of objective-specific scores, Q(x), not a measured binding activity. Computational sequences use T in place of U.

## Installation

Use Python 3.10 or later. Install a PyTorch build compatible with your hardware when GPU training is needed; otherwise the implementation runs on CPU.

~~~bash
git clone https://github.com/hmdlab/RaptGFN.git
cd RaptGFN
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

The Python package and command are named raptgfn. ViennaRNA provides the required RNA module. W&B is disabled by default. Enabling online logging can transmit supplied settings, generated sequences, scores, and plots.

## Prepare the inputs separately

No datasets, biological reference sequences, primers, trained checkpoints, or reward dictionaries are bundled. No dataset names or dataset-to-accession mappings are included.

Empty directories are provided for local inputs and outputs:

| Directory | Contents to supply or generate locally |
|---|---|
| data/sequences/ | Preprocessed offline training CSV files |
| data/rewards/ | Frequency and enrichment score dictionaries in JSON format |
| outputs/ | Training checkpoints, resolved run configurations, and logs |
| results/ | Generated sequence CSV files |

Only empty .gitkeep markers are included. Actual files placed in these directories are ignored by Git. The directories do not configure input paths automatically; pass the paths explicitly in the commands below.

Supply:

- an offline CSV with a sequence column containing filtered, deduplicated variable-region sequences;
- frequency and enrichment dictionaries as flat JSON mappings from k-mers to rank-scaled scores;
- the minimum, maximum, and nominal variable-region lengths.

The default task is ht_selex_template. Required values are marked with ???. Inspect it without starting a run:

~~~bash
python -m raptgfn.commands.train --cfg job
~~~

The core algorithm configuration contains defaults, not a complete dataset-specific replication recipe. Set the architecture, batch size, training budget, and seed to the intended experiment.

## Train

Run from the repository root. The shell variables below must be set to your own approved inputs and experiment settings; they have no hidden defaults.

~~~bash
python -m raptgfn.commands.train \
  task=ht_selex_template \
  seed="$TRAIN_SEED" exp_name=training_run wandb_mode=disabled \
  task.offline_dataset_path="$OFFLINE_CSV" \
  task.min_len="$MIN_LENGTH" task.max_len="$MAX_LENGTH" \
  task.ideal_length="$NOMINAL_LENGTH" \
  task.kmer_objectives.kmer_freq_3.path="$FREQUENCY_JSON" \
  task.kmer_objectives.kmer_freq_5.path="$FREQUENCY_JSON" \
  task.kmer_objectives.kmer_enrich_3.path="$ENRICHMENT_JSON" \
  task.kmer_objectives.kmer_enrich_5.path="$ENRICHMENT_JSON" \
  algorithm.train_steps="$TRAIN_STEPS" \
  algorithm.batch_size="$BATCH_SIZE" algorithm.model.batch_size="$BATCH_SIZE" \
  algorithm.model.num_hid="$HIDDEN_SIZE" \
  algorithm.model.num_layers="$NUM_LAYERS" algorithm.model.num_head="$NUM_HEADS"
~~~

Use a new exp_name for each run. Preserve the resolved Hydra configuration with the checkpoints. Inspect the log and actual output files: the inherited training entry point can catch an exception without returning a failing exit code.

The best checkpoint is selected by the training-time metric logged as hypervolume, not by the lowest loss. The local higher-dimensional fallback is a sum of box volumes and does not subtract overlaps.

The offline loader may reuse an existing filtered CSV cache. Confirm that any cache belongs to the intended input; do not reuse a different experiment's cache.

Known inherited limitation: a local test with equal minimum and maximum lengths failed in reward/mask handling after no valid online samples were retained. This boundary behavior is not fixed by the documentation and naming changes. Use the intended admissible length interval and inspect the log and generated lengths; fixed-length training is not validated by this release.

## Generate

Use a trusted checkpoint and repeat the training task, objective order, sequence bounds, and model architecture. The checkpoint path alone does not restore the complete task configuration.

~~~bash
python -m raptgfn.commands.generate \
  task=ht_selex_template \
  checkpoint_path="$CHECKPOINT" num_samples="$NUM_SAMPLES" \
  output_file="$OUTPUT_CSV" seed="$GENERATION_SEED" wandb_mode=disabled \
  task.offline_dataset_path=null \
  task.min_len="$MIN_LENGTH" task.max_len="$MAX_LENGTH" \
  task.ideal_length="$NOMINAL_LENGTH" \
  task.kmer_objectives.kmer_freq_3.path="$FREQUENCY_JSON" \
  task.kmer_objectives.kmer_freq_5.path="$FREQUENCY_JSON" \
  task.kmer_objectives.kmer_enrich_3.path="$ENRICHMENT_JSON" \
  task.kmer_objectives.kmer_enrich_5.path="$ENRICHMENT_JSON" \
  algorithm.batch_size="$BATCH_SIZE" algorithm.model.batch_size="$BATCH_SIZE" \
  algorithm.model.num_hid="$HIDDEN_SIZE" \
  algorithm.model.num_layers="$NUM_LAYERS" algorithm.model.num_head="$NUM_HEADS" \
  hydra.run.dir=outputs/generation_run
~~~

Create the parent directory of OUTPUT_CSV first. Set GENERATION_SEED to an explicit integer or null; null chooses a seed for that run rather than reusing the training seed. Offline training sequences are not needed for generation when the offline path is null, but the reward dictionaries are still needed.

Only load trusted checkpoints: the loader may fall back to pickle-compatible deserialization. A fixed preference is not guaranteed by the top-level pref option alone in the inherited sampling path.

The output contains sequence, reward, and one column per objective score. Generated duplicates and exact training-set matches are not globally excluded. Check the retained row count rather than assuming every requested sample is valid.

## Runtime scope

Input loading, objective scoring, and internal monitoring remain necessary for training and best-checkpoint selection. Generation also calls the internal evaluation() method to obtain sequences and scores. These are not standalone analysis scripts.

This distribution includes no standalone evaluation suite, comparison plots, preprocessing/export utilities, or internal audit files. Prepared inputs and any downstream analyses are handled separately.

## Attribution and license

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream MIT notice and attribution. No manuscript DOI or archived-code DOI is asserted here.
