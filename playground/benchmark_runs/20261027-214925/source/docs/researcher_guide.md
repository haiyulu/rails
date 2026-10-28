# Researcher guide to understanding and extending RAILS

This guide is for researchers who know PyTorch and want to understand this repository, reproduce a baseline, and make incremental changes without accidentally changing the experimental protocol. It follows one ML-1M HSTU experiment from data to checkpoint, then explains the extension points for other backbones and scoring functions.

The descriptions below were checked against the working tree on **2026-10-27**. Executable code and the resolved configuration are the source of truth. Research plans describe intended changes, not necessarily implemented behavior. Commands are examples to run from the repository root; documenting a command does not mean it was executed while writing this guide.

## Contents

1. [Start here: the repository's mental model](#1-start-here-the-repositorys-mental-model)
2. [Repository map and reading order](#2-repository-map-and-reading-order)
3. [Environment and prepared data](#3-environment-and-prepared-data)
4. [How Gin defines an experiment](#4-how-gin-defines-an-experiment)
5. [How an HSTU model is assembled](#5-how-an-hstu-model-is-assembled)
6. [A batch through the original HSTU](#6-a-batch-through-the-original-hstu)
7. [Training targets, losses, and optimizer updates](#7-training-targets-losses-and-optimizer-updates)
8. [Implemented model variants and proposed changes](#8-implemented-model-variants-and-proposed-changes)
9. [Interfaces and prepared-memory contracts](#9-interfaces-and-prepared-memory-contracts)
10. [Retrieval, evaluation, and metric interpretation](#10-retrieval-evaluation-and-metric-interpretation)
11. [Running and resuming experiments](#11-running-and-resuming-experiments)
12. [Four recipes for incremental changes](#12-four-recipes-for-incremental-changes)
13. [Validation and fair comparisons](#13-validation-and-fair-comparisons)
14. [Troubleshooting](#14-troubleshooting)

## 1. Start here: the repository's mental model

An experiment is composed from independently selected pieces:

```text
Gin configuration
       │
       ▼
Dataset → DataLoader → batch features
                           │
                    item embedding lookup
                           │
                    input preprocessing
                           │
                  sequence encoder: HSTU/SASRec
                           │
                    output postprocessing
                           │
                     similarity scorer ← candidate item embeddings
                           │
                   recommendation loss
                           │
                   backward → optimizer

Evaluation: prepared user outputs + catalog → matching top-k method → metrics
```

The **encoder** determines how history is represented. The **scorer** determines how that representation interacts with a candidate item. The **retrieval method** must rank by that scorer; a dot-product index is not an exact replacement for MoL or LCAR scoring.

The original recommendation experiments in this checkout are retrieval models. Do not assume they implement the industrial target-aware HSTU ranker merely because both use HSTU blocks. Here, original HSTU produces history states and scores candidate lookup embeddings afterward. LCAR introduces virtual-candidate reads and a prepared-memory scoring interface.

## 2. Repository map and reading order

| Location | Role | Start with |
| --- | --- | --- |
| [configs/](../configs/) | Dataset/model-specific Gin bindings | [Original ML-1M HSTU config](../configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin) |
| [train.py](../train.py) | Assembly, training loop, distributed setup, evaluation, checkpoints | `train_fn`, then `mp_train_fn` and `main` |
| [data/](../data/) | Preprocessing, sequence datasets, catalog metadata, metrics | [reco_dataset.py](../data/reco_dataset.py), then [dataset.py](../data/dataset.py) |
| [trainer/data_loader.py](../trainer/data_loader.py) | Samplers, workers, batching | `create_data_loader`, `ReusableDataLoader` |
| [modeling/sequential/](../modeling/sequential/) | Encoders, embeddings, feature preprocessing, losses | [encoder_utils.py](../modeling/sequential/encoder_utils.py), then [hstu.py](../modeling/sequential/hstu.py) |
| [modeling/similarity_module.py](../modeling/similarity_module.py) | Native encoder/scorer bridge | `similarity_fn`, `encode_with_history`, `prefix_loss` |
| [modeling/similarity_utils.py](../modeling/similarity_utils.py) | Scorer factories and Gin registration | `get_similarity_function` |
| [rails/similarities/](../rails/similarities/) | DotProduct, MoL, LCAR | [SimilarityModule](../rails/similarities/module.py) |
| [indexing/](../indexing/) | Application-level catalog handling and retrieval dispatch | [utils_rails.py](../indexing/utils_rails.py), [candidate_index.py](../indexing/candidate_index.py) |
| [rails/indexing/](../rails/indexing/) | Top-k interface and retrieval implementations | [TopKModule](../rails/indexing/candidate_index.py) |
| [eval_from_checkpoint.py](../eval_from_checkpoint.py) | Reconstruct and evaluate a saved model | Its CLI flags and checkpoint-loading path |
| [eval_batch.py](../eval_batch.py) | Batch evaluation recipes, including retrieval timing | Inspect its GPU, dtype, and timing settings before use |
| [playground/](../playground/) | Research plans, registries, correctness tests, benchmark tools | [model_registry.py](../playground/model_registry.py) |
| `tmp/` | Raw/prepared datasets | `tmp/ml-1m/sasrec_format.csv`, `tmp/processed/ml-1m/movies.csv` |
| `ckpts/`, `exps/`, `logs/` | Saved checkpoints, TensorBoard events, and run logs | Inspect the paths printed by the training entry point |
| `playground/benchmark_runs/`, `playground/inference_flops_runs/` | Generated run artifacts and source snapshots | Inspect manifests before interpreting results |

The two indexing directories serve different layers. The top-level `CandidateIndex` manages candidate IDs, filtering, and presentation of results. Package-level modules implement score computation and top-k selection. The factory in `indexing/utils_rails.py` connects them.

Files under a run's `source/` or `resume_source/` directory are archived copies, not the active implementation. Search live source first; otherwise, a recursive search may return an old implementation with the same filename.

A useful reading sequence is **config → training assembly → encoder factory → HSTU → loss → evaluation → retrieval → LCAR and its tests**. Follow the existing call graph before editing a low-level block.

## 3. Environment and prepared data

### Environment provenance

The historical [requirements.txt](../requirements.txt) pins PyTorch 2.2.2 and FBGEMM-GPU 0.6.0. The inspected project virtual environment has a different, newer dependency set:

| Package | Inspected local version |
| --- | --- |
| PyTorch | `2.9.0+cu128` |
| FBGEMM-GPU | `1.4.0+cu128` |
| Gin | `0.5.0` |
| NumPy | `2.0.1` |
| pandas | `2.2.2` |
| absl-py | `2.1.0` |

This is a record of the local environment, not a universal installation recipe. HSTU uses FBGEMM operators, and training uses CUDA/NCCL. A successful `import torch` alone does not establish operator compatibility. For an existing reproduction, use its environment lock and validation log; for a different machine, validate the chosen PyTorch/FBGEMM/CUDA combination before a long run. The [benchmark instructions](../playground/BENCHMARK.md) and [reproduction plan](../playground/rails_benchmark_reproduction_plan.md) provide additional context, but some operational descriptions predate newer runner features.

**Inspection only; no training:**

```bash
.venv/bin/python - <<'PY'
import importlib.metadata as metadata
for name in ("torch", "fbgemm-gpu", "gin-config", "numpy", "pandas"):
    print(name, metadata.version(name))
PY
```

Invoke `.venv/bin/python` by its virtual-environment path. Resolving that symlink to its underlying system interpreter before launching a child process can lose the virtual environment's package discovery.

### From ratings to examples

[preprocess_public_data.py](../preprocess_public_data.py) invokes the preprocessors in [data/preprocessor.py](../data/preprocessor.py) for ML-1M, ML-20M, and Books. Prepared data may already be present. That script is a data-producing workflow and may involve large downloads; it is not a lightweight environment check. To prepare only one dataset, call its corresponding preprocessor explicitly rather than assuming the entry point selects ML-1M alone.

[get_reco_dataset](../data/reco_dataset.py) builds `DatasetV2` instances and the candidate catalog. For the supplied ML-1M setup:

- Training uses `ignore_last_n=1`, excluding the user's latest interaction from the training sequence.
- Evaluation uses `ignore_last_n=0`, making that latest interaction the evaluation target.
- Each dataset example separates its latest remaining interaction as `target_ids` and returns earlier events as history.
- History is truncated to the configured maximum and returned in chronological order. Padding ID is **0**; valid item IDs are positive.

For a user with chronological interactions `[A,B,C,D,E]`, a sufficiently large history limit gives:

```text
Training dataset row:    history [A,B,C],   provided target D
Evaluation dataset row:  history [A,B,C,D], provided target E
```

The later training loop decides whether to supervise every next-item position or sample one prefix. A dataset row's provided target and the loop's sampled target are therefore not always the same item.

Dataset rows also carry aligned ratings, timestamps, history lengths, and a user ID. [movielens_seq_features_from_row](../modeling/sequential/features.py) moves tensors to the device and creates `SequentialFeatures`. With history limit 200 and the default 11 reserved output slots, the padded model input width is 211. A payload containing timestamps or ratings does not imply every model consumes it: the default embedding preprocessor uses item IDs and positions; HSTU can consume timestamps through relative attention bias; the configured MoL query tower can consume user IDs.

## 4. How Gin defines an experiment

Gin supplies arguments to functions decorated with `@gin.configurable`. Python classes define the computation; the configuration selects their dimensions and settings.

The [original ML-1M HSTU config](../configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin) specifies:

```gin
train_fn.main_module = "HSTU"
train_fn.item_embedding_dim = 50
train_fn.max_sequence_length = 200
train_fn.local_batch_size = 128
train_fn.num_epochs = 101

hstu_encoder.num_blocks = 8
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 25
hstu_encoder.dv = 25

train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"
train_fn.num_negatives = 128
train_fn.temperature = 0.05
```

Omitted parameters use Python defaults. For example, `train_fn.supervision_mode` defaults to `all_positions`; it need not appear in the baseline Gin file. `hstu_encoder` has its own defaults, some overridden here. Inspect both the configuration and the function signature.

Use an included config for an incremental experiment, then override only the intended settings. A complete example appears in [Recipe A](#recipe-a-configuration-only-experiment). The CLI accepts `--gin_config_file`; it does **not** expose an arbitrary `--gin_bindings` flag. Put overrides in a derived Gin file.

Bindings become known when their defining modules are imported. Importing `train` registers training and factory configurables for an inspection script:

```bash
.venv/bin/python - <<'PY'
import gin
import train
gin.parse_config_file("configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin")
for selector in ("train_fn.main_module", "train_fn.item_embedding_dim",
                 "hstu_encoder.num_blocks", "train_fn.interaction_module_type"):
    print(selector, gin.query_parameter(selector))
PY
```

This parses configuration without building a model or running training. `gin.config_str()` describes registered bindings; `gin.operative_config_str()` records configurable values actually used. Benchmark training writes an operative config after construction. Do not treat the operative snapshot as a replacement for the original included configs or for a source/environment snapshot.

## 5. How an HSTU model is assembled

The launch path is:

```text
train.py: main
  → one mp_train_fn process per visible GPU
  → gin.parse_config_file
  → train_fn
      → get_reco_dataset + create_data_loader
      → LocalEmbeddingModule
      → get_similarity_function
      → input/output preprocessor construction
      → get_sequential_encoder → hstu_encoder → HSTU
      → SampledSoftmaxLoss + negative sampler
      → move to GPU → DistributedDataParallel → AdamW
```

[encoder_utils.py](../modeling/sequential/encoder_utils.py) is the construction boundary. `get_sequential_encoder` selects SASRec or HSTU. `hstu_encoder` translates Gin parameters such as `dqk` and `dv` into constructor arguments.

[hstu.py](../modeling/sequential/hstu.py) contains three useful levels:

| Symbol | Responsibility |
| --- | --- |
| `SequentialTransductionUnitJagged` | One native HSTU block: normalization, U/V/Q/K projection, attention, gated output, residual |
| `HSTUJagged` | Execute the block stack and handle dense/jagged conversion and native cache plumbing |
| `HSTU` | Own embeddings, input/output processing, the stack, and the public encoding interface |

The superclass [SequentialEncoderWithLearnedSimilarityModule](../modeling/similarity_module.py) coordinates scorer access and prepared-history training. The scorer is held under `_ndp_module`; that name is historical and does not mean the scorer must be a dot product. Factories live in [modeling/similarity_utils.py](../modeling/similarity_utils.py); implementations live under `rails/similarities/`.

This separation is why changing a scoring function usually belongs in the scorer and its integration points, rather than inside every HSTU block.

## 6. A batch through the original HSTU

Let B=128 users, padded input width T=211, and hidden width d=50. Valid history length L differs by user and is at most 200.

| Stage | Tensor | Shape |
| --- | --- | --- |
| Features | `past_ids`, `past_lengths` | `[128,211]`, `[128]` |
| Shared item lookup | Raw history embeddings | `[128,211,50]` |
| Input preprocessing | Position-enriched, masked events X | `[128,211,50]` |
| Native HSTU storage | Valid event states in jagged form | `[sum(lengths),50]` |
| Block stack and postprocessing | History output H | `[128,211,50]` |
| Last-valid gather for retrieval | User representation u | `[128,50]` |
| Candidate lookup/normalization | N shared candidates | `[1,N,50]` |
| Dot-product scorer | Scores | `[128,N]` |

The shared table in [embedding_modules.py](../modeling/sequential/embedding_modules.py) has one row per item ID plus the padding slot. Both history items and candidates use it. The original candidate representation does not pass through the HSTU stack.

[Input preprocessing](../modeling/sequential/input_features_preprocessors.py) computes, for history position t:

```text
x_t = Dropout(sqrt(50) * item_embedding[item_id_t] + position_embedding[t])
```

It then masks padding. Candidate embeddings do not receive history-position embeddings. The baseline normalizes candidates through the negative sampler/catalog path; it postprocesses user states using L2 normalization.

The native block computes SiLU-activated U/V/Q/K projections from normalized inputs, then pointwise attention with relative position/time bias and a causal mask, followed by normalized, gated value aggregation and a residual update. It is not ordinary softmax Transformer attention. Two heads each use 25-dimensional Q/K and V in this configuration. The native pointwise attention denominator is the padded reference capacity, 211 here, not each user's valid length.

Sequence outputs and retrieval outputs are different interfaces:

- `HSTU.forward(...)` normally returns all postprocessed history states `[B,T,d]`, used for all-position training.
- `HSTU.encode(...)` normally gathers the last valid state `[B,d]`, used for original retrieval.
- `generate_user_embeddings(...)` normally returns sequence states plus cache states; optional history/cache flags change its tuple structure.
- `encode_with_history(...)` returns `(current_user, user_memory)` for prepared-memory scorers.

Use lengths to identify the last valid event. Taking `states[:, -1]` selects a padded slot in these batches. The HSTU docstrings and some variable names are inherited from older layouts; the chronological data construction and actual gather logic determine current behavior.

## 7. Training targets, losses, and optimizer updates

### All-position supervision: original baselines

The loop appends the dataset-provided target into the history ID buffer and shifts supervision by one position. For a row with history `[A,B,C]` and provided target D, the conceptual examples are:

```text
[A]     → B
[A,B]   → C
[A,B,C] → D
```

These predictions share one causal encoder pass. A later target being present in the padded ID buffer does not grant an earlier query access to it. Native lengths and attention masking are part of this behavior; preserve them when changing the encoder.

The loss receives `seq_embeddings[:, :-1]`, `supervision_ids[:, 1:]`, and a nonpadding supervision mask. [SampledSoftmaxLoss.forward](../modeling/sequential/losses/sampled_softmax.py) uses lengths to gather valid prediction positions into a jagged batch.

### Prefix supervision: full history by default, sampled cuts for studies

[sample_causal_prefix](../modeling/sequential/prefix_sampling.py) constructs one causal example per user. In the same row, it can choose `[A] → B`, `[A,B] → C`, or `[A,B,C] → D`. It returns only the selected prefix as input and masks later sequence payloads. Cuts are seeded from the configured seed, rank, and optimizer-step index.

LCAR needs its selected originals and summaries to describe the prefix available for that target. Reusing a full-history memory to supervise an earlier item would leak future events. LCAR requires one causal prefix per target; its current configs use `full_prefix`, while `sampled_prefix` remains an explicit study option. The full-prefix helper masks reserved future IDs and timestamps without mutating the supplied batch.

**Full-available-history next-item training is executable as `supervision_mode="full_prefix"`.** It always uses `[A,B,C] → D` in this example. The training entry point accepts `all_positions`, `sampled_prefix`, and `full_prefix`.

### Scores and loss

[LocalNegativesSampler](../modeling/sequential/autoregressive_losses.py) samples from catalog item IDs and retrieves embeddings from the shared lookup table. Its normalization setting also determines normalization of positive candidates. In-batch sampling is a separate selectable strategy; it should not silently replace local sampling in a baseline comparison.

For each supervised target, sampled softmax constructs one positive and 128 sampled negatives in the ML-1M configurations. Negatives equal to the positive ID are masked rather than treated as valid competing labels:

```text
logits = [positive_score, negative_score_1, …, negative_score_128] / temperature
loss   = -log_softmax(logits)[positive_column]
```

The original dot-product config uses loss temperature 0.05. The MoL config uses internal scorer temperature 0.05 and loss temperature 1. LCAR uses loss temperature 1. Changing scorer and loss temperatures together without tracing both locations can accidentally scale scores twice.

Scorers return `(scores, auxiliary_losses)`. `get_weighted_loss` adds only configured auxiliary terms to the recommendation loss. Every configured key must exist in that loss dictionary. MoL configures user-ID embedding regularization and mutual-information load balancing; LCAR configures sparse-code reconstruction/L1 loss. Diagnostics are not automatically objectives.

### One optimizer step

```text
Set model.train()
Choose supervision → zero gradients → form embeddings
Run DDP forward → recommendation loss + weighted auxiliary terms
Backward → optional warmup learning-rate update
AdamW.step()
Optional scorer.post_optimizer_step() (e.g. dictionary norm projection)
```

AdamW uses betas `(0.9,0.98)`; the original ML-1M HSTU config sets learning rate and weight decay to `1e-3`. Data samplers receive `set_epoch(epoch)`. One process is spawned per visible GPU and the backend is NCCL; restrict `CUDA_VISIBLE_DEVICES` when a single-GPU experiment is intended. `local_batch_size` is per rank.

Sampled-prefix recommendation preparation runs inside the model's distributed forward. Moving it outside DDP, detaching prepared states, or registering the same backbone under a second owner can break gradients or distributed parameter accounting even if inference still produces scores.

## 8. Implemented model variants and proposed changes

| Variant / policy | Current status and behavior |
| --- | --- |
| Original SASRec/HSTU + DotProduct | Implemented. Sequence encoder plus a single candidate lookup vector; supplied baseline configs use all-position training. |
| SASRec/HSTU + MoL | Implemented. Multiple user/item component embeddings, component-pair dot products, learned pair-dependent mixing, auxiliary losses. |
| Superseded ExpertHistory | Removed from active source/configuration paths; historical experiment snapshots are untouched. |
| Historical LCAR benchmarks | Version 3: two experts, 512 atoms each, three groups, sampled-prefix training. Preserved artifacts are not version-4 results. |
| Current LCAR configs | Implemented version 4: four experts, 128 atoms each, 512 total latents, global-history token and four source types; full-prefix training. |
| Full-available-history next-item supervision | Implemented `full_prefix`; current LCAR default. Baseline configs retain all-position training. |

For the supplied ML-1M HSTU+MoL config, a 50-dimensional item lookup is projected to four normalized 64-dimensional item components. Eight user components give 32 pairwise logits. Importantly, seven user components come from the transformed HSTU representation and one comes from a user-ID embedding. The item component projection is a single linear layer in this config (`item_hidden_dim=-1`), not a hidden-layer SwiGLU network. Its user-output normalization also differs from the dot-product config. Inspect the [MoL config](../configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin), [query tower](../rails/similarities/mol/query_embeddings_fns.py), [item tower](../rails/similarities/mol/item_embeddings_fns.py), and [mixing function](../rails/similarities/mol/similarity_fn.py) before attributing every difference to the mixture formula alone.

Current LCAR is an independent [package](../rails/similarities/lcar/README.md). Parameter-free adapters in [adapters/](../rails/similarities/lcar/adapters/) reuse native SASRec/HSTU weights. The generic [encoder runner](../rails/similarities/lcar/encoding.py) computes history once per layer and lets independent latent tokens read it. History cannot read latents; latents cannot read one another. First-block coefficients select original events; final latent states produce summaries. Actual item queries activate dictionary entries and read one concatenated memory of recent originals, a projected global-history token, selected originals, and activated summaries.

The [modeling plan](../playground/model_idea/retrieval_model_improvement_plan.md) and [scoring plan](../playground/new_similarity_scoring_plan.md) describe the implemented four-source design. The independent [LCAREncoder](../rails/similarities/lcar/encoder.py) wraps the host and scorer; native blocks have no LCAR branches. Keep historical results separately labeled.

## 9. Interfaces and prepared-memory contracts

### Ordinary scorer contract

[SimilarityModule.forward](../rails/similarities/module.py) accepts:

```python
scores, auxiliary_losses = scorer(
    query_embeddings=user_vectors,     # [B,d_q]
    item_embeddings=item_vectors,      # [1,N,d_i] or [B,N,d_i]
    **payloads,
)
# scores: [B,N]; auxiliary_losses: dict[str, scalar Tensor]
```

Shared catalog candidates use leading dimension 1. Training negatives may be user-aligned. Preserve both cases. Auxiliary terms must remain differentiable if they are configured as objectives. Keep raw logits separate from probabilities and loss-temperature scaling.

### Prepared-memory contract

Current LCAR exposes these entry points:

```python
memory = scorer.prepare_user(backbone, lengths, history_ids, history_embeddings, payloads)
items = scorer.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = scorer.score_prepared(memory, items)
```

The integrated model instead exposes `current_user, memory = model.encode_with_history(...)`. Its current-user vector is also retained in version-4 `UserMemory` with the projected global token, its mask and K/V.

The immutable dataclasses in [states.py](../rails/similarities/lcar/states.py) define the actual contract:

| State | Current purpose |
| --- | --- |
| `ModelContext` | Shared projected initial latent bases, four projected source offsets and version |
| `HistoryCache` | Original inputs, masks, timestamp provenance, per-layer history read contexts, final history states |
| `EncoderOutput` | Original/history/latent states, first-block selection positions/masks, context and history cache |
| `UserMemory` | Packed originals and K/V, recent/selected references, all summaries and K/V, global history state/token/K/V, masks, shared model context, payloads and provenance |
| `ItemState` | IDs, expert queries, routing support, coefficients, final expert gates, sparse-loss statistics and provenance |
| `JointMemory` | Gathered candidate-specific keys, values, validity masks, source IDs, and selected-position diagnostics |

For current ML-1M defaults, all-latent selections are `[B,4,128,16]`, summaries are `[B,4,128,32]`, item queries are `[1/B,N,4,32]`, and scores are `[B,N]`. The global token is `[B,1,32]`; pair memories have at most 297 valid tokens for ML-1M.

User memory must support `index_batch`; item state must support candidate slicing and shared/per-user batching. Caches are evaluation accelerators, not permission to detach training tensors. Cache validity includes parameter versions, encoder provenance, mode, dtype/device, masks, and relevant timestamps. Register canonical trainable parameters once; the LCAR adapter must not register a second copy/owner of the host backbone.

Construct and move trainable LCAR modules outside `torch.inference_mode()`. Its cache version checks require normal parameter tensors with version counters. Enter evaluation/no-gradient contexts for preparation and scoring afterward; training preparation must retain autograd.

The prepared branch of `SampledSoftmaxLoss.prefix_forward` currently expects dictionary-related item diagnostics and a scorer `dictionary_size`. It is not yet an arbitrary, dictionary-free prepared-scorer protocol. A new prepared scorer must either satisfy that contract or deliberately generalize this branch and its tests.

### Top-k contracts

Low-level [TopKModule.forward](../rails/indexing/candidate_index.py) returns **`(top_k_scores, top_k_ids)`**, both `[B,k]`. The application [CandidateIndex.get_top_k_outputs](../indexing/candidate_index.py) returns **`(top_k_ids, top_k_scores, optional_embeddings)`** after filtering. Their tuple order differs. Do not interchange them.

## 10. Retrieval, evaluation, and metric interpretation

[get_eval_state](../data/eval.py) prepares a catalog from the model's item embeddings using the candidate-normalization policy. Evaluation computes user representations once, or calls `encode_with_history` for prepared-memory scorers, then ranks the catalog and filters historical IDs.

| Scorer | Matching exact method |
| --- | --- |
| DotProduct | `MIPSBruteForceTopK` |
| MoL | `MoLBruteForceTopK` |
| LCAR | `LearnedBruteForceTopK` |

[get_exact_top_k_method](../indexing/utils_rails.py) and `get_top_k_module` perform dispatch. [LearnedBruteForceTopK](../rails/indexing/learned_top_k.py) prepares item states and scores candidate chunks, initially 256 candidates per chunk. It forwards sliced user memory and rejects stale catalog/model state.

MoL also has approximate methods in [mol_top_k.py](../rails/indexing/mol_top_k.py): component-based candidate selection, averaged-component selection, and combinations. Validate their retrieved candidates against the matching exact scorer. Approximate retrieval agreement is a different quantity from recommendation HR against held-out labels.

Metrics include HR, NDCG, and MRR. Accuracy evaluation uses up to 2,500 retrieved ranks; the MRR implementation assigns a non-retrieved target rank 2,501, so it is not an unrestricted full-catalog MRR computation. Timing mode changes retrieval cutoffs and truncation, as well as adding repeated timing work. Use `--include_eval_time=false` for strict recommendation-accuracy comparisons. See [data/eval.py](../data/eval.py).

**GPU evaluation command; replace the checkpoint path. This reconstructs and evaluates a model:**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_from_checkpoint.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin \
  --inference_from_ckpt=/absolute/path/to/checkpoint.pt \
  --top_k_method=MIPSBruteForceTopK \
  --eval_batch_size=128 \
  --limit_eval_to_first_n=0 \
  --include_eval_time=false \
  --eval_against_brute_force=false \
  --metrics_output=playground/hstu-evaluation.json \
  --master_port=12356
```

Use the checkpoint's matching architecture/config, not merely the same dataset. For a derived training config, remove unsupported training-only benchmark bindings from the evaluator config as the benchmark runner does, while retaining every model-defining override. `eval_batch.py` includes timing-oriented defaults and an explicit GPU selection; it is not a neutral substitute for the accuracy command above.

## 11. Running and resuming experiments

### First smoke step

Before a long run, create a **new** file such as `configs/ml-1m/hstu-smoke-local.gin` containing:

```gin
include 'configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin'
train_fn.custom_date_str = "hstu-smoke-local"
train_fn.local_batch_size = 4
train_fn.smoke_steps = 1
create_data_loader.num_workers = 2
create_data_loader.prefetch_factor = 4
```

**GPU optimizer workload; not a parse-only check:**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --gin_config_file=configs/ml-1m/hstu-smoke-local.gin \
  --master_port=12355
```

Smoke mode performs the configured number of optimizer steps, checks finite loss and gradients, and prints `SMOKE_OK`. It skips normal evaluation and returns before the usual training completion path; it still creates logs. The smaller batch above is a smoke-only setting, not the reported baseline batch size.

### Normal training and outputs

**Long GPU training workload:**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin \
  --master_port=12355
```

Normal training writes TensorBoard events under `exps/` and checkpoints under `ckpts/`. Intermediate checkpoint frequency is controlled by `save_ckpt_every_n`; do not assume a checkpoint appears every epoch under the baseline defaults. There is also a final checkpoint. Inspect its stored epoch semantics before using it for continuation: intermediate/benchmark checkpoints store the completed zero-based epoch, whereas the ordinary final save occurs after the epoch counter has advanced. `--restore_from_ckpt` resumes at the stored epoch plus one.

With `train_fn.benchmark_output_dir` enabled, training instead retains `best.pt`, `last.pt`, `epochs.jsonl`, and `operative_config.gin` in that directory. Best means highest observed evaluation HR@10; strict improvement retains the earlier checkpoint on a tie. Epoch records include steps, supervision mode, target exposures, metrics, parameter count, cumulative training/evaluation time, and peak allocated CUDA memory. Use a fresh directory for a new experiment; direct training does not provide the runner's new-directory protection.

### Benchmark orchestration

[run_ml1m_benchmark.py](../playground/run_ml1m_benchmark.py) performs preflight checks, one-step smoke runs, full training, and independent best/final checkpoint evaluation. It is separate from the model implementation. Select models explicitly rather than relying on defaults that may expand as models are added.

**Long, single-GPU benchmark workload; choose a fresh run directory:**

```bash
.venv/bin/python playground/run_ml1m_benchmark.py \
  --models hstu sasrec \
  --run-dir playground/benchmark_runs/my-baseline-study
```

The runner fixes GPU 0 and checks for 101 epochs. Benchmark output handling requires a single GPU. Additional models must be registered in [model_registry.py](../playground/model_registry.py) before they become CLI choices.

Run artifacts include:

| Artifact | Why retain it |
| --- | --- |
| `manifest.json` | Commands, status, seed, model metadata, data hashes, GPU/environment information and deviations |
| `requirements.lock.txt` | Resolved environment package versions |
| `source.patch`, `source/`, `source_sha256.json` | Tracked diff plus eligible tracked/untracked source snapshots and hashes |
| Per-model Gin files and `operative_config.gin` | Requested and used experiment settings |
| `epochs.jsonl`, `best.pt`, `last.pt` | Per-epoch evidence and reusable checkpoints |
| `logs/` | Validation, training, smoke and checkpoint-verification output |
| `comparison.md`, `comparison.csv`, `lcar_vs_paper.csv` when applicable | Generated summaries, including partial-run labels |
| `resume_source/` and resumption manifest entries | Source used after an interruption and explicit resume limitations |

**Report-only command; writes reports but does not run training:**

```bash
.venv/bin/python playground/run_ml1m_benchmark.py \
  --report-only --run-dir playground/benchmark_runs/my-baseline-study
```

The current runner also refreshes reports while a child command is running, approximately every ten seconds. A manually terminated runner can leave a stale manifest status: confirm active processes and recent logs before calling a run live. Partial rows are labeled `latest_incomplete`; they are not completed-model results.

**Resume command; may launch substantial GPU training and evaluation:**

```bash
.venv/bin/python playground/run_ml1m_benchmark.py \
  --resume --run-dir playground/benchmark_runs/my-baseline-study
```

Resume validates data and model-source hashes, takes a resume lock, uses contiguous epoch records and each model's `last.pt`, continues incomplete training, verifies checkpoints, and proceeds to queued models. It preserves original logs and records resume sources. Inspect the checks before attempting to resume after model-code edits; a fresh experiment is usually the appropriate way to evaluate a different architecture.

Current checkpoints retain model/optimizer state, counters, and relevant training metadata, but not the full Python/CPU/CUDA RNG state. Sampled cuts have step-based seeding, but dropout and negative-sampling streams need not match an uninterrupted run after restart. Report that limitation instead of claiming bitwise continuation. Cumulative benchmark time excludes the inactive gap and preserves the prior recorded peak memory on resume.

## 12. Four recipes for incremental changes

### Recipe A: configuration-only experiment

**Example question:** does reducing HSTU depth from eight blocks to four improve the quality/cost tradeoff?

1. Create a derived Gin file; leave the baseline config unchanged.
2. Override only depth and experiment/output naming. Keep the scorer, supervision, normalization, learning rate, negatives, and split fixed.
3. Parse the config and inspect the binding, run a smoke derivative, then schedule full training only when intended.
4. Evaluate the resulting checkpoint with the same architecture override and exact retrieval method.

Example new file `configs/ml-1m/hstu-b4-study.gin`:

```gin
include 'configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin'
hstu_encoder.num_blocks = 4
train_fn.custom_date_str = "hstu-b4-seed42"
train_fn.random_seed = 42
train_fn.benchmark_output_dir = "playground/benchmark_runs/hstu-b4-seed42"
create_data_loader.num_workers = 2
create_data_loader.prefetch_factor = 4
```

Run this through `train.py` using `--gin_config_file` as above. To use the multi-model benchmark runner instead, add a distinct model-registry entry; its fixed model choices do not accept an arbitrary config path. Confirm the global batch size and target exposure before comparing runs.

### Recipe B: new similarity function

**Example question:** does a different interaction between the existing user/item vectors improve ranking?

1. Implement a module under `rails/similarities/` using the `SimilarityModule.forward` contract. Support both shared candidates `[1,N,d]` and aligned candidates `[B,N,d]`; return scores `[B,N]` and an auxiliary-loss dictionary.
2. Add a Gin-configurable factory and a dispatch branch in `modeling/similarity_utils.py`. Keep the original backbone and embedding dimensions unless the experiment explicitly changes them.
3. Configure auxiliary-loss weights only for keys the scorer returns. Check whether temperatures and normalization belong inside the scorer or in the loss, and apply them once as intended.
4. Integrate an exact retrieval method through `indexing/utils_rails.py`. `LearnedBruteForceTopK` requires prepared-state methods; it is not a generic wrapper for any arbitrary `forward`. A new vector-only scorer may need a matching chunked top-k module.
5. Add distinct registry metadata, a derived config, checkpoint coverage, and tests before using a benchmark label.

Minimum checks: shared/aligned candidate equality against direct references, finite gradients through user/item inputs and scorer parameters, duplicate-positive behavior, exact top-k agreement with explicit full scores, strict reload, and matching training/checkpoint evaluation. Test cache invalidation if preparation is introduced. Compare against the original scorer under matched supervision, not only matched epochs.

### Recipe C: backbone modification

**Example question:** does a changed HSTU block transformation improve sequence modeling?

1. Decide whether the change belongs in one block (`SequentialTransductionUnitJagged`) or the enclosing encoder/factory. Keep experiment choices in configuration rather than hard-coded dataset conditionals.
2. Expose the choice through `hstu_encoder` and a derived Gin config. If introducing an entirely new encoder class, register it in `get_sequential_encoder` and provide the forward/encode contracts used by training and evaluation.
3. Preserve chronological order, causal visibility, padding/length handling, last-valid extraction, output shapes, and matching relative-bias/normalization semantics. Explicitly label experiments that intentionally change these.
4. Check both native history execution and the LCAR adapter if the new block is intended to support LCAR. The adapter reuses parameters but explicitly implements native equations; changing only `hstu.py` can leave it implementing the old block.
5. Version incompatible checkpoints. Do not use non-strict loading to hide changed projections or missing parameters in a reported comparison.

Minimum checks: future-event perturbations cannot change earlier outputs, padding is not evidence, dense/jagged reference agreement, native/adapter equivalence where supported, finite gradients, activation checkpointing, and saved-model evaluation. For a new LCAR host, implement/register `BackboneAdapter` instead of copying its dictionary/scorer into the backbone.

### Recipe D: prepared-memory extension

**Example question:** how can another user-memory source be added? The existing global-history token is a worked example.

1. Begin with the current `EncoderOutput`, `UserMemory`, `ItemState`, and `JointMemory` contracts. Define which computations are model-only, user-only, item-only, and per pair. Keep user preparation independent of the candidate.
2. Extend state dataclasses, slicing, masks, byte accounting, version checks, and both standalone/integrated preparation. For the implemented global token, gather the last valid already-postprocessed history state, project once per user, and mask empty histories. No second history encoder pass is needed.
3. Modify memory assembly and exact candidate-chunk scoring consistently. Preserve unpooled token information, original-event provenance, chronological union ordering, and declared recent/selected overlap.
4. Keep gradients attached during training. Register canonical parameters once; adapters remain parameter-free host views. Evaluation caches must reject stale data after relevant parameter, prefix, timestamp, mode, dtype, or device changes.
5. Extend the prefix-loss bridge deliberately, including sparse-loss accounting and dictionary-specific diagnostics. Keep preparation inside DDP forward, and verify activation-checkpoint recomputation is consistent.
6. Use a new checked state/checkpoint format when layout or computation changes. Increment the current version 4 if the state semantics become incompatible.

For the adopted global-token/source-embedding design, follow the linked plans: source offsets must enter K/V exactly once, including affine projection biases. Four expert queries and four source types are different axes. Test a direct concatenated-memory implementation against cached gathers, shared/aligned candidates, user and candidate slicing, empty histories, gradients through each new parameter, strict reload, and exact chunked retrieval.

## 13. Validation and fair comparisons

Start with small deterministic references, then one optimizer step, then a checkpoint round trip and exact evaluation. Broad accuracy/latency benchmarks come after these checks, not instead of them.

Useful existing test entry points:

| Command | Scope / workload |
| --- | --- |
| `.venv/bin/python -m unittest playground.test_benchmark_resume -v` | Synthetic runner-resume/report tests; no model training |
| `.venv/bin/python -m unittest discover -s playground -p 'test_inference_flops_counter.py' -v` | Arithmetic-accounting unit tests |
| `.venv/bin/python -m unittest playground.test_lcar playground.test_retrieval_integration -v` | Integration suites; include small optimizer/DDP/checkpoint tests and CUDA checks where available |
| `.venv/bin/python playground/validate_ml1m_benchmark.py` | GPU/data-dependent preflight, including jagged gradients and causality; not a long benchmark |

Read test prerequisites before running them on another host. The [LCAR tests](../playground/test_lcar.py), [baseline/integration tests](../playground/test_retrieval_integration.py), [resume tests](../playground/test_benchmark_resume.py), and [counter tests](../playground/test_inference_flops_counter.py) are useful executable references, but some use small fixture dimensions rather than production configurations.

For every comparison, record:

- Dataset hashes, split/truncation policy, chronological order, and seen-item filtering.
- Initial weights and random seeds, data order, sampled cuts, negative sampling, and the number of supervised targets and optimizer steps. Equal seeds alone do not guarantee identical stochastic streams after architecture changes.
- Global/per-rank batch sizes, negatives, normalization, score/loss temperatures, auxiliary objectives, dropout, and precision/TF32 settings.
- Architecture, scorer, retrieval algorithm, candidate/user chunking, environment, and checkpoint format.
- Final-epoch metrics separately from the best observed evaluation HR@10 checkpoint. The current benchmark has no independent validation split for that selection.
- Recommendation quality separately from approximate-retrieval agreement, latency, arithmetic accounting, and memory. A FLOP ledger is not a latency measurement.

All-position and one-prefix training have different target exposure. Switching scorer and supervision together cannot isolate the scorer's effect. Likewise, a single-seed improvement is not evidence of statistical significance. Compare paper Table 3 targets through the runner's explicit references rather than silently substituting the older root README table.

## 14. Troubleshooting

| Symptom | What to inspect first |
| --- | --- |
| Gin reports an unknown selector or binding | Import the module defining the configurable; check the exact factory name and signature; avoid unsupported CLI override flags. |
| A setting appears ignored | Inspect included configs, Python defaults, construction path, and operative config. A nonlinearity setting may be inactive when a factory takes a linear-only branch. |
| Scores/loss differ greatly after changing scorer | Trace input/output normalization, both temperature locations, auxiliary terms, and supervision mode before changing the learning rate. |
| Last user vector is zero or comes from padding | Gather index `length-1`, not the last padded column; verify valid IDs and native lengths. Explicitly handle empty histories in new code. |
| Future information affects predictions | Check causal masks, shifted labels, sequence payloads/timestamps, prefix restriction, and whether a full-history cache was reused for an earlier target. |
| Missing `fbgemm` operator or operator execution fails | Verify the virtual environment, FBGEMM import/registration, installed wheel compatibility, and the actual forward/backward operator check. Package imports alone are insufficient. |
| CPU execution or distributed startup fails | Training uses CUDA/NCCL and one process per visible GPU. Check visible devices, a free master port, and whether another experiment owns the resources. |
| A detached child cannot import installed dependencies | Preserve the `.venv/bin/python` path rather than resolving it to the base interpreter. |
| LCAR reports stale user/catalog state | Rebuild preparation after weight, prefix, timestamp, mode, dtype/device changes; do not bypass provenance checks. |
| New parameters receive no recommendation gradients | Check detached caches, preparation outside DDP, module registration, hard selection-only paths, and activation-checkpoint closures. |
| A checkpoint cannot load | Reconstruct the exact architecture and Gin settings; distinguish historical checkpoints from current LCAR version 4; no implicit migration is provided. For supplied checkpoints, follow the root README's Git LFS setup and verify the file is actual weights rather than an LFS pointer. Use explicit warm-start mappings only when intended. |
| Evaluation and training metrics disagree | Match the saved epoch, normalization, precision, seen-item filtering, exact retrieval method, and timing flags; use an independent checkpoint evaluation. |
| Report looks live but epochs do not advance | Inspect the process and latest log, not just the manifest's recorded status. Report-only regeneration does not resume training. |
| Resume refuses to proceed | Check data/source hashes, the resume lock, contiguous epoch records, and the correspondence between `last.pt` and the last JSONL row. Keep architectural changes in a new run. |
| Searching finds conflicting implementations | Exclude generated `benchmark_runs/**/source/` and `inference_flops_runs/**/source/`; read active source before research plans or historical snapshots. |

For the next change, first write down the experimental question, the smallest component that must change, its tensor contract, the baseline to preserve, and the check that would falsify an incorrect implementation. That makes later ablations easier to interpret and keeps implementation changes reviewable.
