# LCAR: Latent Cross-Attention Retrieval

LCAR prepares user memory before the candidate is known, then scores candidates
with actual-item cross-attention. `LCAR-SASRec` and `LCAR-HSTU` share the same
LCAR module; only their parameter-free native-block adapters differ.

## Computation and dimensions

The ML-1M configurations use hidden width 50, memory/query/value width 32,
4 item experts, 128 atoms per expert, 16 activated atoms per expert, 16 selected
original events per atom, and 32 recent events.

1. Project the shared dictionary `[4,128,32]` into 512 virtual-candidate bases
   `[512,50]`. Add each user's logical next-candidate position to initialize
   user-specific states `[B,512,50]`.
2. Run history and independent virtual candidates through the same native block
   weights. History reads causal history; each latent reads valid history and
   itself. No history-to-latent or interlatent attention edges exist. Compute
   history K/V once per layer and process latents in chunks of 128.
3. Average first-block pre-dropout latent-to-history coefficients across native
   heads to select up to 16 original positions per atom: `[B,4,128,16]`.
   SASRec uses native softmax coefficients; HSTU uses signed SiLU coefficients
   and its fixed reference capacity, independent of latent count/chunking.
4. Preserve originals before sequence blocks. Retain the last valid postprocessed
   history state `[B,50]` and project it into one global token `[B,1,32]`.
   Project final latent states into individual summaries `[B,4,128,32]`.
5. Each item produces queries `[1/B,N,4,32]`, sparse supports
   `[1/B,N,4,16]`, and item-only expert gates `[1/B,N,4]`.
6. Concatenate **recent originals; global token; selected-original union; active
   summaries**. Deduplicate the selected union chronologically by event position
   across all four experts. Retain overlap between recent and selected segments.
   Add the corresponding one of four learned source embeddings.
7. Each item query attends to that entire unpooled memory with one softmax.
   Outputs `[B,N,4,32]` enter expert heads `96 → 64 → 1`; final scores `[B,N]`
   are the pi-weighted expert logits. Alpha describes sparse support and the
   auxiliary objective; it does not scale tokens or bias attention.

ML-1M has at most `32 + 1 + 200 + 64 = 297` valid memory tokens per pair.
Empty histories mask every segment, including the global slot despite affine
biases. All-masked attention returns finite zero read outputs.

## Standalone API

```python
from rails.similarities.lcar import LCAR

lcar = LCAR(query_embedding_dim=50, item_embedding_dim=50)
memory = lcar.prepare_user(
    backbone, lengths, history_ids, history_embeddings, payloads)
items = lcar.prepare_items(normalized_item_embeddings, item_ids)
scores, auxiliary_losses = lcar.score_prepared(memory, items)
```

The host may retain its existing scorer. Standalone LCAR does not register or
copy the host. Include both modules' parameters in your optimizer and distributed
forward. Item normalization must agree between training and catalog preparation.

## Optional encoder wrapper

```python
model = lcar.wrap_encoder(backbone)  # LCAREncoder
last_history_state, memory = model.encode_with_history(
    lengths, history_ids, model.get_item_embeddings(history_ids), payloads)
```

`LCAREncoder` owns the host once under `backbone` and LCAR once under `_ndp_module`.
The encoder factory attaches this wrapper for `interaction_module_type="LCAR"`,
using a parameterless native scorer inside the host. SASRec and HSTU contain no
LCAR imports, dictionary parameters, latent branches, or LCAR-specific cache hooks.
Baseline and MoL parameter paths are unchanged.

`encode_history(...)` returns `EncoderOutput` for inspection; its original,
history, latent, selection, mask, and cache fields remain distinct.
`finish_user_memory(encoded)` packs those outputs and prepares base token K/V.
`forward(..., prefix_targets=..., negatives_sampler=..., sampled_loss=...)`
prepares memory and the loss inside the distributed forward.

## Adding a backbone

The generic runner, sparse coder, state objects, scorer, and wrapper have no imports
from `modeling`. Native details live only in [adapters/](adapters/).

```python
from rails.similarities.lcar import register_backbone_adapter
register_backbone_adapter(MyBackbone, MyAdapter)
```

Implement `BackboneAdapter`: `num_layers`, `activation_checkpoint`, `preprocess`,
`target_positions`, `postprocess`, `history_step`, and `latent_step`.
`history_step` returns next history and its reusable read context. `latent_step`
returns next latent states and pre-dropout latent-to-history coefficients
`[B,heads,latent_chunk,history_length]`. Reuse the host's existing weights and
independent-target masks; adapters must not own trainable copies. Pass an explicit
adapter to `prepare_user`, `encode_history`, or `wrap_encoder` for local integration.

The built-in SASRec adapter supports ordinary same-width MultiheadAttention.
The HSTU adapter supports native `rel_bias`/`hstu_rel_bias` pointwise attention,
including relative position/time biases and padded-capacity normalization.
Latent position is the prefix length; time is `request_timestamps` when supplied,
otherwise the last valid prefix timestamp. Future payloads are ignored.

## Training, caches, and checkpoints

- Both supplied LCAR configs use `full_prefix`: the complete available history
  predicts the dataset's supplied next item. Reserved target slots/timestamps are
  masked out. `sampled_prefix` remains an explicit study option; baselines retain
  `all_positions`. Match supervision and target exposure for fair ablations.
- Sparse reconstruction/L1 loss is counted once across positive/negative candidate
  occurrences and experts. Recommendation and auxiliary gradients stay attached;
  dictionary norms are projected after optimizer updates.
- `ModelContext` caches initial atom projections and four projected source offsets.
  Base original/global/summary K/V are user-only. At gather, add `W_K tau_source`
  and `W_V tau_source` without another affine bias, exactly once per occurrence.
- In evaluation without gradients, `history_cache` reuses native history K/V.
  Dictionary changes rerun latent reads; backbone changes reject stale caches.
  Changed prefix/time inputs recompute history. No incremental-speed claim is made.
- `UserMemory.index_batch`, `ItemState.index_batch`, and `slice_candidates` preserve
  masks and provenance. `LearnedBruteForceTopK` performs exact chunked scoring.
  Scoring rejects stale or foreign prepared states. Training never reuses detached
  evaluation caches. Construct/move parameters outside `torch.inference_mode()`.
- Current architecture/checkpoint format is **4**. Earlier LCAR/ExpertHistory
  checkpoints are incompatible and are rejected; there is no implicit migration.
  Historical benchmark artifacts remain historical. Active obsolete implementations
  and configs have been removed.

## Checks (no benchmarks)

```sh
.venv/bin/python -m unittest playground.test_lcar playground.test_retrieval_integration playground.test_benchmark_resume -v
.venv/bin/python -m unittest discover -s playground -p 'test_inference_flops_counter.py' -v
.venv/bin/python playground/validate_ml1m_benchmark.py
```

These cover native separate-target equivalence, history independence, first-block
selection, direct source-tagged attention, global readout/gradients, sparse coding,
chunking/slicing, cache invalidation, strict baseline/v4 checkpoint loads, exact
retrieval, tiny optimizer/DDP checks, configured real-data updates, and operation
accounting. The validation script requires the local CUDA/FBGEMM environment.
