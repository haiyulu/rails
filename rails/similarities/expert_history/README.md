# ExpertHistory similarity

Each item expert reads a shared event-level history through two paths: actual-item
attention over selected events and sparse-coefficient-weighted summaries of the
unselected tail. Dictionaries and their memory-query projections organize user
history independently of candidate items.

## API

```python
memory = scorer.prepare_user(history_states, history_mask, event_payloads)
items = scorer.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = scorer.score_prepared(memory, items)
```

Item embeddings have shape `[1 or batch, candidates, item_dim]`. Scores have shape
`[batch, candidates]`. `UserMemory.index_batch`, `ItemState.index_batch`, and
`ItemState.slice_candidates` preserve alignment without detaching gradients.
`forward` preserves the existing similarity signature and requires explicit
`user_memory`. The legacy user query tensor is used only for batch alignment.

## Numerical and cache behavior

- Nonnegative sparse codes use a fixed-step projected-gradient solver, a hard
  support cap, and a deterministic single-atom fallback for all-zero codes.
  Coefficients are normalized only for summary reads. The L1 objective uses their
  original magnitudes. A Frobenius-norm Lipschitz upper bound provides a cheap,
  conservative step size. Dictionary atoms are projected after optimizer steps.
- User preparation stores full summary statistics and stable deep complements.
  Up to `K_act * events_per_slot` numerical guard events per slot are retained by
  reference to shared K/V storage. These guards do not expand the raw selection.
  The scorer reads guards and subtracts additional union events from the deep
  complement, using FP64 accumulation and separate scales. It never rescans
  omitted history. Guard/complement bytes are included in memory accounting.
- Prepared states are invalid after scorer weights, precision, device, or mode
  change. Model-bridge memory also tracks its source history encoder's version.
  Item caches are constructed only for evaluation; rebuild the evaluation state
  after loading or changing weights or item features.

## Training and evaluation integration

`model.encode_with_history` returns the current user vector and explicit memory.
Both backbones support sampled-prefix loss within their forward call so DDP can
see the complete training graph. The sampler masks all future item IDs and event
payloads. Positive and sampled-negative item states are prepared together; sparse
loss is averaged once over the complete candidate batch. Candidate chunking and
non-reentrant activation checkpointing do not change that averaging.

`LearnedBruteForceTopK` caches item states and scores catalog chunks exactly under
the model logits. The root candidate wrapper retains seen-item filtering.
Evaluation slices memory and user payloads together for user microbatches.

The two ready-to-use configurations are
`configs/ml-1m/sasrec-expert-history.gin` and
`configs/ml-1m/hstu-expert-history.gin`. The experiment registry also exposes
raw-only, equal-weight-tail, shared-memory, one-expert, and full-history ablations.
Training and inference runners require explicit execution; importing the modules
does not run them. The inference runner's optional `--measure-latency` flag enables
CUDA-event timing; otherwise it records logical operation counts only. It reports
model-only preparation separately, user/item cache bytes, and peak memory.

## Correctness checks

```sh
.venv/bin/python -m unittest playground.test_expert_history -v
```

The tests include direct masked-history references, concentrated-tail cases,
chunked gradients, serialization, dictionary constraints, causal prefixes,
exact retrieval, DDP forwarding, real local ML-1M rows for both backbones, and
operation-counter reconciliation. No benchmark entry point is invoked.
