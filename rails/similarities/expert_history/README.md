# ExpertHistory similarity

Each item expert reads user evidence through three paths:

1. Actual-ad attention over the latest configured number of original events.
2. Actual-ad attention over the deduplicated original events selected by its active latents.
3. Sparse-coefficient-weighted full-history contextual summaries.

Original events are captured after input preprocessing and before any sequence
block or dictionary adapter. Selection uses these original embeddings. The entire
valid prefix enters the SASRec/HSTU backbone; each block is preceded by gated
event-to-static-dictionary attention. Each latent then summarizes every valid
final contextualized event, including recent and selected events.

The same bounded dictionary atoms drive sparse ad coding, original-event
selection, inside-block conditioning, and contextual summary queries. Each item
expert has its own dictionary and readout. One shared backbone reads all expert
dictionaries. Its computation never uses a target ad or the target's active set.

## API

The normal model entry point prepares original memory before encoding and final
summaries afterward:

~~~python
user_vector, memory = model.encode_with_history(
    past_lengths, past_ids, past_embeddings, past_payloads)
items = model._ndp_module.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = model._ndp_module.score_prepared(memory, items)
~~~

The scorer also exposes explicit component APIs:

~~~python
context = scorer.prepare_model()
raw = scorer.prepare_raw_history(raw_embeddings, history_mask, payloads, context)
# The model's SASRec/HSTU blocks consume context.atoms through their adapters.
memory = scorer.finish_user_memory(raw, contextual_states, history_mask, context)

# Equivalent scorer-only preparation when both representations already exist:
memory = scorer.prepare_user(raw_embeddings, contextual_states, history_mask, payloads)
~~~

Scorer-only preparation does not run the sequence encoder; callers must supply its
actual contextualized output. The bridge rejects preparation from final states
alone. Recent and selected lists reference packed original K/V shared by storage.
Full-history summaries are normalized vectors; no tail exclusion, complement
statistics, or numerical guard lists are used.

Item embeddings have shape [1 or batch, candidates, item_dim]; scores have shape
[batch, candidates]. UserMemory.index_batch, ItemState.index_batch, and
ItemState.slice_candidates preserve alignment and gradients. The legacy
similarity forward requires explicit user_memory and uses the user vector for
batch alignment.

## Configuration

Both ML-1M ExpertHistory configs set:

| Setting | Value |
| --- | --- |
| Item experts | 2 |
| Dictionary atoms per expert | 512 |
| Maximum active latents per expert | 16 |
| Selected events per latent | 16 |
| Recent original events | 32 |
| Sparse solver steps | 8 |
| Dictionary attention per block | Enabled; one head of width 32 |
| Initial dictionary gate | 0.01 |
| Full-history contextual summaries | Enabled |

The factory exposes these budgets, summary/selection switches, adapter dimensions,
and gate initialization as Gin bindings. The experiment registry offers raw-only,
uniform-summary, ordinary-context, recent-only, no-recent, shared-memory,
one-expert, and full-history selection ablations. No benchmark is run by importing
these modules.

## Training, caches, and checkpoints

Nonnegative sparse coding uses a fixed-step projected-gradient solver, positive
support truncation, and a deterministic one-atom fallback. Normalize coefficients
only for summary mixing; retain their magnitudes for the reconstruction/L1 loss.
A cached Frobenius-norm step bound stays safe across inference-to-training
transitions. Project dictionary atoms into the unit ball after optimizer steps.

Selection and summarization share an atom-to-history query projection so hard
selection does not leave a separate selector projection without gradients.
Dictionary attention is pointwise over static atoms, preserving causal sequence
dependencies. All-position event states remain causal, but global memory lists
and summaries belong to one prefix and cannot supervise earlier cuts.

Both backbones prepare sampled-prefix loss inside their distributed forward.
Positive and negative item states share one prefix memory; aggregate sparse loss
once per logical candidate batch. Activation checkpointing preserves the gradient
path through SASRec adapters and the dictionary.

Prepare and move model parameters outside torch.inference_mode; use inference
mode for execution. Versioned caches require normal parameter version counters.
Cache dictionary projections only in evaluation, and retain autograd in training.
Changing dictionary atoms invalidates model context, user/item states, and
HSTU history caches because the atoms now affect contextualization. Adapter and
other encoder changes invalidate their dependent history states as well.

Checkpoint/prepared-state architecture version is **2**. Complete version-1
two-path checkpoints are incompatible. New checkpoints include block adapters,
separate original/contextual projections, three-path fusion, and recent-event
settings. Existing DotProduct/MoL parameter paths remain compatible.

LearnedBruteForceTopK caches item states and scores catalog chunks exactly under
the new logits. Evaluation slices every memory field alongside user payloads.
Inference accounting charges all block adapters and full-history preparation to
U, item coding to I, and the three prepared readouts to S; model-only dictionary
projection caches are reported separately.

## Correctness checks

~~~sh
.venv/bin/python -m unittest playground.test_expert_history -v
.venv/bin/python -m unittest discover -s playground -p 'test_inference_flops_counter.py' -v
~~~

Checks cover independent three-path attention references, original/contextual
provenance, full-history coverage, coefficient sensitivity, isolated adapter
gradients, causal prefixes, cache invalidation, checkpoint versions, exact
retrieval, sampled-prefix DDP forwarding, baseline checkpoint compatibility,
tiny local real-data updates for both backbones, and operation reconciliation.
These tests do not invoke a benchmark entry point.
