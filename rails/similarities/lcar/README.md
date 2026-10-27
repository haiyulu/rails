# LCAR: Latent Cross-Attention Retrieval

LCAR prepares a target-independent memory using learned virtual-candidate tokens,
then scores an actual ad using one shared-memory cross-attention. The supported
variants are LCAR-SASRec and LCAR-HSTU.

## Computation

1. Project each expert's 512 dictionary atoms to the host's hidden width. Initialize
   all 1,024 tokens at the same logical next-candidate position after each prefix.
2. Run independent target reads through the host's native block parameters. History
   is causal and cannot read latents. Each latent reads all valid history and itself,
   never another latent. History projections are shared across latent chunks.
3. Reuse first-block attention coefficients, averaged over native heads before
   attention dropout, to select up to 16 original events per atom. HSTU uses its
   native signed pointwise coefficients; SASRec uses softmax coefficients.
4. Project final latent states to individual 32-dimensional summary tokens. Preserve
   original-event tokens and recent/selected references separately.
5. Each actual ad expert activates up to 16 atoms by regular nonnegative sparse
   coding. Form one original-event union across both experts, gather all active
   summary tokens, and concatenate recent originals, selected originals, and summaries.
6. Both ad query tokens attend to that same memory with one normalization over all
   tokens. Score each updated query using a 96 -> 64 -> 1 expert head. Apply the
   item-only expert gates pi only to the final expert logits.

No segment is pooled before cross-attention. Alpha coefficients determine support
and the sparse objective, not a summary mixture or attention prior. The selected
union is deduplicated by event position across both experts, while overlap with
the recent segment remains present as separate attention positions.

## Standalone API

~~~python
from rails.similarities.lcar import LCAR

lcar = LCAR(query_embedding_dim=50, item_embedding_dim=50)
memory = lcar.prepare_user(
    backbone, lengths, history_ids, history_embeddings, payloads)
items = lcar.prepare_items(normalized_item_embeddings, item_ids)
scores, auxiliary_losses = lcar.score_prepared(memory, items)
~~~

The host may keep its existing DotProduct/MoL similarity module. LCAR does not
register or copy the host as a child. Include both host and LCAR parameters in
the optimizer/distributed forward when using this standalone form. Match item
normalization between training and catalog preparation.

The repository-integrated form uses interaction_module_type="LCAR". A small
delegation hook in each supported host routes to this same external implementation:

~~~python
user_vector, memory = model.encode_with_history(
    lengths, history_ids, history_embeddings, payloads)
items = model._ndp_module.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = model._ndp_module.score_prepared(memory, items)
~~~

For inspection, encode_history returns EncoderOutput with original/history/final
latent states, first-block positions, masks, and a shared HistoryCache. A subsequent
finish_user_memory packs originals and caches tokenwise K/V; no pooling pass runs.

## Adding another backbone

All native SASRec/HSTU knowledge lives in backbones.py. The generic encoder and
scorer do not depend on their implementations. Register a parameter-free adapter:

~~~python
from rails.similarities.lcar import register_backbone_adapter

register_backbone_adapter(MyBackbone, MyAdapter)
~~~

Implement the BackboneAdapter protocol: num_layers, activation_checkpoint,
preprocess, target_positions, postprocess, history_step, and latent_step.
history_step returns the shared next history plus a per-layer read context;
latent_step returns next latent states plus pre-dropout latent-to-history
coefficients. These operations must reuse the host's existing block weights and
obey independent-target masks. An explicit adapter can also be passed to
prepare_user/encode_history without global registration.

The built-in SASRec adapter supports ordinary same-width MultiheadAttention.
The HSTU adapter supports native rel_bias/hstu_rel_bias pointwise attention. It
uses the host's padded reference capacity for normalization, never the number
of latent tokens. Reserved output positions provide the logical next-candidate
slot. Native relative-position weights and shifted time buckets are preserved;
the candidate time uses request_timestamps or the last valid prefix timestamp.
Supplied timestamps after the prefix are ignored.

## Caches and checkpoints

- Initial projected dictionary tokens are model-only cache entries. After reading
  history, latent states are user-specific.
- Shared history K/V depend on the prefix and native backbone/input/output weights,
  not dictionary values. In evaluation without gradients, pass history_cache to
  encode_history to reuse them. Changed prefixes recompute history; changed backbone
  weights reject stale caches. Dictionary changes rerun only latent reads when a
  valid history cache is supplied.
- Native HSTU encode cache calls return an LCAR HistoryCache. Updated prefixes may
  fall back to full history recomputation; no incremental-speed claim is made.
- UserMemory and ItemState retain version checks and slicing helpers. Joint K/V are
  projected during user preparation; query projections and gates during item
  preparation. Pair scoring gathers caches and runs only joint attention/heads.
- Construct/move model parameters outside torch.inference_mode. Training preparation
  stays attached to autograd; caches are reused only for evaluation.
- Architecture/checkpoint version is 3. Version-2 ExpertHistory preconditioning
  checkpoints are not complete LCAR checkpoints. Legacy ExpertHistory remains a
  separate scorer/configuration path; existing DotProduct/MoL paths are preserved.

## Configuration and checks

Use configs/ml-1m/sasrec-lcar.gin or configs/ml-1m/hstu-lcar.gin. Both retain two
experts, 512 atoms/expert, 16 active atoms/expert, 16 selected events/atom, 32 recent
events, and eight sparse solver steps. latent_chunk_size=128 bounds temporary
attention storage while still preparing every latent. The registry exposes LCAR
labels, exact retrieval dispatch, and applicable memory-segment ablations.

~~~sh
.venv/bin/python -m unittest playground.test_lcar playground.test_expert_history -v
.venv/bin/python -m unittest discover -s playground -p 'test_inference_flops_counter.py' -v
~~~

Checks include native separate-target equivalence, latent permutation/chunk-count
invariance, first-block selection, original-event provenance, direct unpooled joint
attention, pi-only final gating, gradients, causal prefixes, cache reuse/invalidation,
strict checkpoints, exact retrieval, DDP/checkpointing, tiny real-data updates, and
operation accounting. A third-backbone adapter and standalone use on unchanged
DotProduct hosts verify the plug-in boundary. These checks do not run benchmarks.
