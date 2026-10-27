# Adding an advanced similarity scorer to RAILS

## Goal and concrete example

Add a scorer that works with both SASRec and HSTU through the existing similarity interface. Keep the history encoders and next-item training objective intact, and compare accuracy and inference compute against the existing four baselines.

Use **ResidualMoL** as the concrete first implementation: retain the existing MoL function and add a learned nonlinear user–item interaction branch. This increases the available scoring capacity; improved test accuracy is an experimental question, not a guarantee. The same integration steps apply to another scoring formula.

For encoder output h and candidate embedding e:

\[
u=P_q(h),\qquad v=P_i(e),\qquad
f=[u;v;u\odot v;|u-v|],
\]
\[
s(h,e)=s_{\mathrm{MoL}}(h,e)+\operatorname{MLP}(f).
\]

Initial architecture:

- Preserve the supplied ML-1M MoL branch, including its user-ID component and existing auxiliary losses.
- Query/item residual projections: learned Linear(input_dimension → 32), with bias. Accept constructor dimensions rather than hardcoding 50; ML-1M inputs happen to be 50-dimensional.
- Pair features: 4×32 = 128 values.
- Residual MLP: Linear(128→64), SiLU, Linear(64→32), SiLU, Linear(32→1, bias=False).
- Add the residual directly to the MoL logit. Do not apply a final sigmoid or candidate softmax inside the scorer. Retain MoL's internal temperature and the original sampled-softmax loss temperature of 1.0.
- Initialize the final residual weight to zero after encoder construction. The initial scorer then equals its contained MoL branch. Earlier residual layers can receive zero gradients on the first step; test learning across multiple steps.
- No new auxiliary loss in the first implementation. Return the original MoL auxiliary losses unchanged.

This is a richer scoring head over the existing history vector. A model that cross-attends to the original history tokens would require changing the encoder/scorer data interface and is outside this first extension.

## File structure and responsibilities

```text
rails/
  similarities/
    module.py                         existing SimilarityModule contract
    dot_product_similarity_fn.py      existing baseline
    mol/                              existing MoL implementation
    residual_mol/                     NEW scorer family
      __init__.py                     export ResidualMoLSimilarity
      similarity_fn.py                forward + prepared query/item/scoring paths
      layers.py                       residual feature fusion and pair MLP
  indexing/
    candidate_index.py                existing TopKModule interface
    mol_top_k.py                      existing MoL-specific retrieval algorithms
    learned_top_k.py                  NEW exact retrieval adapter for learned scorers

modeling/
  similarity_module.py                existing encoder → scorer bridge
  similarity_utils.py                 EDIT: Gin builder + factory registration
  sequential/
    encoder_utils.py                  EDIT: optional scorer post-construction initialization hook
    sasrec.py                         existing history encoder; architecture unchanged
    hstu.py                           existing history encoder; architecture unchanged
    losses/sampled_softmax.py         existing training loss; normally unchanged

indexing/
  utils_rails.py                      EDIT: register the new exact retrieval adapter
  candidate_index.py                 existing candidate IDs and seen-item filtering

configs/ml-1m/
  sasrec-residual-mol.gin             NEW SASRec experiment
  hstu-residual-mol.gin               NEW HSTU experiment

train.py                             existing assembly/optimization; normally unchanged
eval_from_checkpoint.py              EDIT: generalize exact-reference selection
eval_batch.py                        EDIT only if adding new batch retrieval experiments

playground/
  test_residual_mol.py                NEW shape/gradient/cache/retrieval tests
  run_ml1m_benchmark.py               EDIT: explicit model/scorer metadata
  benchmark_inference_flops.py        EDIT: new prepared-stage adapter and comparison rows
  inference_flops_counter.py          EDIT only for genuinely new operations
```

`rails/` owns scorer mathematics and retrieval implementations. `modeling/` connects those reusable components to recommendation encoders. The root `indexing/` directory integrates retrieval algorithms with catalog IDs and history filtering. `configs/` chooses a complete experiment. `playground/` holds experiments, tests, and reports.

## Step 1 — Preserve the scorer contract

Subclass `rails.similarities.module.SimilarityModule`. Its required interface is:

```python
def forward(self, query_embeddings, item_embeddings, **kwargs):
    # query_embeddings: [B, Dq]
    # item_embeddings:  [1, N, Di] or [B, N, Di]
    # kwargs may include item_ids, user_ids, timestamps, etc.
    return scores, auxiliary_losses
    # scores: [B, N]; auxiliary_losses: dict[str, scalar Tensor]
```

Support both candidate layouts. Evaluation commonly shares a catalog across users; sampled-softmax training provides different negative candidates for each flattened training position. Also support N=1 for positive scores. Here B during training is the number of valid history positions, not necessarily the original number of users.

Accept and forward `**kwargs`; preserve `user_ids` for the inner MoL branch. Store parameters in registered modules/parameters so the optimizer and checkpoint machinery discover them automatically.

## Step 2 — Separate preparation from strict pair interaction

Implement benchmark- and inference-friendly helpers on the new scorer:

```python
prepare_query(query_embeddings, **kwargs) -> query_state
prepare_items(item_embeddings, item_ids=None, **kwargs) -> item_state
score_prepared(query_state, item_state, **kwargs) -> (scores, auxiliary_losses)
```

`forward` delegates to these methods. Prepared states are scorer-owned structures containing tensors; they are not changes to the existing global SimilarityModule contract.

- Query state: the original h, MoL query components, query-only gate outputs, residual u, and query auxiliary losses.
- Item state: original candidate embeddings/IDs, MoL item components, item-only gate outputs, and residual v. Retain the leading 1-or-B candidate dimension.
- Joint scoring: MoL component contractions, the logit-dependent gate branch and mixing, residual pair features, the residual MLP, and final score addition.
- Reuse existing MoL projection/gating modules and their formulas. Validate the prepared MoL computation against `MoLSimilarity.forward` in evaluation mode. Preserve its inference renormalization behavior.

Do not call the entire inner MoL forward from `score_prepared`, because that would repeat query/item preparation. Query projections and query-only gates belong to the user compute bucket; candidate projections and item-only gates belong to candidate preparation.

Candidate caches are valid only for the same model weights, candidate IDs, dtype, device, and evaluation mode. Rebuild them when the model changes during training or a different checkpoint is loaded.

Keep prepared tensors attached to autograd during training. Materialized inference caches must not become a route for silently detaching training gradients.

## Step 3 — Register construction and initialization

In `modeling/similarity_utils.py`, add a Gin-configurable `create_residual_mol_interaction_module(...)` that constructs the existing MoL branch using `create_mol_interaction_module(...)`, wraps it, and returns `(module, debug_string)`.

Add a `module_type == "ResidualMoL"` branch to `get_similarity_function`. Include the base MoL settings and residual dimensions in the debug string so experiment/checkpoint names remain distinguishable.

Both encoder constructors currently run parameter-reset loops that include scorer parameters. To preserve zero initialization of the final residual layer, add an optional `initialize_after_encoder_construction()` hook call in `get_sequential_encoder`, immediately before returning the model. Only the new scorer implements this hook; it zeros only the final residual weight and leaves the contained MoL branch intact. Existing DotProduct and MoL construction behavior stays unchanged.

Load checkpoint weights after this hook runs. A new ResidualMoL checkpoint has a different parameter namespace from an existing MoL checkpoint; use strict loading for its own checkpoints. Start the first comparison from scratch. Any later MoL warm start needs an explicit checked key mapping into the contained MoL module, rather than unrestricted `strict=False` loading.

## Step 4 — Keep sampled-softmax training compatible

The existing bridge calls `model._ndp_module`, and the optimizer already receives all model parameters. No SASRec/HSTU forward changes are necessary for this vector-based head.

Keep score outputs as unnormalized logits. Preserve positive/negative shapes, masking, and temperature behavior.

Important existing behavior: `SampledSoftmaxLoss.jagged_forward` keeps auxiliary losses from the positive-score call and discards the negative-score call's auxiliary dictionary. Initially retain exactly that behavior and the existing MoL MI/UID losses. A future regularizer requiring negative candidates must explicitly change the loss aggregation; merely returning another negative auxiliary loss will not optimize it. Training loss weights must match the auxiliary keys actually returned.

Check peak memory on a real batch before long training. If necessary, checkpoint the residual pair MLP, recomputing its pair features within the checkpointed function; preserve the batch size, negative count, and global MI-loss definition. Do not average independent chunk-level MI losses and assume that equals the original batch-level loss.

## Step 5 — Add exact catalog retrieval

Implement `LearnedBruteForceTopK` in `rails/indexing/learned_top_k.py`, inheriting `TopKModule`. Its constructor receives the learned scorer, catalog embeddings, catalog IDs, and `candidate_chunk_size=256`.

For the new scorer, materialize item state once, prepare query state once per request, score candidate chunks using `score_prepared`, and maintain the running exact top-k. Keep scores and their original item IDs aligned. The public return order is `(top_k_scores, top_k_ids)` with shapes `[B, k]`, respecting the `sorted` argument. An optional fallback may call the standard forward interface for scorers without preparation helpers; that fallback does not promise cached preparation costs.

Register `top_k_method == "LearnedBruteForceTopK"` in `indexing/utils_rails.py`, passing `model._ndp_module`. Leave seen-item filtering in the existing root CandidateIndex wrapper, including its oversampled k request.

Do not use MIPS or the MoL-specific approximate algorithms for the new score. The residual can change rankings, so their existing retrieval guarantees do not automatically transfer. Exact retrieval is the reference for the first accuracy experiment; approximate candidate generation is a separate later experiment.

## Step 6 — Remove scorer-specific evaluation assumptions

`eval_from_checkpoint.py` currently hardcodes `MoLBruteForceTopK` when constructing an exact reference. Introduce an explicit exact-method mapping based on scorer type:

| Scorer type | Exact retrieval method |
|---|---|
| DotProduct | MIPSBruteForceTopK |
| MoL | MoLBruteForceTopK |
| ResidualMoL | LearnedBruteForceTopK |

Use that mapping for retrieval-reference evaluation. Recommendation-accuracy runs must continue to use held-out target IDs, with `eval_against_brute_force=false` and timing disabled.

Keep score semantics identical between training, direct checkpoint evaluation, cached evaluation, and chunked catalog retrieval. Existing MoL-specific checkpoint key renaming is not a substitute for supporting the new scorer's state layout.

## Step 7 — Add two experiment configs

Start the HSTU config by including the existing matching HSTU+MoL ML-1M config, then override:

```gin
include 'configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin'

train_fn.interaction_module_type = "ResidualMoL"
train_fn.top_k_method = "LearnedBruteForceTopK"

create_residual_mol_interaction_module.residual_dim = 32
create_residual_mol_interaction_module.residual_hidden_dims = [64, 32]
```

Create the SASRec counterpart by including the SASRec+MoL config instead. Preserve the backbone, sequence length, original MoL settings, negatives, batch size, optimizer, normalization, and existing auxiliary-loss weights. Use separate output directories and descriptive run names.

The first comparison has six models: the four existing baselines plus SASRec+ResidualMoL and HSTU+ResidualMoL. Reuse the existing baseline reports/checkpoints while preserving their environment and evaluation protocol.

## Step 8 — Extend benchmark metadata and FLOP accounting

The current playground scripts hardcode four model names and sometimes identify MoL by a `"mol"` substring. Replace those decisions with explicit metadata: model ID, config, scorer type, exact top-k method, and inference adapter. Otherwise ResidualMoL may accidentally be evaluated using a MoL-only adapter.

Add the two new training configurations to the accuracy runner and report generation. Add a prepared-stage adapter to the inference FLOPs runner, using the same U/I/S boundaries as the existing plan.

For the residual branch, count Pq in U, Pi in I, and pair products/differences/absolute values, the residual MLP, and the final score addition in S. Explicitly classify absolute-value operations in the counter, preserving the distinction between arithmetic and separately reported special/non-arithmetic operations. Keep unknown floating-point operations as failures, not zero-cost fallbacks.

Report accuracy together with parameter counts, user FLOPs, strict scoring FLOPs per pair, cached/uncached whole-model FLOPs, and candidate-cache size. A gain in accuracy accompanied by more compute should be reported as such. A later compute-matched MoL ablation can distinguish architectural effects from additional budget.

## Step 9 — Validate before full training

Required tests:

1. Shared `[1,N,D]` and per-user `[B,N,D]` candidates; B=1, N=1, multiple candidates, valid IDs/payloads, and output `[B,N]`.
2. Zero residual after full encoder construction reproduces the contained MoL score in eval mode; test both SASRec and HSTU to catch reset-loop interference.
3. Original-inner-MoL versus prepared-MoL equivalence; direct versus cached ResidualMoL equivalence; candidate subsets and full catalog.
4. Finite sampled-softmax loss and gradients on a real ML-1M batch; residual final weights update, and earlier residual layers receive gradients after the zero-initialized output starts learning.
5. Correct MI/UID auxiliary keys and weights, with no duplicated query regularization from candidate chunking.
6. Exact chunked top-k equals full scoring on small catalogs, with original IDs retained; verify equivalent valid choices at tied boundary scores rather than demanding an arbitrary tie order.
7. Seen-item filtering remains correct, user IDs are forwarded, and cached preparation is never executed inside the strict pairwise scorer.
8. Save/reload and end-to-end checkpoint evaluation produce matching scores and recommendation metrics.
9. All existing DotProduct/MoL tests and baseline construction remain unchanged in behavior; existing checkpoint loading still succeeds.
10. New FLOP buckets reconcile with the composed forward and obey user-only, item-only, and B×N joint-scaling checks.

## Step 10 — Run and report the initial experiment

Run the two new ML-1M variants for the same 101 epochs and seed 42 as the completed baseline comparison, after the shape/gradient/memory/retrieval checks pass. Retain best and final checkpoints. Report final-epoch and best evaluation HR@10 separately, since the existing split has no independent validation set.

Evaluate HR@1/10/50/200, NDCG, and MRR using exact full-catalog scores. Then run the existing inference FLOPs workload (B=1, L=200, T=211, actual full catalog) and its B/L/N sweep for all six models. Preserve manifests, configs, source hashes, validation evidence, and per-operation ledgers.

Acceptance means the new head trains, reloads, retrieves exactly, and has audited compute accounting with both encoders. Claim an accuracy improvement only if measured results support it; stronger evidence requires additional seeds and a controlled compute/parameter comparison.
