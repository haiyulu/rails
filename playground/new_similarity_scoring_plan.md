# Implementing expert-conditioned history similarity in RAILS

## Goal and architecture

Implement the three innovations in the [model improvement plan](model_idea/retrieval_model_improvement_plan.md) for both SASRec and HSTU. The [architecture SVG](model_idea/retrieval_model_research_idea.svg) illustrates the data flow. This document translates that design into scorer, training, indexing, and benchmark changes.

The objective is to recover useful ranking-style, actual-ad-aware history interaction while preparing user memory independently of candidate ads. Use the existing four SASRec/HSTU + DotProduct/MoL models as baselines. Improved accuracy and acceptable serving cost must be demonstrated experimentally.

| Innovation | Implementation |
| --- | --- |
| **1. Actual-ad history attention** | Preserve event-level history K/V and let the actual item-expert query attend directly to selected events. |
| **2. Ad-derived latent memory** | Use sparse ad-to-dictionary coefficients to address reusable history selections and summary statistics. Continuous ad-dependent weights mix summaries of the unselected tail. |
| **3. Item-expert-specific interaction** | Each item expert has its own ad query, codebooks, and history-memory views. Mix expert scores after their independent interactions. |

Use one shared history encoder and a dense gated mixture of item experts. Each expert has two evidence paths: selected raw events and activation-weighted tail summaries. Here, "raw events" means preserved event-level representations and payloads.

The implementation name is `ExpertHistorySimilarity`, registered as `interaction_module_type = "ExpertHistory"`. Scoring returns recommendation logits with shape `[batch, candidates]`.

## 1. Specify the model computation

### 1.1 Shared history encoding and item experts

For a valid user history prefix:

\[
H_u=\operatorname{Encoder}(\mathrm{history}_u),\qquad
k_{u,t}=W_K H_{u,t},\qquad v_{u,t}=W_V H_{u,t},\qquad
\bar k_{u,t}=k_{u,t}/\sqrt d.
\]

Share these history keys and values across item experts. Retain event positions, item IDs, masks, and available action/time payloads.

For candidate item/ad `i`:

\[
x_i=\operatorname{ItemEncoder}(\mathrm{features}_i),\qquad
q_{i,e}=F_e(x_i),\qquad
\pi_e(i)=\operatorname{softmax}_e(g(x_i)).
\]

Run all `E` item experts. Each query independently reads history before expert scores are combined. For ML-1M, use the current item embedding as the input feature representation; datasets with richer ad features can extend that input later.

### 1.2 Expert-specific sparse coding and memory queries

Give expert `e` one dictionary `C_e = [c[e,1], ..., c[e,M]]` with `M` learned atoms. For coefficient inference, normalize the expert query as `z = q / max(norm(q), eps)`; raw-event attention still uses the actual query `q`. Solve the nonnegative sparse-coding objective approximately:

\[
\gamma^*_{i,e}=\arg\min_{\gamma\ge0}
\frac12\|z_{i,e}-C_e\gamma\|_2^2+\lambda\|\gamma\|_1,
\qquad \|c_{e,j}\|_2\le1.
\]

Use eight unrolled projected-gradient steps initially, starting at zero:

\[
\gamma^{(n+1)}
=\operatorname{ReLU}\left[
\gamma^{(n)}+\eta_e C_e^\top(z-C_e\gamma^{(n)})-\eta_e\lambda
\right].
\]

Use a stable step size, initially `eta_e = 0.9 / (norm(C_e, 2)**2 + eps)`, refreshed with dictionary weights as model-only preparation. The finite-step result is approximate. Retain at most `K_act` largest positive coefficients after inference, giving `gamma_tilde` and support `A(i,e)`. This hard support cap bounds scorer work.

Normalize retained coefficients to obtain continuous summary weights:

\[
\alpha_{i,e,j}
=\frac{\widetilde\gamma_{i,e,j}}
{\sum_{h\in\mathcal A(i,e)}\widetilde\gamma_{i,e,h}},
\qquad j\in\mathcal A(i,e).
\]

When all inferred coefficients are zero, select the atom with the highest query compatibility and give it normalized weight one. Track the fallback and use deterministic ties in evaluation. Pad active-index lists to `K_act` with an explicit mask; fewer than `K_act` positive coefficients are valid.

A memory slot is `s=(e,j)`. Its history query is

\[
a_{e,j}=A_e c_{e,j}+b_e.
\]

The dictionary represents ad-query structure while the expert-specific projection learns useful history access. Keep the L1 penalty on unnormalized coefficients: normalized nonnegative weights sum to one, so penalizing their L1 norm would not encourage sparsity. Different coefficient values can distinguish ads sharing the same support.

### 1.3 User-only preparation

For every slot, compute

\[
w_{u,s,t}=\exp(a_s^\top\bar k_{u,t}),\qquad
S_{u,s}=\operatorname{TopB}_{t}(a_s^\top\bar k_{u,t}),
\]

where `TopB` retains at most `b` valid event positions. Also store full-history sufficient statistics:

\[
Z_{u,s}=\sum_{t=1}^{L}w_{u,s,t},\qquad
U_{u,s}=\sum_{t=1}^{L}w_{u,s,t}v_{u,t}.
\]

Store the selected event K/V once per user, with slot lists referencing deduplicated storage. Each slot additionally needs a scalar `Z`, a value vector `U`, a numerical scale, and masks. Preparation uses only user history and model parameters.

### 1.4 Raw-event attention and strict tail summaries

For each candidate and expert, form the union of events selected by its active slots:

\[
S_{u,i,e}=\operatorname{Unique}\left(
\bigcup_{j\in\mathcal A(i,e)}S_{u,(e,j)}
\right).
\]

Deduplicate by event position, preserving repeated item interactions at different times.

**Raw path.** Use the actual expert query on retained events:

\[
r^{\rm raw}_{u,i,e}
=\sum_{t\in S_{u,i,e}}
\operatorname{softmax}_{t\in S_{u,i,e}}
(q_{i,e}^{\top}\bar k_{u,t})\,v_{u,t}.
\]

**Tail path.** For every active slot, subtract the entire expert-specific union from the cached statistics:

\[
Z^{\rm tail}_{u,i,s}=Z_{u,s}
-\sum_{t\in S_{u,i,e}}w_{u,s,t},\qquad
U^{\rm tail}_{u,i,s}=U_{u,s}
-\sum_{t\in S_{u,i,e}}w_{u,s,t}v_{u,t}.
\]

Normalize each slot before applying the ad's continuous weights:

\[
z^{\rm tail}_{u,i,s}=U^{\rm tail}_{u,i,s}/Z^{\rm tail}_{u,i,s},\qquad
r^{\rm tail}_{u,i,e}
=\sum_{j\in\mathcal A(i,e)}\alpha_{i,e,j}
z^{\rm tail}_{u,i,(e,j)}.
\]

Compute subtraction weights only for the retained event union. The scorer must not scan omitted history. A tail event may contribute to several normalized latent views; mixing those views is intentional.

**Expert scores and final mixture.**

\[
s_e(u,i)=G_e(q_{i,e},r^{\rm raw}_{u,i,e},r^{\rm tail}_{u,i,e},\mathrm{masks}),
\qquad
s(u,i)=\sum_e\pi_e(i)s_e(u,i).
\]

Use a small MLP per expert with query/raw/tail features and elementwise query/value products after dimension projection. Keep outputs as unnormalized logits. This combines exact attention on selected events with learned tail evidence; it does not assert equivalence to full-history attention.

## 2. Add prepared states and extend the encoder/scorer bridge

The existing `SimilarityModule.forward` receives a rank-2 user tensor and shared or per-user candidate embeddings. `modeling/similarity_module.py` asserts that user-tensor rank. Both backbones already produce event-level states during training, but evaluation currently keeps only the current-user embedding. The new model needs explicit history-memory plumbing.

Add the following scorer-owned interfaces:

~~~python
prepare_user(history_states, history_mask, event_payloads) -> UserMemory
prepare_items(item_embeddings, item_ids, **kwargs) -> ItemState
score_prepared(user_memory, item_state) -> (scores, auxiliary_losses)
~~~

Define typed states with batch slicing and candidate slicing operations:

| State | Required contents |
| --- | --- |
| `UserMemory` | Retained event K/V, original positions/IDs and payloads, slot-to-event indices and selection masks, full-history `U/Z`, scales, history/tail masks, model version. |
| `ItemState` | Candidate IDs, expert queries, active atom indices/masks, sparse coefficients, normalized slot weights, item-only expert gates, candidate layout, model/dictionary version; training-only sparse-coding statistics when requested. |

Representative tensor axes are `[batch, E, M, b]` for slot selections, `[batch, E, M, d_v]` for `U`, and `[batch, E, M]` for `Z`. Item active indices, weights, and masks use `[1/batch, candidates, E, K_act]`. Retained K/V can be packed or padded with explicit masks.

Preserve the public `SimilarityModule.forward(query_embeddings, item_embeddings, **kwargs)` signature. The new scorer requires an explicit `user_memory` keyword and delegates to prepared scoring. Its legacy `query_embeddings` argument is the user-side tensor used for batch alignment; the actual ad queries are produced from item embeddings. Missing user memory is an error. DotProduct and MoL retain their current tensor paths.

Add an explicit history-encoding helper at the model bridge, using each backbone's event-level outputs plus valid positions. Return the current-user tensor and prepared memory together for evaluation. Prepare memory once per request/prefix and pass it to positive/negative scoring and retrieval through explicit arguments; do not hide it in mutable module attributes.

Use auxiliary payload forwarding through `TopKModule.forward(..., **kwargs)` and `CandidateIndex` for `user_memory`. The wrapper still needs a user tensor for batch dimensions. Extend the payload typing to accept prepared-state objects, and slice memory and user payloads together when evaluation splits user batches. Read training/evaluation kwargs without mutating shared state.

Keep all parameterized pieces registered on the scorer so the optimizer, distributed training, and checkpoints discover them. Build inference caches only in evaluation mode. Training prepared states must remain attached to autograd.

## 3. Implement the components and registration

| Area | Planned changes |
| --- | --- |
| `rails/similarities/expert_history/` | Add `ExpertHistorySimilarity`, typed states, sparse coefficient inference, dictionary-to-history queries, memory selection/summaries, two-path attention, item experts, routers, and expert scoring. |
| `modeling/similarity_utils.py` | Add Gin-configurable `create_expert_history_interaction_module(...)` and the `"ExpertHistory"` factory branch. Include dimensions, experts, dictionary size, active-atom budget, selection budget, coefficient-solver settings, and scorer widths in run metadata. |
| `modeling/similarity_module.py` and backbone integration | Add event-state preparation/forwarding and explicit user-memory handling while retaining the baseline paths. Account for differing SASRec/HSTU event-state return values. |
| `train.py` and sampled-softmax integration | Prepare causal-prefix memory, reuse it across candidate scores, and aggregate recommendation/sparse-coding losses correctly. |
| `rails/indexing/learned_top_k.py` and root indexing integration | Add prepared-state exact chunked retrieval and register `LearnedBruteForceTopK`. |
| `data/eval.py` and `eval_from_checkpoint.py` | Prepare history memory, slice it with evaluation microbatches, and choose the exact retrieval method by scorer metadata. |
| Configs and playground scripts | Add the two backbone variants, model tests, staged ablations, explicit benchmark metadata, and prepared-stage compute accounting. |

Constructor dimensions must come from configuration and backbone/item embedding dimensions. Use separate configuration and checkpoint namespaces for the new family. The initial pilot follows the modeling plan: `E=2`, `M=32`, `K_act=2`, `b=16`, eight coefficient-inference steps, and one attention head per expert. Treat widths and budgets as configurable experiment parameters.

Encoder reset loops may touch scorer parameters. Initialize and constrain dictionary atoms after full model construction, and load complete checkpoints after construction. Persist dictionary atoms, expert-specific projections, expert gates, coefficient-solver settings, and normalization settings. A resume restores the learned dictionary without refitting. Any backbone warm start needs an explicit checked key mapping rather than unrestricted non-strict checkpoint loading.

## 4. Adapt next-item training without future leakage

The current `train.py` computes states for a sequence and asks `SampledSoftmaxLoss.forward` to flatten all supervised positions into jagged examples. The loss then separately scores positives and negatives; it keeps only the positive call's auxiliary dictionary. These details require deliberate changes.

### Causal-prefix supervision

Introduce an explicit supervision mode, `"sampled_prefix"`. Initially sample one valid next-item cut per training sequence using a seeded generator, build one user memory for that prefix, and supervise the immediate next item. Sample only cuts whose targets belong to the existing training partition; held-out evaluation items remain held out.

Use the current negative sampler, duplicate-positive masking, supervision weights, and sampled-softmax objective. Reuse each prefix's memory for its positive and all negatives.

Retain the existing all-position mode for baselines. For a controlled comparison, also run baseline models with the same sampled-prefix cuts and target exposures. Original all-position baseline reports are historical references, not evidence that isolates the new architecture.

Never reuse a final-prefix memory for earlier positions: global top-b lists and summary sums include future events even if the encoder's individual states are causal. Any all-position extension must implement prefix-aware selections and statistics, with row indexing matching the flattened targets.

### Candidate preparation and auxiliary losses

For each logical prefix batch, sample negatives, assemble positive-plus-negative candidate occurrences, and prepare their item states once. Score them with the shared user memory and split logits for the unchanged sampled-softmax formula. Preserve candidate normalization identically in training and evaluation.

Initialize random unit-norm dictionary atoms and train jointly with the recommendation model. Backpropagate through the unrolled coefficient solver and retained coefficient values; hard support and event-selection indices do not receive ordinary gradients. Project dictionary atom norms into the unit ball after optimizer updates. Recommendation gradients train item experts, memory-query projections, coefficient weights, and expert scorers.

Aggregate the sparse-coding auxiliary objective over positive-plus-negative candidate occurrences once per logical batch:

\[
\mathcal L_{\rm sparse}
=\operatorname{mean}_{i,e}\left[
\tfrac12\|z_{i,e}-C_e\widetilde\gamma_{i,e}\|_2^2
+\lambda\|\widetilde\gamma_{i,e}\|_1
\right].
\]

Use a named configurable weight for this objective in addition to recommendation loss. Keep the all-zero inferred coefficient vector for its auxiliary loss even when the routing fallback selects one slot. Chunking must preserve the sums/counts and must not duplicate the loss. Evaluation produces no training auxiliary loss. Do not rely on the existing negative-call auxiliary return path, which is currently discarded.

Record dictionary initialization, sparse reconstruction/L1 weights, solver steps and step-size rule, zero-code fallbacks, seeded prefix sampling, target exposures, and optimizer steps. Preserve gradient flow through memory statistics, and measure real-batch peak memory before long runs.

## 5. Implement stable tail exclusion and edge cases

Store a per-slot scale for exponentials and use the same scale for cached statistics and all subtracted event weights. Accumulate summaries with sufficient precision; cancellation can be severe when selected events contain almost all the latent attention mass.

Provide a stable complement representation for each slot: prepare the sum over events outside that slot's own top-b set, then subtract contributions from the additional events in the candidate's union. Its reference equations remain the full-history-minus-union equations above. Rebase the complement scale when needed to retain small positive tail mass; apply it consistently to numerator and denominator. The union always contains the slot's own selection.

Required behavior:

- Mask padding before selection and attention. Select fewer than `b` events for short prefixes.
- Deduplicate by original event position and retain IDs, timestamps, and payload alignment.
- Exclude every event in the expert's union from every active slot's tail.
- For an empty history, produce finite masked zero raw/tail states.
- For an empty tail, produce a zero summary and a tail-present mask.
- Determine actual tail emptiness from event counts/masks; do not discard a nonempty tail solely because subtraction underflowed.
- Compare cached/subtracted summaries with direct masked calculations in stable precision, including highly concentrated latent weights.
- Version user memory by history prefix and model/codebook weights. Version item caches by candidate IDs/features, model/codebook weights, dtype, device, and evaluation mode.

## 6. Add exact full-catalog retrieval and checkpoint evaluation

Implement `LearnedBruteForceTopK` as a `TopKModule` adapter with `candidate_chunk_size=256` as the initial default.

Prepare the catalog's `ItemState` once for a fixed evaluation checkpoint. For each user request, encode the history and prepare `UserMemory` once, then score candidate-state chunks and maintain the running exact top-k. The public return order is `(top_k_scores, top_k_ids)` with shape `[batch,k]`, respecting `sorted`. Maintain original catalog-ID alignment.

Register the adapter in `indexing/utils_rails.py`. Pass explicit user memory through the root `CandidateIndex` payloads, retaining the wrapper's oversampled request and seen-item filtering. Test valid catalogs with enough unseen items and document existing wrapper behavior when fewer than k valid items exist.

In `data/eval.py`, replace the vector-only preparation for this scorer with the paired current-user tensor/user-memory result. Slice every memory tensor and user payload consistently with evaluation microbatching. Training-time evaluation, checkpoint evaluation, and any enabled batch retrieval entry point must use this same path.

Replace the hardcoded exact reference in `eval_from_checkpoint.py` with explicit scorer metadata:

| Scorer | Exact method |
| --- | --- |
| DotProduct | `MIPSBruteForceTopK` |
| MoL | `MoLBruteForceTopK` |
| ExpertHistory | `LearnedBruteForceTopK` |

Recommendation-accuracy runs use held-out target IDs, `eval_against_brute_force=false`, and timing disabled. Retrieval-reference checks compare methods for the same scoring function. Exact catalog retrieval means exact ranking of this model's two-path logits, not equivalence to a full-history reference model.

Use strict loading for new-model checkpoints and preserve existing baseline checkpoint behavior. Rebuild prepared caches after every weight or codebook change.

## 7. Add experiment configs and staged comparisons

Create `configs/ml-1m/sasrec-expert-history.gin` and `configs/ml-1m/hstu-expert-history.gin`. Start from the corresponding backbone settings and dataset protocol; set the new factory, retrieval method, prefix supervision, sparse-coding controls, expert gates, and sparse-coding loss weights explicitly.

Planned Gin wiring, once the named controls are implemented:

~~~gin
train_fn.interaction_module_type = "ExpertHistory"
train_fn.top_k_method = "LearnedBruteForceTopK"
train_fn.supervision_mode = "sampled_prefix"

create_expert_history_interaction_module.num_item_experts = 2
create_expert_history_interaction_module.dictionary_size = 32
create_expert_history_interaction_module.max_active_latents = 2
create_expert_history_interaction_module.sparse_coding_steps = 8
create_expert_history_interaction_module.sparse_l1_weight = 0.01
create_expert_history_interaction_module.events_per_slot = 16
create_expert_history_interaction_module.attention_heads_per_expert = 1
~~~

The existing DotProduct and MoL configs use different score-temperature and auxiliary-loss settings. Configure the new scorer explicitly rather than inheriting those accidentally. Use sampled-softmax temperature `1.0` as the initial setting for the learned logits and record it.

Preserve ML-1M's sequence limit, split, backbone dimensions, initial batch/negative budget, and evaluation metrics. Use 101 total epochs and seed 42 for the first comparable scheduled run, with target exposures and optimizer steps recorded; reduced supervised positions per epoch must be reported. Retain best and final checkpoints, and separate their metrics because the existing split has no independent validation set. Use separate run directories.

Run staged ablations for each backbone:

1. One item expert with code-selected raw attention.
2. Add tail summaries; compare equal versus ad-dependent activation weights.
3. Sweep dictionary size, active-atom budget, and coefficient-inference steps; compare equal and sparse-coefficient weights on identical supports.
4. Add multiple item experts; compare shared versus expert-specific memory banks.
5. Compare against a full-history actual-ad-attention reference and compute/parameter-controlled MoL variants.

Match total selected-event attention, user-memory capacity, backbone settings, and training exposure where the comparison requires it. ML-1M validates mechanics and item-conditioned retrieval; richer ad features require a suitable feature-bearing dataset. Additional seeds establish whether measured gains are consistent.

## 8. Extend benchmark metadata and compute accounting

Update the accuracy and inference benchmark runners to use explicit metadata: model ID, config, scorer family, exact retrieval adapter, supervision mode, preparation adapter, and model/codebook version. Do not identify scorer behavior through name substrings.

Use the existing user/item/scoring boundaries:

| Bucket | Work |
| --- | --- |
| **U: user-only** | History encoder, shared K/V, all-slot latent relevance, top-b selection, event packing, summary/complement preparation, scales and masks. |
| **I: item-only** | Item feature encoding, all expert queries, fixed-step sparse coefficient inference, support selection/normalization, expert gates, and catalog-state materialization. |
| **S: per user–item pair** | Active-slot gathers, per-expert unions/deduplication, actual-query attention, union exclusion, tail normalization/mixing, expert MLPs, and final gated score. |

Count code-to-memory-query construction as model-only preparation if cached across requests, with explicit refresh cost after weight changes. Do not hide user preparation inside strict scoring or rebuild item projections in catalog chunks.

For `C_total = E*M` slots, user latent preparation scales approximately as `O(C_total*L*(d+d_v))` plus selection/packing overhead and the backbone. Summary storage scales as `O(C_total*(d_v+1))` plus scales, complements, selected indices, and retained K/V.

Per-pair work includes roughly `O(sum_e |S_e|*(d+d_v))` raw attention and `O(sum_e K_act*|S_e|*(d+d_v))` union exclusion, plus summary reads and expert scoring. Account for actual union sizes and masks. Item-only sparse inference costs approximately O(E*T*M*d) per candidate for T projected-gradient steps, plus support selection. Cache this result for fixed model/item features. Pair arithmetic is independent of omitted history length; user preparation still depends on it.

Report arithmetic FLOPs separately from comparisons, support/event top-k, gathers, sorting/deduplication, and special operations. Keep unknown floating-point operations as failures. Benchmark wall-clock latency and memory because indexing and irregular memory access can dominate despite low arithmetic counts.

Report parameters, user FLOPs, item preparation FLOPs, strict scoring FLOPs/pair, cached/uncached whole-model work, actual catalog latency, user/item cache bytes, dictionary usage, active-atom counts, expert utilization, and peak memory. Reuse the existing inference workload and batch/history/catalog sweeps, recording expert/slot/event budgets and source/config hashes.

## 9. Validate before full training

Required checks:

1. **Layouts:** shared and per-user candidates, batch/candidate size one, masks, valid IDs/payloads, and logits `[batch,candidates]`.
2. **User-only boundary:** prepared user memory is unchanged when candidate IDs, queries, codes, or activation weights change.
3. **Causality:** perturb future events and held-out targets; earlier-prefix memory, selections, summaries, and logits remain unchanged.
4. **Raw attention:** match direct attention over the same deduplicated selected set and preserve repeated events at distinct positions.
5. **Tail semantics:** union-subtracted summaries match direct masked full-history sums, including cross-slot selected events, concentrated weights, empty tails, and short histories.
6. **Activation:** fixed active atom indices with changed continuous coefficients alter the weighted summary when slot summaries differ.
7. **Sparse coding and experts:** coefficients are nonnegative, support has at most K_act entries, fallback is deterministic, activation weights sum to one, atom norms are bounded, and every expert reads its intended memory. Verify solver gradients and objective behavior for fixed normalized dictionaries.
8. **Training:** finite real-batch sampled-softmax loss, gradients through relevant history/item/summary/gating parameters, and correctly aggregated sparse-coding reconstruction/L1 losses. Candidate chunking preserves scores, loss weighting, and gradients within tolerance.
9. **Cache/serialization:** direct and prepared scores agree; batch/candidate slicing, save/reload, codebook state, and cache invalidation work.
10. **Retrieval:** chunked top-k matches full model scoring with aligned IDs, seen-item filtering, user microbatching, and valid tie handling.
11. **Compatibility:** existing DotProduct/MoL construction, training mode, checkpoint loading, metrics, and retrieval paths retain their behavior.
12. **Compute:** prepared-stage totals reconcile with composed forward; U/I/S scaling follows history, expert/codebook, and user–candidate axes.

Acceptance requires both backbone variants to train, reload, score prepared states consistently, retrieve exactly under the new logits, and have audited accuracy/memory/compute reports. Claim an improvement only when the measured comparisons support it.
