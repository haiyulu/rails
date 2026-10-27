# Expert-conditioned history retrieval: innovations 1, 2, and 3

## Goal and agreed design

Recover as much of a ranking model's target-ad-aware cross-attention over user history as possible, while preparing user memory without seeing any target ad. Improve over HSTU/SASRec + MoL through richer access to history, not just a larger scorer over the final user vector. Accuracy and serving efficiency remain experimental questions.

The model combines three innovations:

| Innovation | Architectural change | Purpose |
| --- | --- | --- |
| **1. Direct ad-aware history interaction** | Actual item-expert queries cross-attend to selected event-level history representations. | Preserve detailed user evidence and let the actual candidate determine its relevance. |
| **2. Shared ad-derived codebooks and tail summaries** | Each code prepares reusable history selections and summaries. An ad's codes select the memory; continuous activation weights combine the tail summaries. | Amortize history organization across ads while retaining information outside the selected events. |
| **3. Item-expert-specific queries and codebooks** | Each item expert emits its own query and has its own codebooks and user-memory views. Combine expert scores after their interactions. | Let different specialists extract different evidence for the same ad. |

The model has **two evidence paths per item expert**: selected raw events and latent-aware tail summaries.

The updated [architecture diagram](retrieval_model_research_idea.svg) shows user-only preparation, item-only preparation, and the two scoring paths for each item expert.

## 1. Model structure and notation

Use a shared SASRec or HSTU history encoder. Retain event-level outputs rather than passing only the final user vector to the scorer. "Raw events" below means preserved individual events with their original identity, action/time payloads, and event-level K/V; it does not mean feeding unembedded raw feature strings directly to attention.

| Symbol | Meaning |
| --- | --- |
| `L` | Number of valid events in the user's causal history prefix |
| `E` | Number of item experts |
| `M` | Dictionary atoms / latent memory slots per item expert |
| `b` | Maximum selected events stored per expert/code slot |
| `d`, `d_v` | History attention key/query width and value width |
| `K_act` | Maximum active atoms / slots per ad and expert, with `K_act <= M` |

For user `u`, the shared encoder produces `H_u`, then shared history keys and values:

\[
H_u=\operatorname{Encoder}(\text{history}_u),\qquad
k_{u,t}=W_K H_{u,t},\quad v_{u,t}=W_V H_{u,t},\quad
\bar k_{u,t}=k_{u,t}/\sqrt d.
\]

For item/ad `i`, a shared item feature encoder feeds `E` expert branches:

\[
x_i=\operatorname{ItemEncoder}(\text{features}_i),\qquad
q_{i,e}=F_e(x_i),\qquad
\pi_e(i)=\operatorname{softmax}_e(g(x_i)).
\]

All item experts run as a dense gated mixture. Keep queries separate through cross-attention and expert scoring so each expert independently reads user evidence.

## 2. Sparse ad coding and continuous activation

Give each expert one learned dictionary `C_e = [c[e,1], ..., c[e,M]]`. Use regular **nonnegative sparse coding** to represent its normalized ad query `z[i,e] = q[i,e] / max(norm(q[i,e]), eps)`:

\[
\gamma^*_{i,e}=\arg\min_{\gamma\ge0}
\frac12\|z_{i,e}-C_e\gamma\|_2^2+\lambda\|\gamma\|_1,
\qquad \|c_{e,j}\|_2\le1.
\]

The norm bound prevents dictionary scale from trivially shrinking the coefficient penalty. Apply the L1 penalty to the unnormalized coefficients `gamma`; applying it to normalized nonnegative weights would give the constant one.

Approximate coefficient inference with a fixed number of unrolled projected-gradient steps, then keep at most `K_act` largest positive coefficients. This truncation is an explicit budget step after the sparse-coding solve. Denote the retained vector by `gamma_tilde` and its support by `A(i,e)`. Normalize retained coefficients for summary mixing:

\[
\alpha_{i,e,j}=\frac{\widetilde\gamma_{i,e,j}}
{\sum_{h\in\mathcal A(i,e)}\widetilde\gamma_{i,e,h}},
\qquad j\in\mathcal A(i,e).
\]

If all inferred coefficients are zero, activate the most compatible atom with normalized weight one and record this fallback. Use deterministic tie handling in evaluation. Two ads with the same active atom indices can still have different coefficient magnitudes and therefore different summary weights.

Each atom addresses a learned history-memory query through an expert-specific projection:

\[
a_{e,j}=A_e c_{e,j}+b_e.
\]

Train dictionary atoms, item experts, and memory-query projections jointly with the recommendation loss and a weighted sparse-coding reconstruction/L1 loss. The dictionary represents ad-query structure; the associated history queries learn which user evidence is useful.

## 3. User-only preparation

For each slot `s=(e,j)`, calculate latent-history relevance:

\[
w_{u,s,t}=\exp(a_s^\top\bar k_{u,t}).
\]

Prepare two kinds of memory:

1. **Event selection:** `S[u,s] = TopB_t(a_s^T kbar[u,t])`, keeping at most `b` valid indices.
2. **Full-history sufficient statistics:**

\[
Z_{u,s}=\sum_{t=1}^{L}w_{u,s,t},\qquad
U_{u,s}=\sum_{t=1}^{L}w_{u,s,t}v_{u,t}.
\]

Store one scalar and one value vector per slot for the tail-summary path.

Keep selected-event storage deduplicated across slots, with slot lists referencing shared event K/V. Unselected K/V need not remain in the prepared inference memory once all selections and statistics are built. During training, preserve autograd through encoder outputs and summary statistics. Hard top-b indices do not supply ordinary gradients to selection scores; the differentiable summary path provides a direct learning route for the memory queries.

No actual target ad, active atom set, or activation weights enter this stage. Dictionaries, their projections, encoder weights, and user history determine the memory. Any ad-dependent query/key crossing belongs in the scorer.

## 4. Per-ad, per-expert scoring

### 4.1 Select and deduplicate the raw evidence

For expert `e`, gather the slots addressed by the ad's active sparse coefficients:

\[
S_{u,i,e}=\operatorname{Unique}\left(\bigcup_{j\in\mathcal A(i,e)}
S[u,(e,j)]\right).
\]

Deduplicate by event position, not item ID: repeated interactions at different times are different events. Preserve event timestamps and original positions. Different experts may legitimately read overlapping events; within one expert, an event appears only once in the raw attention.

### 4.2 Raw-event path: actual-ad attention

\[
\rho_{u,i,e,t}=\operatorname{softmax}_{t\in S_{u,i,e}}
(q_{i,e}^\top\bar k_{u,t}),\qquad
r^{\rm raw}_{u,i,e}=\sum_{t\in S_{u,i,e}}\rho_{u,i,e,t}v_{u,t}.
\]

This evaluates the true expert query on retained event K/V. It is exact for that selected set, not equivalent to full-history attention.

### 4.3 Tail-summary path: subtract the union, normalize, then activate

For every active slot `s`, exclude the **entire expert-specific raw union**, not just the events selected by that slot:

\[
Z^{\rm tail}_{u,i,s}=Z_{u,s}-\sum_{t\in S_{u,i,e}}w_{u,s,t},\qquad
U^{\rm tail}_{u,i,s}=U_{u,s}-\sum_{t\in S_{u,i,e}}w_{u,s,t}v_{u,t}.
\]

\[
z^{\rm tail}_{u,i,s}=U^{\rm tail}_{u,i,s}/Z^{\rm tail}_{u,i,s},\qquad
r^{\rm tail}_{u,i,e}=\sum_{j\in\mathcal A(i,e)}\alpha_{i,e,j}
z^{\rm tail}_{u,i,(e,j)}.
\]

Normalize each slot before mixing, so `alpha` directly controls its contribution. This is a mixture of latent-conditioned tail summaries; it is not asserted to equal softmax attention at an averaged latent or at the actual ad query. The same tail event may contribute to several normalized latent summaries by design.

The scorer recalculates slot weights only for retained events to perform subtraction; it does not scan omitted history. Therefore strict tail exclusion introduces `K_act * |S[u,i,e]|` scorer work and must be included in compute accounting.

Use a stored per-slot log scale for exponential stability and apply it consistently to both cached statistics and subtracted weights. Guard cancellation in tail subtraction with sufficient accumulation precision. If needed, compute a stable complement during user preparation for the slot's own selected set, then subtract only additional union events during scoring. Validate against direct masked sums. Empty tails yield a zero summary plus a tail-present mask; do not divide by zero. Empty histories yield masked zero raw/tail states.

### 4.4 Expert scoring and late mixture

An initial small scorer takes the query, both representations, and masks; include elementwise query/value interactions after any necessary dimension projection:

\[
s_e(u,i)=G_e(q_{i,e},r^{\rm raw}_{u,i,e},r^{\rm tail}_{u,i,e},\text{masks}),
\qquad s(u,i)=\sum_{e=1}^{E}\pi_e(i)s_e(u,i).
\]

Return unnormalized recommendation logits. Normalize raw attention, latent summary activation, and expert gates within their respective axes; do not apply a final candidate softmax inside the scorer. Raw and tail representations are combined by a learned scorer, not presented as an exact reconstruction of full-history attention.

## 5. Training and repository integration

The current bridge in `modeling/similarity_module.py` expects a rank-2 user query tensor. This architecture requires explicit history/prepared-state plumbing; it cannot be installed solely as another vector-in/vector-out similarity head.

Add scorer-owned preparation interfaces:

```python
prepare_user(history_states, history_mask, event_payloads) -> UserMemory
prepare_items(item_features, item_ids) -> ItemState
score_prepared(user_memory, item_state) -> (scores, auxiliary_losses)
```

`UserMemory` contains retained event K/V and payloads, per-slot selection indices, `U/Z` statistics and scales. `ItemState` contains expert queries, active atom indices/masks, sparse coefficients, normalized slot weights, expert gates, and item IDs. Preserve shared-catalog and per-user candidate layouts, output `[batch, candidates]`, and user/item ID alignment.

During next-item training, build memory only from the history prefix available at the scored position. Full-sequence summary statistics leak future events even if individual encoder states are causal. A valid first implementation samples causal cut positions and builds one prefix memory per cut; report this protocol and train comparator models with the same cuts/negative samples. Preserving all-position supervision requires prefix-aware sums and prefix-aware selections, not reuse of the final-prefix memory for every position.

Keep the existing recommendation objective and negative-sampling semantics. Initialize normalized dictionary atoms randomly and learn them jointly with the model. Unroll projected-gradient coefficient inference so active coefficients are differentiable; hard support truncation is piecewise differentiable through retained values, not through the chosen indices. Project atom norms back into the unit ball after optimizer updates. Monitor active-atom counts, zero-code fallbacks, dictionary usage, and expert utilization.

Aggregate sparse-coding reconstruction/L1 losses explicitly in the training loop, once per logical candidate batch; do not assume a dictionary returned only from negative scoring will be optimized. Use distinct configuration/checkpoint namespaces.

Implement exact chunked full-catalog retrieval using prepared user memory and item states first. Existing MoL-specific approximate retrieval guarantees do not automatically transfer. Cache item states only for unchanged model/codebook versions, item features, dtype, and device; user memory additionally depends on the history prefix. Run evaluation with inference caches and training with gradient-carrying states.

## 6. Experiments and acceptance

Implement in stages so each innovation has an interpretable comparison:

1. Keep the existing SASRec/HSTU + MoL results as baselines. Add a full-history actual-ad-attention reference for the attention-quality ceiling; do not assume it is empirically superior before measurement.
2. Use one item expert and code-selected raw events to isolate innovation 1 plus the selection mechanism.
3. Add the tail summaries, comparing equal weights against sparse-coefficient weights on the same active atom set; sweep dictionary size and active-atom budget.
4. Add item experts with separate queries/codebooks and late score mixing. Compare against multiple queries sharing one memory bank to isolate expert-specific history organization.

Use existing backbone widths, sequence limits, training/evaluation splits, and negatives where applicable. Proposed small pilot: `E=2`, `M=32`, `K_act=2`, `b=16`, eight coefficient-inference steps, and one attention head per expert. These are starting hyperparameters, not measured optima. Match total selected-event attention, user-summary memory, and parameter/compute budgets in the relevant ablations. A richer ad-feature experiment requires a dataset containing those features; ML-1M first establishes mechanics and item-conditioned retrieval behavior.

Validate causal prefix masking, top-b padding, event-position deduplication, shared/per-user candidate layouts, empty histories/tails, finite gradients, dictionary save/reload, and direct-versus-cached score equivalence. Specifically verify that union-subtracted summaries match a direct masked history calculation, including events selected by another active slot and numerically concentrated attention weights. Verify prepared user memory is unchanged when only the target ad changes. Hold active atom indices fixed and change coefficients to check the continuous summary read; change expert queries to check independent raw attention. Check nonnegative coefficients, the support budget, normalized activation weights, dictionary norm constraints, and zero-code fallback.

Report HR/NDCG/MRR with exact full-catalog scores; preparation time, per-pair latency/FLOPs, full-catalog latency, peak memory, item/user-cache bytes, code usage, and expert-gate utilization. Record selected-event coverage of full-history attention as a diagnostic, not a proof of equal ranking accuracy. Compare multiple seeds before making improvement claims.

For `C_total = E*M` slots, latent preparation requires roughly `O(C_total*L*(d+d_v))` work in addition to the history encoder and selection overhead. Summary storage is `O(C_total*(d_v+1))`, plus retained events and indices. Item-only work includes the fixed-step sparse-coding solver. Pair scoring includes raw attention, continuous summary reads, and union-exclusion work; it is independent of omitted history length, not independent of expert count or active memory budget. No QPS, kernel speedup, or accuracy claim is made before benchmarking.

## References for context

- [SIM](https://arxiv.org/abs/2006.05639): candidate-conditioned history search followed by detailed attention.
- [TWIN V2](https://arxiv.org/abs/2407.16357): compressed history clusters and target-aware reads.
- [RAILS / MoL](https://arxiv.org/abs/2407.15462): the learned-similarity baseline.

This document specifies a research direction and experimental plan, not a novelty determination or a measured improvement.
