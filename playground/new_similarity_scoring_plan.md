# Implementing LCAR (Latent Cross-Attention Retrieval) in RAILS

## Goal and implementation status

Implement **LCAR (Latent Cross-Attention Retrieval)** as specified in [the model improvement plan](model_idea/retrieval_model_improvement_plan.md). Name the backbone variants **LCAR-SASRec** and **LCAR-HSTU**. Approximate an actual ad token's multi-layer history read with reusable virtual candidates, with one concatenated memory and a sequence of actual-ad expert queries.

**Naming convention:** the implemented scorer is LCARSimilarity, also exported as LCAR, in rails/similarities/lcar. Factory registration is create_lcar_interaction_module, scorer family is LCAR, and configs are sasrec-lcar.gin/hstu-lcar.gin. Registry/report labels are LCAR-SASRec and LCAR-HSTU. ExpertHistory remains a separate legacy family and is not an alias for the new computation.

**Adopted revision:** keep one dictionary per item expert for sparse ad coding, appended latent tokens, first-block original-event selection, and final latent summaries. For scoring, concatenate recent originals, one union of originals selected by both experts' activated latents, and all individual activated summaries. The two actual-ad expert query tokens read that single memory through one cross-attention. Pi weights final expert scores only.

**Current repository status:** the appended-latent and joint-memory architecture is implemented as an independent module with parameter-free native-block adapters. Hosts retain their block weights and delegate through small hooks; a standalone API and custom adapter registry also support hosts without those hooks. The [component README](../rails/similarities/lcar/README.md) documents the actual APIs and cache semantics. New correctness checks validate LCAR independently of the legacy version-2 tests. Existing SVG diagrams still depict the earlier design; their update remains separate from code correctness and benchmark execution.

Benchmark execution remains deferred until explicitly requested. The user has confirmed that processing all 1,024 latent tokens per prefix is affordable; do not prune user preparation using a target ad's unknown active support.

| Innovation | Required behavior |
| --- | --- |
| **1. Joint actual-ad memory read** | Preserve recent originals, a cross-expert selected-original union, and individual active summaries as memory tokens; normalize one attention over the whole concatenation per expert query. |
| **2. Multi-layer virtual-candidate reads** | Append all latents, isolate them with independent-target masks, select originals with first-block attention, and summarize with final latent outputs. |
| **3. Item experts** | Each expert owns a query and dictionary. Both query tokens read all concatenated memory, including either dictionary's selected evidence. Item-only pi gates the final expert scores. |

## 1. Parameters and model computation

### 1.1 Initial budgets and input representations

| Setting | Value |
| --- | --- |
| Item experts E | 2 |
| Dictionary atoms per expert M | 512 |
| Total appended latent tokens C_total=E*M | 1,024 |
| Maximum active latents per ad/expert K_act | 16 |
| Original events selected per latent b | 16 |
| Recent original events R | 32 |
| Sparse coding iterations | 8 |
| Hidden width d_h | Existing backbone width; 50 in ML-1M |
| Scorer key/query/value widths | 32 |
| Backbone blocks / heads | Existing settings: SASRec 2 / 1; HSTU 8 / 2 |

The native first-block attention dimensions are the backbone's dimensions, not the scorer's width/head count. Preserve the configured SASRec/HSTU architecture and causal event ordering.

Let X be original event embeddings after input preprocessing, before the first sequence block. Keep original positions, IDs, masks, time/action payloads, and chronological order. In current ML-1M preprocessing, X contains scaled item lookup, learned positions, and dropout. Both original-event memory segments use projections of X, never contextualized block outputs. Keep individual tokens until the single joint scorer attention.

Each expert owns one bounded dictionary \(C_e\in\mathbb R^{d\times M}\), whose columns are \(c_{e,j}\). Do not create independent selection and summarization dictionaries. Role-specific learned transformations are allowed; atom identities and routing remain shared.

### 1.2 Item experts and regular sparse coding

For item/ad i:

\[
y_i=\operatorname{ItemEncoder}(\mathrm{features}_i),\qquad
q_{i,e}=F_e(y_i),\qquad
\pi_e(i)=\operatorname{softmax}_e(g(y_i)).
\]

Reuse the item expert MLPs, normalization, gates, and regular nonnegative L1 coding:

\[
\bar q_{i,e}=q_{i,e}/\max(\|q_{i,e}\|_2,\epsilon),
\]

\[
\gamma^*_{i,e}=\arg\min_{\gamma\ge0}
\frac12\|\bar q_{i,e}-C_e\gamma\|_2^2+\lambda\|\gamma\|_1,
\qquad \|c_{e,j}\|_2\le1.
\]

Use eight unrolled projected-gradient steps initially:

\[
\gamma^{(n+1)}
=\operatorname{ReLU}\left[
\gamma^{(n)}+\eta_e C_e^\top(\bar q-C_e\gamma^{(n)})
-\eta_e\lambda\mathbf1\right],\qquad
\gamma^{(0)}=0,
\]

\[
\eta_e=0.9/\max(\|C_e\|_F^2,\epsilon),\qquad \lambda=0.01.
\]

Keep at most K_act largest positive coefficients after inference, producing support \(\mathcal A(i,e)\) and \(\widetilde\gamma\). Normalized retained coefficients may be kept for diagnostics:

\[
\alpha_{i,e,j}
=\widetilde\gamma_{i,e,j}/
\sum_{k\in\mathcal A(i,e)}\widetilde\gamma_{i,e,k}.
\]

For an all-zero code, activate the most compatible atom with normalized weight one, retain the zero pre-fallback coefficients for auxiliary loss, and record the fallback. Keep deterministic evaluation ties. Actual-query attention uses q, not its coding-normalized version. Sparse support selects memory tokens. In this default, alpha does not weight a summary sum, scale tokens, or enter attention logits; changing only alpha at fixed support, queries, and memory must leave scores unchanged.

### 1.3 Append all atoms as independent virtual candidates

Use an expert-specific token projection from ad-code width to backbone width:

\[
z^{(0)}_{u,e,j}=P_ec_{e,j}+p_{\rm target}(u)
\in\mathbb R^{d_h}.
\]

Keep backbone-compatible input scaling and positional conventions explicit. Every latent occupies the same logical next-candidate position after its user's valid prefix. It is not an additional historical interaction.

For HSTU, use the same logical candidate time for all latents in that prefix. Use an available request timestamp or a documented prefix-derived fallback; never read held-out future timestamps. Time and position biases must match a separate single-latent reference.

The logical stream is [X; Z]. All 1,024 tokens run independently of the actual ad. Their initial projected atom bases may be cached per checkpoint, but user-dependent latent states cannot be cached globally as static dictionary K/V.

### 1.4 Independent-target masks and sequence updates

Allowed query-to-key interactions are:

\[
\mathcal M=
\begin{bmatrix}
\operatorname{Causal}_L & 0\\
\mathbf1_{C_{\rm total}\times L} & I_{C_{\rm total}}
\end{bmatrix}.
\]

Every history query reads its causal history. Every latent query reads all valid history and its own current token. All history-to-latent and cross-latent edges are forbidden, including between experts.

Run the existing sequence-block equations on the allowed interactions:

\[
H^{(\ell+1)}=F_\ell^{\rm causal}(H^{(\ell)}),\qquad
z^{(\ell+1)}_{u,e,j}
=G_\ell(z^{(\ell)}_{u,e,j},H^{(\ell)}).
\]

History does not become latent-conditioned. The latent token becomes user-history-aware through every block, matching a causal candidate-last ranker's dependency pattern. Changing only the dictionary/token values must not change the history stream when backbone parameters are fixed.

A joint masked implementation or equivalent two-stream implementation is acceptable. Both must share native block parameters and reproduce separate single-latent inference. Reuse history Q/K/V and pointwise transforms across latents rather than rerunning a full history encoder for every atom.

**HSTU normalization is part of the semantics.** Choose the history-plus-one-virtual-candidate reference under the established padding policy, then use its position/time biases and length normalization for both history and latent streams. A physical length T+1024 or a latent microbatch size must not silently replace that reference length. A diagonal latent mask alone does not fix length-dependent scaling.

The implemented HSTU adapter uses the host's fixed padded reference capacity, including reserved output positions. Each virtual candidate occupies logical position L within that capacity; the denominator remains the host capacity regardless of latent count/chunking. Native single-candidate reference checks validate this policy with relative position/time biases.

For padded dense batches or HSTU jagged layouts, explicitly track history and latent token identities, offsets, masks, logical positions, and expert/atom IDs. Place latents logically after each user's valid prefix, not after an arbitrary padding boundary with different semantic positions.

### 1.5 First-block attention supplies raw-event selections

Expose the first block's latent-to-history attention coefficients before attention dropout and value aggregation. These coefficients must be shared with the real differentiable latent read, using identical projections, biases, scaling, and masks.

Let \(\mathcal W^{(1)}_{u,h,e,j,t}\) denote this coefficient. Initially reduce native attention heads with a parameter-free arithmetic mean:

\[
r^{\rm select}_{u,e,j,t}
=\frac1{h_1}\sum_{h=1}^{h_1}\mathcal W^{(1)}_{u,h,e,j,t},
\qquad
I_{u,e,j}=\operatorname{TopB}_{t\ \mathrm{valid}}r^{\rm select}_{u,e,j,t}.
\]

For SASRec use native masked softmax coefficients. For HSTU use native pointwise attention coefficients after its score activation and reference scaling; do not replace them with softmax. HSTU coefficients may be negative. Rank by largest reduced coefficient, masking ineligible positions to negative infinity for selection; zero padding must not outrank valid negative coefficients.

Exclude latent self positions, every other latent, and padded history from selection candidates. Use deterministic evaluation ties. Selection uses first-block input history, which is still original X; a pointwise normalization/projection does not make it a contextualized event.

Top-b extraction is a side output of first-block execution. It does not prune history or cause an additional full relevance pass. Keep the full differentiable attention path even though selected indices do not carry ordinary gradients.

Remove the standalone atom-to-selector projection and its memory-query reuse with final pooling. The native first-block latent query/key parameters now learn through final latent summaries. No new selection-only auxiliary objective is required initially.

### 1.6 Final latent representations supply full-history summaries

Run every latent through all D blocks; each layer can read all valid history plus latent self. Then:

\[
s_{u,e,j}
=W^{\rm sum}_e\operatorname{OutputPostproc}(z^{(D)}_{u,e,j})
\in\mathbb R^{d_v}.
\]

Use the backbone's configured output postprocessing consistently and an expert-specific summary projection into scorer value width. Cache summaries [B,E,M,d_v].

There is no separate final attention-pooling scan over H, no full-history exponential statistics, and no event exclusion from latent reads. Recent and selected events remain accessible in every layer. A summary is the learned virtual-candidate output, not a guaranteed convex combination or lossless representation of history.

For an empty history, mask every memory segment, including final latent summaries, and return zero cross-attention outputs. Latent-only self states must not appear as history evidence.

### 1.7 Preserve recent and selected original memory

From X, keep the latest R valid original positions J_u and all first-block latent selection lists. Deduplicate physical storage by original event position, preserving repeated item interactions at distinct positions.

Project preserved originals to a common memory-token width:

\[
m^{\rm raw}_{u,t}=P_{\rm raw}x_{u,t}\in\mathbb R^{d_m},
\qquad d_m=32.
\]

This projection is separate from the first block's native selection Q/K/V. It can run after original-position packing because it is pointwise. Final summaries already have the same width. The full original history still enters every sequence block; packing affects scorer storage only.

### 1.8 Gather one candidate-specific memory across both experts

Let the active token identities be:

\[
\mathcal A(i)=\{(e,j):j\in\mathcal A(i,e)\},
\qquad K_i=|\mathcal A(i)|\le32.
\]

Build a single selected-event union across all active lists from both experts:

\[
S_{u,i}=\operatorname{Unique}
\left(\bigcup_{(e,j)\in\mathcal A(i)}I_{u,e,j}\right),
\qquad
U_{u,i}=|S_{u,i}|\le\min(L,E K_{\rm act}b).
\]

Deduplicate by original event position, preserving repeated item IDs at different positions, and sort the selected union chronologically. Both query experts receive the same union; do not retain expert-specific visibility masks.

Gather three segments without pooling:

\[
M^{\rm recent}_{u,i}=[m^{\rm raw}_{u,t}]_{t\in J_u},
\qquad
M^{\rm selected}_{u,i}=[m^{\rm raw}_{u,t}]_{t\in S_{u,i}},
\]

\[
M^{\rm latent}_{u,i}=[s_{u,e,j}]_{(e,j)\in\mathcal A(i)}.
\]

Keep one separate token per active latent, with stable expert/atom identity and an active mask. There is no alpha-weighted summary sum. Concatenate the token axis:

\[
\mathcal T_{u,i}
=\operatorname{Concat}_{\rm tokens}
[M^{\rm recent}_{u,i};M^{\rm selected}_{u,i};M^{\rm latent}_{u,i}]
\in\mathbb R^{T_{u,i}\times32},
\]

\[
T_{u,i}=R_u+U_{u,i}+K_i,\qquad R_u=\min(R,L).
\]

Preserve the specified literal concatenation. Deduplicate within the selected union across both experts, while allowing its overlap with the recent segment. A repeated original across these two segments occupies two attention positions; document and test that multiplicity rather than silently removing it. Distinct expert/atom summaries are separate tokens even when their vectors coincide.

Gather/concatenation belongs to per-pair scoring. User preparation stores all lists and summaries without seeing any target. Pad memory to the candidate chunk's maximum T and propagate token masks. With L<=200 and current budgets, U<=200, K<=32, R_u<=32, so T<=264. Across longer histories, use the configured bounds rather than hard-coded 264.

### 1.9 One cross-attention with the expert query sequence

Stack all actual-ad expert queries:

\[
Q_i=[q_{i,1};q_{i,2}]\in\mathbb R^{2\times32},
\]

\[
\widehat Q_i=Q_iW_Q,\qquad
K_{u,i}=\mathcal T_{u,i}W_K,\qquad
V_{u,i}=\mathcal T_{u,i}W_V,
\]

\[
A_{u,i}
=\operatorname{softmax}_{\rm memory}
\left(\widehat Q_iK_{u,i}^\top/\sqrt d+\mathrm{padding\ mask}\right),
\qquad
O_{u,i}=A_{u,i}V_{u,i}.
\]

Start with one scorer attention head. Both expert queries see the entire same concatenation, including summary tokens and selected originals originating from either expert. One softmax per query spans all segments. Do not run independent recent/selected/summary attentions, pool any segment first, or add query-token self-attention.

Default logits contain query-key compatibility plus masks. Neither pi nor alpha biases or scales memory/attention. Pi is reserved for final expert score mixing. Adding a constant log pi_e to query row e would cancel under the memory softmax and is not expert contribution gating.

Common-width token projections and shared scorer K/V projections are learned. Their pointwise nature allows prepare_user to cache K/V for all original and summary tokens, and score_prepared to concatenate gathered K/V with the same result as projecting a concatenated token tensor. In either implementation there is one conceptual shared memory and one joint normalization.

### 1.10 Expert scoring and pi-only final mixture

Each updated query token o_e supplies one expert output, [B,N,E,d_v]. Use:

\[
v^q_{i,e}=W_{vq}q_{i,e},\qquad
f^{\rm score}_{u,i,e}
=[q_{i,e};o_{u,i,e};v^q_{i,e}\odot o_{u,i,e}],
\]

\[
s_e(u,i)=\operatorname{MLP}_e(f^{\rm score}_{u,i,e}),\qquad
s(u,i)=\sum_e\pi_e(i)s_e(u,i).
\]

Initially the score head has 96 inputs and layers 96 -> 64 -> 1. Replace the previous 227-input head, three independent pooled vectors, and three path-present score features. Memory masks handle absent segments and all-empty attention safely.

Pi is produced from item features, normalized over experts, and applied only after expert scores. At fixed queries/memory, changing pi leaves attention weights, output tokens, and individual expert logits unchanged; it changes only the final weighted score.

Output remains unnormalized logits [B,N]. Joint cross-attention aggregates tokens at the final query read, after individual event and summary information has been preserved up to that point. It is not equality to an actual-ad full-history ranker's multi-layer read.

## 2. Encoder/scorer bridge and prepared states

The bridge delegates to LCAR's external encoder runner. It captures originals, performs shared-history/independent-latent native block updates, extracts first-block selections, and projects final latent states. Native SASRec/HSTU details live only in the adapter module; the dictionary, orchestration, memory assembly, and scorer remain backbone-independent.

Implemented standalone API:

~~~python
memory = scorer.prepare_user(backbone, lengths, history_ids, history_embeddings, payloads)
items = scorer.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = scorer.score_prepared(memory, items)
~~~

For inspection and cache reuse, encode_history returns EncoderOutput; finish_user_memory consumes that output. A custom adapter can be passed explicitly or registered with register_backbone_adapter. The integrated public encode_with_history still returns a current-user tensor plus UserMemory. Prepared scoring gathers unpooled memory and runs joint attention without rerunning the encoder. Sampled-prefix training prepares all states inside distributed forward.

| State | Required contents |
| --- | --- |
| ModelContext | Cached initial token projections in fixed expert/atom order and model version; canonical dictionary parameters remain owned by LCAR. |
| Original input tensors | Per-forward X, valid mask/lengths, positions, IDs, payloads; transient until packing/encoding complete. |
| EncoderOutput / HistoryCache | Original/history/final latent states, first-block selected positions/masks, context, shared per-layer history read contexts and cache provenance. Full attention matrices do not persist. |
| UserMemory | Packed original tokens/payloads, recent references/masks, all latent selected-position lists, all final summary tokens/presence, optional cached joint K/V for both token types, complete provenance/version. |
| ItemState | IDs, expert query sequence and optional projected queries, active atom indices/masks, coefficients and optional diagnostic alpha, final expert gates pi, layout/version, training sparse-loss statistics. |

Representative tensor axes:

- X / history outputs: [B,T,d_h], mask [B,T].
- Canonical atoms: [E,M,d]; initial/final latent states: [B,E,M,d_h].
- First-block latent-history coefficients, if materialized: [B,h_1,E,M,T].
- Selection references/masks: [B,E,M,b]; recent references/masks [B,R].
- Packed original tokens: [B,P,d_m], P<=L; optionally cache K/V [B,P,d] / [B,P,d_v].
- Final summary tokens: [B,E,M,d_m], presence [B,E,M]; optional projected summary K/V.
- Item active indices/masks: [1/B,N,E,K_act]; query sequence [1/B,N,E,d]; pi [1/B,N,E].
- Candidate-specific active summaries: [B,N,K_max,d_m]; shared selected union [B,N,U_max].
- Concatenated memory: [B,N,T_max,d_m], mask [B,N,T_max].
- Joint attention: [B,N,E,T_max] for one head; outputs [B,N,E,d_v].
- Expert logits [B,N,E]; final logits [B,N].

Capture top-b and discard transient full attention coefficients when possible; do not detach coefficients used for latent value aggregation. Support dense/jagged masks, empty/short prefixes, shared/per-user candidates, user slicing, and candidate chunking.

Maintain one canonical dictionary parameter registration on the scorer. Backbones consume atoms/token tensors through explicit arguments; keep native blocks and token/summary projections registered exactly once. Avoid mutable hidden per-request state, circular module ownership, or detached training caches.

## 3. Implemented module boundaries

| Area | Responsibility |
| --- | --- |
| modeling/sequential/sasrec.py and hstu.py | Small optional delegation hooks; existing native blocks/weights and baseline paths remain intact. |
| rails/similarities/lcar/backbones.py | Parameter-free BackboneAdapter protocol/registry plus native SASRec/HSTU adapters; all backbone-specific attention/projection/bias details live here. |
| rails/similarities/lcar/encoding.py | Backbone-independent history reuse, latent chunks, first-block selection, final outputs, checkpointing, and history cache provenance. |
| modeling/similarity_module.py | Coordinate original input, first-block selection, final latent output, packing, sampled-prefix loss, and explicit user memory. |
| rails/similarities/lcar/similarity_fn.py | Canonical dictionary, token/summary/common-memory projections, candidate preparation, joint attention, 96-input expert heads, and pi-only final gating. |
| rails/similarities/lcar/states.py | ModelContext, EncoderOutput, HistoryCache, UserMemory, ItemState, and JointMemory with slicing/version/accounting support. |
| rails/similarities/lcar/sparse_coding.py | Bounded dictionary, unrolled nonnegative sparse inference, support cap, fallback, and differentiable coefficient objective. |
| modeling/similarity_utils.py and configs | LCAR factory/dispatch plus sasrec-lcar.gin and hstu-lcar.gin; legacy ExpertHistory remains separate. |
| Training and sampled-softmax | Keep one causal-prefix user preparation inside distributed forward; reuse for candidates and count sparse loss once. |
| Retrieval and evaluation | Reuse exact chunked scoring with revised preparation, slicing, state versions, and cache dependencies. |
| playground/test_lcar.py, component README, accounting | Native single-target references, standalone/custom-host integration, full training/retrieval checks, and independent LCAR accounting. SVG refresh remains separate. |

Preserve existing DotProduct/MoL construction, parameter paths, training, checkpoints, and retrieval. No unrelated baseline rewrite is needed.

## 4. Training and gradient requirements

Keep sampled-prefix supervision with seeded cuts and no held-out target inputs/timestamps. All latents see only the available prefix; one final-prefix memory cannot supervise earlier cuts. Comparisons must match target exposures, sampled cuts, negatives, and optimizer steps.

Retain positive/negative candidate preparation, duplicate-positive masking, candidate normalization, and sampled-softmax temperature 1.0. Aggregate sparse reconstruction/L1 loss over candidate occurrences and experts once per logical batch. Initially:

\[
\mathcal L=\mathcal L_{\rm rec}+0.1\,
\operatorname{mean}_{i,e}
\left[\tfrac12\|\bar q_{i,e}-C_e\widetilde\gamma_{i,e}\|_2^2
+0.01\|\widetilde\gamma_{i,e}\|_1\right].
\]

Project atoms into the unit ball after updates. Sparse-code support and selected-event indices remain hard. Continuous user-side latent attention and scorer joint attention remain differentiable. Retained coefficient magnitudes train through the sparse auxiliary objective; no alpha-weighted recommendation read remains in this default.

Require recommendation gradients through atom-to-token projections, all latent block updates, native first-layer Q/K/V, final summary projections, dictionary atoms, actual query/common-memory projections, joint cross-attention, item experts, expert score heads, and final pi gates. First-layer selector parameters learn through those latent updates, not through top-b indices. Do not retain a selector-only projection with no differentiable loss path.

Preserve activation checkpointing without recomputing inconsistent stochastic selections. Selection uses pre-attention-dropout coefficients from the actual first-layer read; dropout still follows the native backbone behavior. Compare parallel and separate latent execution with dropout disabled or in evaluation, rather than expecting random masks to match automatically.

## 5. Cache dependencies and checkpoint migration

Separate dependencies now that history cannot read latent tokens:

| Cache | Dependencies |
| --- | --- |
| Shared history layer K/V | Prefix, history embedding/preprocessing, backbone weights, position/time and normalization policy, dtype/device/mode. No dictionary-value dependence when those weights/policies are fixed. |
| Static initial latent bases | Dictionary and atom-to-token projection weights, dtype/device. User-specific target positions/times remain request dependent. |
| Latent states, summaries, and selections | Shared history plus dictionary, token projection, native blocks, first-block selection policy, output/summary projections, masks/layout. |
| Packed original tokens and optional joint K/V | Original features, raw/common-memory projections, and retained positions from recent lists and all-latent selections. Summary K/V also depend on final latent states and summary/joint projections. |
| ItemState / catalog index | Item IDs/features/embedding weights, item experts/gates, dictionary/coding settings, dtype/device/mode. |

After the first block, latent Q/K/V are user-dependent; do not treat them as static model-only dictionary projections. Prefix-level history/latent caches may be reused only with matching provenance. An isolated dictionary update must invalidate latent outputs and selections while leaving independent history values unchanged.

Advance the new checkpoint/prepared-state format to version 3, after current version 2. Record token/common-memory/QKV projections, joint-attention settings, cross-expert union/overlap policy, 96-input expert heads, pi-only final mixing, native-block and summary configuration, target mask/position/time/normalization policies, first-block head reduction, and budgets. Version-2 preconditioning checkpoints are not complete appended-latent checkpoints. Warm starts require an explicit checked key mapping; complete new checkpoints load strictly.

## 6. Exact retrieval and configuration

Reuse LearnedBruteForceTopK with candidate_chunk_size=256 initially. Prepare user memory once, fixed-checkpoint catalog item states once, and score catalog chunks exactly under the joint-memory expert-mixture logits. Preserve IDs, filtering, sorting, output order, user microbatching, and candidate slicing.

Registered scorer-family dispatch is DotProduct -> MIPSBruteForceTopK; MoL -> MoLBruteForceTopK; LCAR -> LearnedBruteForceTopK. LCAR has explicit factory, config, registry, inference adapter, and checkpoint metadata. ExpertHistory remains a separate legacy scorer. Recommendation-accuracy evaluation uses held-out targets, timing disabled, and eval_against_brute_force=false. These settings do not authorize execution now.

Both new ML-1M LCAR configs explicitly set:

~~~gin
create_lcar_interaction_module.num_item_experts = 2
create_lcar_interaction_module.dictionary_size = 512
create_lcar_interaction_module.max_active_latents = 16
create_lcar_interaction_module.events_per_slot = 16
create_lcar_interaction_module.recent_events = 32
create_lcar_interaction_module.sparse_coding_steps = 8
create_lcar_interaction_module.sparse_l1_weight = 0.01
create_lcar_interaction_module.latent_chunk_size = 128
~~~

Joint scorer attention currently uses one head; first-block selection uses the host's native heads. latent_chunk_size bounds temporary preparation storage while still processing every atom. Token/summary/common-memory widths and memory-segment ablations are factory options; mask, target-time/position, native normalization, and final-gating policies are recorded in version-3 extra state. Run names include a compact settings hash; training checkpoints also record the host description and operative Gin configuration.

Keep seed 42, 101 total epochs, existing backbone widths/depths, sequence limit, splits, and initial batch/negative budgets for the first scheduled comparable run. Benchmark schedules remain deferred.

## 7. Numerical behavior and edge cases

- Exclude history padding and every non-self latent key before native attention. Restrict selection to valid history columns.
- Preserve native SASRec softmax and HSTU pointwise attention semantics; do not label HSTU coefficients as probabilities.
- Use pre-attention-dropout coefficients and the specified mean across native heads for selection.
- Preserve deterministic top-b tie behavior, repeated-event positions, chronological recent lists, and original payload alignment.
- For short prefixes, retain fewer than R or b events with explicit masks.
- Normalize one scorer attention over every valid concatenated token for each query expert; normalize pi over experts for final scores. Alpha may remain diagnostic but does not enter attention. Summary tokens remain individual projected learned vectors.
- Empty histories produce all-masked memory and finite zero updated query outputs; do not expose latent-only self tokens as history evidence. Handle a missing segment without normalizing it separately.
- Deduplicate selected originals across both experts' active lists. Preserve declared recent/selected overlap as separate memory positions; no subtraction or silent cross-segment deduplication is introduced.
- Independent target position/time and reference normalization must not depend on latent enumeration, count per execution chunk, or batch padding accidents.

## 8. Compute accounting and future comparisons

| Bucket | Work |
| --- | --- |
| Model-only | Cacheable initial atom-to-token bases and sparse solver constants, with refresh costs. |
| U: user-only | Original input, shared history blocks, all latent updates, first-block selection, recent lists, packed original-token projections, final latent-summary projections, and optional common K/V caches for all token types. |
| I: item-only | Expert query sequence and optional query/value-space projections, sparse inference/support, pi gates, catalog state; alpha diagnostics if retained. |
| S: per pair | Cross-expert selected union, active summary-token gathers, three-segment concatenation/masks, one joint attention for E queries, expert score heads, and pi-weighted final sum. |

With efficient independent streams, attention interaction work is approximately O(D*(L^2+C_total*L+C_total)*d_a), plus native projection/pointwise work on L+C_total tokens. Selection reuses first-layer attention; its additional work is head reduction/top-b/reference construction. No extra standalone selector matmul or final pooling scan is required.

A naive dense [T+C_total,T+C_total] backend still pays for masked latent-to-latent regions. Record actual executed arithmetic and memory rather than only the allowed-edge formula. Consider latent chunking with shared per-layer history K/V if needed, preserving exact masks, logical positions, and normalization.

Prepared token storage is approximately O(P*d_m+C_total*(b+d_m)+R), plus masks/metadata. Optional common K/V caches add O((P+C_total)*(d+d_v)). After cached projections, pair attention costs O(E*T_pair*(d+d_v)); otherwise add O(T_pair*d_m*(d+d_v)) K/V projection work. Include routing, gather/concatenation, and expert heads. Here T_pair=R_valid+U+K, U<=min(L,E*K_act*b), K<=E*K_act. At current E=2 budgets there are up to 32 active lists, U<=min(L,512), and T_pair<=264 for ML-1M. Do not claim lower pair cost without measurement.

Preserve arithmetic ledger failure on unclassified floating operations; report special functions, top-k, indexing/deduplication, model-only caches, user/item cache bytes, peak memory, and actual union sizes separately. Reconcile prepared-stage totals with the composed model.

Future ablations: matched-supervision baselines; full-history actual-ad-token reference; recent-only and selected-original paths; individual latent-summary token inclusion; joint attention versus separately pooled evidence; uniform versus learned final pi mixing; native first-layer selection versus controlled alternatives; one versus multiple experts; token/event/recent budgets and latent chunk sizes. No experiments or benchmarks execute as part of this documentation task.

## 9. Correctness requirements before benchmarks

1. **Independent-target equivalence:** batched all-latent outputs and selected scores match separately appended single-latent references in evaluation, including logical positions/time and HSTU normalization.
2. **Permutation/count invariance:** permuting or chunking latent tokens only permutes their outputs/lists; adding other independent latents does not change an existing latent's result.
3. **History independence:** changing only dictionary/token values leaves all history layer outputs/K/V unchanged at fixed backbone/input weights. Perturbing one latent does not affect another latent.
4. **Raw provenance:** selected lists match top-b of the actual first-block history coefficients; gathered scorer K/V come from X, not block outputs.
5. **Attention semantics:** verify native head averaging, pre-dropout selection, self/padding exclusion, negative HSTU coefficients, and deterministic ties.
6. **Full-history access:** every latent can read every valid history event at each block; selected lists never prune sequence processing. Final summaries match projected final latent outputs.
7. **Gradients:** isolate a final-latent/summary loss with no sparse auxiliary term; check nonzero finite gradients in dictionary/token projection and native first-layer Q/K. Verify end-to-end sampled-prefix DDP and activation checkpointing.
8. **User-only and causal prefixes:** changing target IDs/queries/codes cannot change prepared memory. Perturb future events/held-out timestamps without affecting earlier-prefix results.
9. **Joint memory and attention:** verify one event union across both experts, chronological original-position deduplication, individual summary tokens, literal recent/selected overlap, variable-length masks, and one softmax over all segments. Compare both query outputs against direct concatenated-memory attention; ensure either query can read either expert's selected evidence.
10. **Routing and final gates:** verify bounded atoms, nonnegative coefficients, support cap, deterministic fallback, and sparse loss counted once. Changing alpha alone at fixed support/queries/memory must not change attention or scores. Changing only pi must leave attention, query outputs, and expert logits unchanged while changing their final weighted mixture when logits differ.
11. **State/checkpoint behavior:** user/candidate slicing, empty/short histories, strict version-3 reload, explicit incompatible-checkpoint rejection, and complete cache dependency invalidation.
12. **Retrieval/compatibility:** exact chunked top-k matches full joint-memory expert-mixture logits; existing DotProduct/MoL parameter/loading/training/retrieval behavior is preserved.
13. **Compute:** U/I/S plus model-only preparation reconcile with actual execution, including every latent token and backend masked-square overhead.

Also check that no segment is pooled before cross-attention, scorer attention masks do not isolate memory by expert origin, and the new expert head has 96 rather than 227 inputs. Verify cached tokenwise K/V gathers equal direct projection after concatenation.

Acceptance requires these checks for both backbone variants before any benchmark claim. Earlier version-2 test results must not be reported as evidence for this appended-latent revision.
