# Implementing LCAR (Latent Cross-Attention Retrieval) in RAILS

## Goal and implementation status

Implement **LCAR (Latent Cross-Attention Retrieval)** as specified in [the model improvement plan](model_idea/retrieval_model_improvement_plan.md). Name the backbone variants **LCAR-SASRec** and **LCAR-HSTU**. Approximate an actual ad token's multi-layer history read with reusable virtual candidates, with one concatenated memory and a sequence of actual-ad expert queries.

**Naming convention:** the implemented scorer is LCARSimilarity, also exported as LCAR, in rails/similarities/lcar. Factory registration is create_lcar_interaction_module, scorer family is LCAR, and configs are sasrec-lcar.gin/hstu-lcar.gin. Registry/report labels are LCAR-SASRec and LCAR-HSTU. The superseded ExpertHistory family and configs have been removed from active code.

**Adopted revision:** keep one dictionary per item expert for sparse ad coding, appended latent tokens, first-block original-event selection, and final latent summaries. For scoring, concatenate recent originals, one projected global-history token, one union of originals selected by all four experts' activated latents, and all individual activated summaries. Add four learned source embeddings before shared scorer K/V projection. The four actual-ad expert query tokens read that single memory through one cross-attention. Pi weights final expert scores only.

**Four-output scoring revision:** project output 2, the last valid postprocessed history state h_u [B,50], to one global token [B,1,32] and include it in the same attention memory as recent originals, selected originals, and summaries. The current integrated API already returns h_u separately from UserMemory. These settings and revised prepared states are implemented as version 4. Historical benchmark checkpoints remain version 3 with two experts, 1,024 latents, and three groups. This adopted design replaces the earlier separate dot-product proposal; its final score remains the pi-weighted expert mixture.

**Current repository status:** the appended-latent and joint-memory architecture is implemented as an independent module with parameter-free native-block adapters. An optional LCAREncoder wrapper owns the host once and LCAR once; native SASRec/HSTU contain no LCAR branches. A standalone API and custom adapter registry support other hosts. The [component README](../rails/similarities/lcar/README.md) documents the actual APIs and cache semantics. New correctness checks validate LCAR independently of the legacy version-2 tests. Existing SVG diagrams still depict the earlier design; their update remains separate from code correctness and benchmark execution.

For the adopted four-expert revision, prepare all 512 latent tokens per prefix; do not prune user preparation using a target ad's unknown active support. Retain 128 atoms and up to 16 activations per expert, and 16 selected events per atom. This reduces the prepared latent count by 75% relative to the previous four-expert, 512-atoms-per-expert plan without changing the shared history computation. Keep latent_chunk_size=128, giving four latent chunks per layer. Maximum active summaries remain 64 and maximum ML-1M pair-memory length remains 297. No version-4 benchmark has been run.

| Component | Required behavior |
| --- | --- |
| **1. Joint actual-ad memory read** | Preserve recent originals, one global-history token, a cross-expert selected-original union, and individual active summaries as source-tagged memory tokens; normalize one attention over the whole concatenation per expert query. |
| **2. Multi-layer virtual-candidate reads** | Append all latents, isolate them with independent-target masks, select originals with first-block attention, and summarize with final latent outputs. |
| **3. Item experts** | Each expert owns a query and dictionary. All four query tokens read all concatenated memory, including any dictionary's selected evidence. Item-only pi gates the final expert scores. |

## 1. Parameters and model computation

### 1.1 Initial budgets and input representations

| Setting | Value |
| --- | --- |
| Item experts E | 4 |
| Dictionary atoms per expert M | 128 |
| Total appended latent tokens C_total=E*M | 512 |
| Maximum active latents per ad/expert K_act | 16 |
| Maximum active summaries / selected lists per ad E*K_act | 64 |
| Original events selected per latent b | 16 |
| Recent original events R | 32 |
| Sparse coding iterations | 8 |
| Hidden width d_h | Existing backbone width; 50 in ML-1M |
| Global-history memory tokens | One per nonempty user history, projected 50 -> 32 |
| Learned source embeddings | Four vectors of width 32: recent, global, selected, summary |
| Scorer key/query/value widths | 32 |
| Backbone blocks / heads | Existing settings: SASRec 2 / 1; HSTU 8 / 2 |

The native first-block attention dimensions are the backbone's dimensions, not the scorer's width/head count. Preserve the configured SASRec/HSTU architecture and causal event ordering.

Expert count and memory-source count are independent. Four experts produce four queries/gates and four dictionaries, but share one recent segment, one global token, and one selected-event union. The four source embeddings still identify recent/global/selected/summary tokens, not expert IDs.

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

The logical stream is [X; Z]. All 512 tokens run independently of the actual ad. Their initial projected atom bases may be cached per checkpoint, but user-dependent latent states cannot be cached globally as static dictionary K/V.

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

**HSTU normalization is part of the semantics.** Choose the history-plus-one-virtual-candidate reference under the established padding policy, then use its position/time biases and length normalization for both history and latent streams. A physical length T+512 or a latent microbatch size must not silently replace that reference length. A diagonal latent mask alone does not fix length-dependent scaling.

The implemented HSTU adapter uses the host's fixed padded reference capacity, including reserved output positions. Each virtual candidate occupies logical position L within that capacity; the denominator remains the host capacity regardless of latent count/chunking. Native single-candidate reference checks validate this policy with relative position/time biases.

Expose the ordinary retrieval-style output from this same history stream. With one-based event positions:

\[
\widehat H^{(D)}_u=\operatorname{OutputPostproc}(H^{(D)}_u),\qquad
h_u=\begin{cases}
\widehat H^{(D)}_{u,L_u}, & L_u>0,\\
0\in\mathbb R^{d_h}, & L_u=0.
\end{cases}
\]

Gather implementation index L_u-1 from already-postprocessed history states, not a padded/latent slot; do not postprocess twice. h_u has shape [B,50], retains gradients, and needs no extra backbone pass. At fixed weights and matching input/mask/position-time policies it matches the original retrieval model's last-history output, independently of dictionary values/order. Joint training may change those shared weights.

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

### 1.8 Gather one candidate-specific memory across all four experts

Let the active token identities be:

\[
\mathcal A(i)=\{(e,j):j\in\mathcal A(i,e)\},
\qquad K_i=|\mathcal A(i)|\le64.
\]

Build a single selected-event union across all active lists from all four experts:

\[
S_{u,i}=\operatorname{Unique}
\left(\bigcup_{(e,j)\in\mathcal A(i)}I_{u,e,j}\right),
\qquad
U_{u,i}=|S_{u,i}|\le\min(L,E K_{\rm act}b).
\]

Deduplicate by original event position, preserving repeated item IDs at different positions, and sort the selected union chronologically. All four query experts receive the same union; do not retain expert-specific visibility masks.

Gather four segments without pooling. Project output 2 once per user using a learned affine global projection:

\[
g_u=P_g h_u\in\mathbb R^{32},\qquad M^{\rm global}_{u,i}=[g_u].
\]

The global slot is shared across candidates and valid exactly when L_u>0. Do not retain N physical copies in UserMemory; mask empty-history slots even if P_g has a nonzero bias.

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
[M^{\rm recent}_{u,i};M^{\rm global}_{u,i};M^{\rm selected}_{u,i};M^{\rm latent}_{u,i}]
\in\mathbb R^{T_{u,i}\times32},
\]

\[
T_{u,i}=R_u+G_u+U_{u,i}+K_i,\qquad
G_u=\mathbf1[L_u>0],\qquad R_u=\min(R,L_u).
\]

Preserve the specified literal concatenation. Deduplicate within the selected union across all four experts, while allowing its overlap with the recent segment. A repeated original across these two segments occupies two attention positions; document and test that multiplicity rather than silently removing it. Distinct expert/atom summaries are separate tokens even when their vectors coincide.

Add a learned source embedding after width-32 token projection and before common scorer K/V projection:

\[
\widetilde m_t=m_t+\tau_{\operatorname{source}(t)},\qquad
\tau\in\mathbb R^{4\times32},\qquad
\operatorname{source}\in\{\mathrm{recent,global,selected,summary}\}.
\]

Repeated originals across recent/selected segments share their base token but receive their respective source embedding at each occurrence. Source type is distinct from expert/atom identity. No segment is averaged before attention.

Gather/concatenation belongs to per-pair scoring. User preparation stores the global token, all lists, and all summaries without seeing any target. The initial implementation may pad memory to the candidate chunk's maximum T and propagate token masks; padding slots carry no evidence. A packed representation with equivalent pair boundaries is mathematically interchangeable. With L<=200 and current budgets, U<=200, K<=64, R_u<=32, G_u<=1, so T<=297. Across longer histories, use configured bounds rather than hard-coded 297.

### 1.9 One cross-attention with the expert query sequence

Stack all actual-ad expert queries:

\[
Q_i=[q_{i,1};q_{i,2};q_{i,3};q_{i,4}]\in\mathbb R^{4\times32},
\]

\[
\widehat Q_i=Q_iW_Q,\qquad
K_{u,i}=\widetilde{\mathcal T}_{u,i}W_K,\qquad
V_{u,i}=\widetilde{\mathcal T}_{u,i}W_V,
\]

\[
A_{u,i}
=\operatorname{softmax}_{\rm memory}
\left(\widehat Q_iK_{u,i}^\top/\sqrt d+\mathrm{padding\ mask}\right),
\qquad
O_{u,i}=A_{u,i}V_{u,i}.
\]

Start with one scorer attention head. All four expert queries see the entire same concatenation, including the global token and evidence originating from any expert. One softmax per query spans all four segments. Do not run independent recent/global/selected/summary attentions, pool any segment first, or add query-token self-attention.

Default logits contain compatibility between projected queries and source-tagged keys plus masks. Neither pi nor alpha biases or scales memory/attention. Pi is reserved for final expert score mixing. Adding a constant log pi_e to query row e would cancel under the memory softmax and is not expert contribution gating. Do not add source-count priors, per-segment normalizations, or further gates initially; measure attention mass by source before considering those changes.

Common-width original/global/summary token projections and shared scorer K/V projections are learned. Cache base original, global, and summary K/V during user preparation, plus the four model-level projected source offsets. On gather, add the appropriate source offset exactly once. For affine K(m)=W_K m+b_K, K(m+tau_s)=K(m)+W_K tau_s; project offsets without a second bias, and apply the same rule to V. This equals direct projection of the source-tagged concatenation, including originals appearing in two source groups. There is one conceptual shared memory and one joint normalization.

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

### 1.11 All four user outputs feed the same attention

Output 2 is projected into the global memory token, not appended to the 96-input score head or used in a separate dot-product score. Its contextualized native retrieval representation complements recent and selected originals and active latent summaries. The adopted output remains s(u,i)=sum_e pi_e(i)*s_e(u,i), with loss temperature 1. Keep the native-width h_u available for inspection and future separately labeled baseline comparisons.

~~~text
User preparation (target-independent)
  1. Recent originals:               [B,R,32] or packed references
  2. Last valid history state h_u:   [B,50] -> global projection -> [B,1,32]
  3. All selected-original lists:    [B,4,128,16] + packed originals
  4. All latent summaries:           [B,4,128,32]

Pair scoring
  [Recent; Global; Selected; Summaries] -> source embeddings
                      [B,N,T,32], T=R+G+U+K <= 297
                                    │ actual-item queries [N,4,32]
                                    ▼
                    One joint cross-attention -> [B,N,4,32]
                                    │ expert heads + final pi
                                    ▼
                             Scores [B,N]
~~~

The actual-ad union is assembled from output 3 only after item activation. User preparation retains every latent's lists, not a union chosen for an unknown target.

## 2. Encoder/scorer bridge and prepared states

The bridge delegates to LCAR's external encoder runner. It captures originals, performs shared-history/independent-latent native block updates, extracts first-block selections, gathers the last valid postprocessed history state, and projects final latent states. Native SASRec/HSTU details live only in the adapter module; the dictionary, orchestration, memory assembly, and scorer remain backbone-independent.

Implemented standalone API:

~~~python
memory = scorer.prepare_user(backbone, lengths, history_ids, history_embeddings, payloads)
items = scorer.prepare_items(item_embeddings, item_ids)
scores, auxiliary_losses = scorer.score_prepared(memory, items)
~~~

For inspection and cache reuse, encode_history returns EncoderOutput; finish_user_memory consumes that output. A custom adapter can be passed explicitly or registered with register_backbone_adapter. The integrated public encode_with_history still returns a current-user tensor plus UserMemory. Prepared scoring gathers unpooled memory and runs joint attention without rerunning the encoder. Prefix training prepares all states inside distributed forward.

Make output 2 explicit in the integrated contract:

~~~python
last_history_state, memory = model.encode_with_history(
    lengths, history_ids, history_embeddings, payloads)
# last_history_state: [B,d_h], the already-postprocessed native history readout
~~~

The implemented four-source scorer retains last_history_state [B,d_h], global_token [B,1,d_m], global_mask [B,1], and optional base global K/V to UserMemory, populated by integrated and standalone preparation. Derive global_mask from existing lengths, keep it false for empty histories, and never encode history a second time. Preserve gradients, user slicing, dtype/device handling, provenance, and byte accounting. ItemState needs no additional base-item cache for this design. Version-4 UserMemory contains these global fields.

| State | Required contents |
| --- | --- |
| ModelContext | Cached initial token projections in fixed expert/atom order and model version; canonical dictionary parameters remain owned by LCAR. |
| Original input tensors | Per-forward X, valid mask/lengths, positions, IDs, payloads; transient until packing/encoding complete. |
| EncoderOutput / HistoryCache | Original/history/final latent states, first-block selected positions/masks, context, shared per-layer history read contexts and cache provenance. Full attention matrices do not persist. |
| UserMemory | Packed original tokens/payloads, recent references/masks, all latent selected-position lists, all final summary tokens/presence, last_history_state, projected global token/mask, optional base K/V for original/global/summary tokens, complete provenance/version. Global fields are version-4 state. |
| Output 2: last_history_state | Postprocessed last valid history state [B,d_h], returned separately and projected to one global memory token in version 4. Zero when history length is zero. |
| ItemState | IDs, expert query sequence and optional projected queries, active atom indices/masks, coefficients and optional diagnostic alpha, final expert gates pi, layout/version, training sparse-loss statistics. |

Representative tensor axes:

- X / history outputs: [B,T,d_h], mask [B,T].
- Output 2 / last_history_state: [B,d_h]=[B,50], gathered from valid history, not a latent token. Global token [B,1,32], mask [B,1], optional base K/V [B,1,d] / [B,1,d_v].
- Source embeddings: [4,32], model-shared; projected source offsets [4,d] / [4,d_v].
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

Maintain one canonical dictionary parameter registration on the scorer. External adapters consume atoms/token tensors; keep native blocks and token/summary projections registered exactly once. Avoid mutable hidden per-request state, circular module ownership, or detached training caches.

## 3. Implemented module boundaries

| Area | Responsibility |
| --- | --- |
| modeling/sequential/sasrec.py and hstu.py | Native blocks/weights and baseline paths; no LCAR-specific logic. |
| rails/similarities/lcar/encoder.py and modeling/sequential/encoder_utils.py | Independent optional wrapper, attached by the encoder factory; owns each module once and prepares prefix loss inside distributed forward. |
| rails/similarities/lcar/adapters/ | Parameter-free BackboneAdapter protocol/registry plus native SASRec/HSTU adapters; all backbone-specific attention/projection/bias details live here. |
| rails/similarities/lcar/encoding.py | Backbone-independent history reuse, latent chunks, first-block selection, final outputs, checkpointing, and history cache provenance. |
| modeling/similarity_module.py | Baseline scorer bridge and generic single-prefix loss support; LCAR coordination lives in its independent wrapper/runner. |
| rails/similarities/lcar/similarity_fn.py | Canonical dictionary, token/summary/common-memory projections, candidate preparation, joint attention, 96-input expert heads, and pi-only final gating; global projection and four source embeddings/offsets. |
| rails/similarities/lcar/states.py | ModelContext, EncoderOutput, HistoryCache, UserMemory, ItemState, and JointMemory with slicing/version/accounting support. |
| rails/similarities/lcar/sparse_coding.py | Bounded dictionary, unrolled nonnegative sparse inference, support cap, fallback, and differentiable coefficient objective. |
| modeling/similarity_utils.py and configs | LCAR factory/dispatch plus sasrec-lcar.gin and hstu-lcar.gin; obsolete ExpertHistory paths removed. |
| Training and sampled-softmax | Keep one causal-prefix user preparation inside distributed forward; reuse for candidates and count sparse loss once. |
| Retrieval and evaluation | Reuse exact chunked scoring with revised preparation, slicing, state versions, and cache dependencies. |
| playground/test_lcar.py, component README, accounting | Native single-target references, standalone/custom-host integration, full training/retrieval checks, and independent LCAR accounting. SVG refresh remains separate. |

Preserve existing DotProduct/MoL construction, parameter paths, training, checkpoints, and retrieval. No unrelated baseline rewrite is needed.

## 4. Training and gradient requirements

Use `full_prefix` supervision by default: all available history predicts the dataset-supplied next item, with reserved target slots and timestamps masked. Seeded `sampled_prefix` remains an explicit study option; original baselines retain `all_positions`. All latents see only the available prefix; one final-prefix memory cannot supervise earlier cuts. Comparisons must match supervision policy, target exposures, cuts where applicable, negatives, and optimizer steps.

Retain positive/negative candidate preparation, duplicate-positive masking, candidate normalization, and sampled-softmax temperature 1.0. Aggregate sparse reconstruction/L1 loss over candidate occurrences and experts once per logical batch. Initially:

\[
\mathcal L=\mathcal L_{\rm rec}+0.1\,
\operatorname{mean}_{i,e}
\left[\tfrac12\|\bar q_{i,e}-C_e\widetilde\gamma_{i,e}\|_2^2
+0.01\|\widetilde\gamma_{i,e}\|_1\right].
\]

Project atoms into the unit ball after updates. Sparse-code support and selected-event indices remain hard. Continuous user-side latent attention and scorer joint attention remain differentiable. Retained coefficient magnitudes train through the sparse auxiliary objective; no alpha-weighted recommendation read remains in this default.

Require recommendation gradients through atom-to-token projections, all latent block updates, native first-layer Q/K/V, final summary projections, dictionary atoms, actual query/common-memory projections, the global projection and h_u readout, source embeddings, joint cross-attention, item experts, expert score heads, and final pi gates. First-layer selector parameters learn through those latent updates, not through top-b indices. Do not retain a selector-only projection with no differentiable loss path.

Preserve activation checkpointing without recomputing inconsistent stochastic selections. Selection uses pre-attention-dropout coefficients from the actual first-layer read; dropout still follows the native backbone behavior. Compare parallel and separate latent execution with dropout disabled or in evaluation, rather than expecting random masks to match automatically.

## 5. Cache dependencies and checkpoint migration

Separate dependencies now that history cannot read latent tokens:

| Cache | Dependencies |
| --- | --- |
| Shared history layer K/V | Prefix, history embedding/preprocessing, backbone weights, position/time and normalization policy, dtype/device/mode. No dictionary-value dependence when those weights/policies are fixed. |
| Output 2: last valid history state | Same history dependencies plus native output postprocessing and last-valid position. No dictionary/token/summary-projection dependence at fixed native weights. |
| Global token and base K/V | h_u, global projection P_g, and joint K/V projections; global validity depends on history length. Cache once per user. |
| Projected source offsets / typed K/V | Four learned source embeddings and joint K/V projection weights; typed K/V also depend on the corresponding base token caches. A source-embedding update invalidates typed views/offsets without changing native history or h_u. |
| Static initial latent bases | Dictionary and atom-to-token projection weights, dtype/device. User-specific target positions/times remain request dependent. |
| Latent states, summaries, and selections | Shared history plus dictionary, token projection, native blocks, first-block selection policy, output/summary projections, masks/layout. |
| Packed original tokens and optional joint K/V | Original features, raw/common-memory projections, and retained positions from recent lists and all-latent selections. Summary K/V also depend on final latent states and summary/joint projections. |
| ItemState / catalog index | Item IDs/features/embedding weights, item experts/gates, dictionary/coding settings, dtype/device/mode. |

After the first block, latent Q/K/V are user-dependent; do not treat them as static model-only dictionary projections. Prefix-level history/latent caches may be reused only with matching provenance. An isolated dictionary update must invalidate latent outputs and selections while leaving independent history values unchanged.

Use format version 4 for four-source scoring. Record the global projection, four source embeddings and their fixed order (recent, global, selected, summary), global state/mask layout, source-offset cache semantics, and all existing token/common-memory/QKV projections, joint-attention settings, union/overlap policies, 96-input heads, pi-only mixing, backbone policies, selection reduction, and budgets. Version 3 denotes the historical three-source appended-latent implementation; version 2 denotes the earlier preconditioning model. Existing benchmark results/checkpoints retain their original version labels. Earlier versions are rejected; no warm-start mapping is implemented. Full version-4 checkpoints load strictly. Do not silently reinterpret a version-3 checkpoint as the adopted four-source model.

## 6. Exact retrieval and configuration

Reuse LearnedBruteForceTopK with candidate_chunk_size=256 initially. Prepare user memory once, fixed-checkpoint catalog item states once, and score catalog chunks exactly under the joint-memory expert-mixture logits. Preserve IDs, filtering, sorting, output order, user microbatching, and candidate slicing.

Registered scorer-family dispatch is DotProduct -> MIPSBruteForceTopK; MoL -> MoLBruteForceTopK; LCAR -> LearnedBruteForceTopK. LCAR has explicit factory, config, registry, inference adapter, and checkpoint metadata. The obsolete scorer has been removed. Recommendation-accuracy evaluation uses held-out targets, timing disabled, and eval_against_brute_force=false. These settings do not authorize execution now.

The executable version-4 ML-1M LCAR configs explicitly set:

~~~gin
create_lcar_interaction_module.num_item_experts = 4
create_lcar_interaction_module.dictionary_size = 128
create_lcar_interaction_module.max_active_latents = 16
create_lcar_interaction_module.events_per_slot = 16
create_lcar_interaction_module.recent_events = 32
create_lcar_interaction_module.sparse_coding_steps = 8
create_lcar_interaction_module.sparse_l1_weight = 0.01
create_lcar_interaction_module.latent_chunk_size = 128
~~~

Joint scorer attention currently uses one head and the adopted revision retains it; first-block selection uses the host's native heads. latent_chunk_size bounds temporary preparation storage while still processing every atom. Token/summary/common-memory widths and existing ablations are factory options. Version 4 adds a global-token enable/disable control and a source-embedding enable/disable control, both enabled for the adopted four-source configuration, with version-4 metadata for these and existing mask/time/position/normalization/gating policies. Run names include a settings hash; training checkpoints record the host description and operative Gin configuration. The active Gin configs enable both additions.

Keep seed 42, 101 total epochs, existing backbone widths/depths, sequence limit, splits, and initial batch/negative budgets for the first scheduled comparable run. Historical version-3 benchmark configs/checkpoints used two experts and 1,024 latents; do not relabel them as four-expert results. Executable configs are updated; scheduling a four-expert benchmark is a separate action.

## 7. Numerical behavior and edge cases

- Exclude history padding and every non-self latent key before native attention. Restrict selection to valid history columns.
- Preserve native SASRec softmax and HSTU pointwise attention semantics; do not label HSTU coefficients as probabilities.
- Use pre-attention-dropout coefficients and the specified mean across native heads for selection.
- Preserve deterministic top-b tie behavior, repeated-event positions, chronological recent lists, and original payload alignment.
- For short prefixes, retain fewer than R or b events with explicit masks.
- Normalize one scorer attention over every valid concatenated token for each query expert; normalize pi over experts for final scores. Alpha may remain diagnostic but does not enter attention. Summary tokens remain individual projected learned vectors.
- Empty histories produce all-masked memory and finite zero updated query outputs; mask the global slot despite projection biases/source embeddings, and do not expose latent-only self tokens as history evidence. Handle a missing segment without normalizing it separately.
- Deduplicate selected originals across all four experts' active lists. Preserve declared recent/selected overlap as separate memory positions; no subtraction or silent cross-segment deduplication is introduced.
- Independent target position/time and reference normalization must not depend on latent enumeration, count per execution chunk, or batch padding accidents.

## 8. Compute accounting and future comparisons

| Bucket | Work |
| --- | --- |
| Model-only | Cacheable initial atom-to-token bases, sparse solver constants, four learned source embeddings/projected offsets, with refresh costs. |
| U: user-only | Original input, shared history blocks, all latent updates, first-block selection, recent lists, packed original-token projections, final latent-summary projections, last valid history gather/global projection, and optional common K/V caches for all token types. |
| I: item-only | Expert query sequence and optional query/value-space projections, sparse inference/support, pi gates, catalog state; alpha diagnostics if retained. |
| S: per pair | Cross-expert selected union, global/active-summary gathers, four-segment concatenation/masks and source offsets, one joint attention for E queries, expert score heads, and pi-weighted final sum. |

With efficient independent streams, attention interaction work is approximately O(D*(L^2+C_total*L+C_total)*d_a), plus native projection/pointwise work on L+C_total tokens. Selection reuses first-layer attention; its additional work is head reduction/top-b/reference construction. No extra standalone selector matmul or final pooling scan is required.

A naive dense [T+C_total,T+C_total] backend still pays for masked latent-to-latent regions. Record actual executed arithmetic and memory rather than only the allowed-edge formula. Consider latent chunking with shared per-layer history K/V if needed, preserving exact masks, logical positions, and normalization.

Prepared token storage is approximately O((P+1)*d_m+d_h+C_total*(b+d_m)+R), plus masks/metadata. Optional common K/V caches add O((P+1+C_total)*(d+d_v)). After cached projections, pair attention costs O(E*T_pair*(d+d_v)); otherwise add O(T_pair*d_m*(d+d_v)) K/V projection work. Include routing, gather/concatenation, source offsets, and expert heads. Here T_pair=R_valid+G+U+K, G=1 for nonempty histories and 0 otherwise, U<=min(L,E*K_act*b), K<=E*K_act. At current E=4 budgets there are up to 64 active lists, U<=min(L,1024), and T_pair<=297 for ML-1M. Do not claim lower pair cost without measurement.

Output 2 adds B*d_h retained values, a last-valid gather, one native-width-to-memory-width projection per user, and one global-token K/V cache per user. Source embeddings add 4*d_m model parameters and their projected offsets. There is no additional history encoding or separate dot-product branch. The global slot increases nonempty pair-memory length by one, and all four sources share the same attention normalization.

Preserve arithmetic ledger failure on unclassified floating operations; report special functions, top-k, indexing/deduplication, model-only caches, user/item cache bytes, peak memory, and actual union sizes separately. Reconcile prepared-stage totals with the composed model.

Future ablations: matched-supervision baselines; full-history actual-ad-token reference; global-token and source-embedding inclusion; recent-only and selected-original paths; individual latent-summary token inclusion; joint attention versus separately pooled evidence; uniform versus learned final pi mixing; native first-layer selection versus controlled alternatives; one versus multiple experts; token/event/recent budgets and latent chunk sizes. Report per-source attention mass alongside union sizes; do not infer that adding a global token guarantees its use or improves quality. Correctness tests use tiny optimizer steps; no benchmark is launched.

## 9. Correctness requirements before benchmarks

1. **Independent-target equivalence:** batched all-latent outputs and selected scores match separately appended single-latent references in evaluation, including logical positions/time and HSTU normalization.
2. **Permutation/count invariance:** permuting or chunking latent tokens only permutes their outputs/lists; adding other independent latents does not change an existing latent's result.
3. **History independence:** changing only dictionary/token values leaves all history layer outputs/K/V unchanged at fixed backbone/input weights. Perturbing one latent does not affect another latent.
4. **Raw provenance:** selected lists match top-b of the actual first-block history coefficients; gathered scorer K/V come from X, not block outputs.
5. **Attention semantics:** verify native head averaging, pre-dropout selection, self/padding exclusion, negative HSTU coefficients, and deterministic ties.
6. **Full-history access:** every latent can read every valid history event at each block; selected lists never prune sequence processing. Final summaries match projected final latent outputs.
7. **Gradients:** isolate a final-latent/summary loss with no sparse auxiliary term; check nonzero finite gradients in dictionary/token projection and native first-layer Q/K. Verify end-to-end prefix DDP and activation checkpointing.
8. **User-only and causal prefixes:** changing target IDs/queries/codes cannot change prepared memory. Perturb future events/held-out timestamps without affecting earlier-prefix results.
9. **Joint memory and attention:** verify the fixed recent/global/selected/summary order, one global token per nonempty history, one event union across all four experts, chronological original-position deduplication, individual summary tokens, literal recent/selected overlap, variable-length masks, and one softmax over all four segments. Compare all four query outputs against direct source-tagged concatenated-memory attention; ensure each query can read global context and any expert's selected evidence.
10. **Routing and final gates:** verify bounded atoms, nonnegative coefficients, support cap, deterministic fallback, and sparse loss counted once. Changing alpha alone at fixed support/queries/memory must not change attention or scores. Changing only pi must leave attention, query outputs, and expert logits unchanged while changing their final weighted mixture when logits differ.
11. **State/checkpoint behavior:** user/candidate slicing, empty/short histories, strict version-4 reload, explicit version-3 incompatibility or checked warm-start mapping, and complete cache dependency invalidation for global/source parameters as well as existing caches.
12. **Retrieval/compatibility:** exact chunked top-k matches full joint-memory expert-mixture logits; existing DotProduct/MoL parameter/loading/training/retrieval behavior is preserved.
13. **Compute:** U/I/S plus model-only preparation reconcile with actual execution, including every latent token and backend masked-square overhead.
14. **Output 2 and source identity:** compare last_history_state with the original native last-valid history output under matched weights, input/mask/position-time policies, and postprocessing. Verify dictionary independence, zero empty-history outputs, aligned user slicing, one shared global cache across candidates, and no repeated backbone pass. Isolate a global-token loss to test finite gradients through P_g and h_u/backbone. Verify gradients to all four source embeddings, correct recent/selected source assignment for repeated originals, offset application exactly once with affine biases, and direct-versus-cached K/V equality. Padded and packed reference attention must agree when given identical valid tokens.

Also check that no segment is pooled before cross-attention, scorer attention masks do not isolate memory by expert origin, and the new expert head has 96 rather than 227 inputs. Verify cached tokenwise K/V gathers equal direct projection after concatenation.

Acceptance requires these checks for both backbone variants before any version-4 benchmark claim. Historical version-2/version-3 benchmark results do not measure this four-source revision.
