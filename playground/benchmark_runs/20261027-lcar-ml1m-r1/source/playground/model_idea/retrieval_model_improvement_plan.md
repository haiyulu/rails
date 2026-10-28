# LCAR: Latent Cross-Attention Retrieval

## Goal and design status

**LCAR (Latent Cross-Attention Retrieval)** is the working name for the model combining innovations 1, 2, and 3. Its backbone variants are **LCAR-HSTU** and **LCAR-SASRec**.

Approximate the multi-layer history read of a causal ranking model while preparing user memory before the target ad is known. Preserve original event details and amortize ranking-style history interaction through reusable latent tokens. Improve over SASRec/HSTU + MoL; accuracy remains an experimental question.

**Adopted design:** one dictionary per item expert supplies appended virtual-candidate tokens. First-block attention selects original events; final latent states supply summaries. At scoring, concatenate recent originals, one selected-event union across both experts, and every activated latent summary into a single unpooled memory. The two ad-expert query tokens cross-attend to that shared memory. Item-dependent expert weights pi combine final expert scores only. History uses ordinary causal contextualization and cannot attend to the appended latents.

LCAR is implemented in the independent [lcar package](../../rails/similarities/lcar/README.md), with external native-block adapters for SASRec and HSTU and thin host delegation hooks. It can also be used as a standalone add-on to an unchanged host or extended through a registered third-backbone adapter. Correctness checks cover native single-candidate equivalence, first-block selection, joint attention, gradients, caches, exact retrieval, and training integration. Benchmarks remain deferred. Existing SVG files still depict the earlier preconditioning design; the text schematics below describe LCAR. See [the engineering plan](../new_similarity_scoring_plan.md).

LCARSimilarity (also exported as LCAR), create_lcar_interaction_module, and the sasrec-lcar.gin/hstu-lcar.gin configs use the new name. Legacy ExpertHistory remains a separate implementation/configuration path for compatibility; it is not an alias for LCAR.

| Innovation | Architectural change | Purpose |
| --- | --- | --- |
| **1. Joint actual-ad memory interaction** | A sequence of ad-expert queries attends to one concatenated memory of recent originals, selected originals, and activated latent summaries. | Let actual ads choose among individual detail and contextual-memory tokens with one attention normalization. |
| **2. Shared ad dictionaries and multi-layer latent reads** | Append every dictionary atom as an independent virtual candidate. Reuse its first-block attention for event selection and its final state for summarization. | Approximate how an actual ad reads history through a causal ranker's sequence blocks. |
| **3. Item-expert-specific interaction** | Each item expert owns its query, dictionary, selected-event lists, and latent summaries; combine scores after interaction. | Let experts address different ad interests and user evidence. |

There are three memory segments, not three separately pooled scoring paths. Preserve their individual tokens until joint cross-attention. Each expert query returns one updated vector after reading the entire memory.

## Architecture walkthrough with dimensions

B = users, L = valid history events, N = candidate ads. The scoring diagrams show shared-catalog candidates broadcast across users. Padding and masks handle variable event and active-latent counts.

### Step 1. Prepare all user memories before knowing the ad

~~~text
Original history X                 Two expert dictionaries
[B,L,50]                           [2,512,32]
     │                                   │ project 32 → 50
     │                                   ▼
     │                             1,024 latent tokens
     │                             [B,1024,50]
     └──────────────────┬────────────────┘
                        ▼
               Shared SASRec / HSTU
              History + appended latents
                        │
          ┌─────────────┴────────────────┐
          ▼                              ▼
 First-block latent attention      Final latent states
          │                              │ project 50 → 32
 Select 16 original events               ▼
 per latent                        Individual summaries
 [B,2,512,16]                      [B,2,512,32]

 Preserve original-event tokens separately:
 X → project 50 → 32 → original memory [B,P,32], P ≤ L
~~~

History attends causally to history. Each latent attends to all valid history and itself, independently of other latents. The same dictionary atom supplies its selected-event list and final summary. The full history continues through every block; selection only determines preserved original-event references for scoring. Original storage includes the recent events and all latent selections, deduplicated by original position.

### Step 2. Gather and concatenate individual tokens for an actual ad

~~~text
Candidate ad embedding [N,50]
                  │
           Two item experts
                  │
     ┌────────────┼─────────────────┐
     ▼            ▼                 ▼
 Query tokens   Sparse activation   Expert gates π
 [N,2,32]       Up to 16/expert     [N,2]
                  │
                  ▼
           Gather user memory
                  │
     ┌────────────┼────────────────────────┐
     ▼            ▼                        ▼
 Recent         One selected-event       Individual activated
 originals      union across BOTH        latent summaries
                experts
 [B,N,R,32]     [B,N,U,32]                [B,N,K,32]
     │            │                        │
     └────────────┴────────────┬───────────┘
                              ▼
                     CONCATENATE TOKENS
                       [B,N,T,32]
                       T = R + U + K
~~~

Here R ≤ 32 is the actual recent-event count, K ≤ 32 is the total active-latent count across both experts, and U ≤ min(L,512) is the selected union size. The selected union is deduplicated across both experts by event position and ordered chronologically. It may overlap the recent segment as specified below. Keep every activated summary as its own token; no segment is pooled or alpha-weighted before concatenation. All three segment widths are 32.

### Step 3. Run joint cross-attention, then mix expert scores

~~~text
Expert query sequence               Concatenated memory
[B,N,2,32]                          [B,N,T,32]
       │ Q                                │ K,V
       └────────────────┬─────────────────┘
                        ▼
               ONE CROSS-ATTENTION
          One normalization over all T tokens
                        │
                        ▼
             Updated expert query tokens
                   [B,N,2,32]
                        │
              One score head per expert
                        ▼
                Expert scores [B,N,2]
                        │
                 Weighted sum using π
                        ▼
                 FINAL SCORES [B,N]
~~~

Both expert queries can read the entire shared memory, including originals and summaries originating from either expert. Pi weights final expert scores only: score(u,i) = π₁(i) score₁(u,i) + π₂(i) score₂(u,i). There is no pi attention prior. Cross-attention is the first stage that aggregates the concatenated memory into expert output vectors.

## 1. Notation and computation boundaries

Let \(X=(x_1,\ldots,x_L)\) be original event embeddings captured after input preprocessing but before the first sequence block. Preserve positions, IDs, masks, and available time/action payloads. Raw/original events mean these pre-block representations, not contextualized history outputs.

| Symbol | Meaning | Initial setting |
| --- | --- | --- |
| \(L\) | Valid events in a causal prefix | Existing limit; ML-1M at most 200 |
| \(D,d_h\) | Sequence blocks and hidden width | SASRec: 2 blocks; HSTU: 8; width 50 |
| \(E\) | Item experts | 2 |
| \(M\) | Dictionary atoms per expert | 512 |
| \(C_{\rm total}=EM\) | Appended latent tokens per user prefix | 1,024 |
| \(K_{\rm act}\) | Maximum active latents per ad/expert | 16 |
| \(b\) | Original events selected per latent | At most 16 |
| \(R\) | Recent original events | At most 32 |
| \(d_m,d,d_v\) | Memory token, query/key, and value widths | 32 |
| \(h_1\) | First sequence block's attention heads | Existing backbone setting; SASRec 1, HSTU 2 |

The user confirms processing all 1,024 latent tokens per prefix is affordable. Prepare all of them before any target is known. Top-16 ad activation reduces pair-scoring reads, not user preparation.

Each expert owns one bounded ad-space dictionary:

\[
C_e=[c_{e,1},\ldots,c_{e,M}]\in\mathbb R^{d\times M},
\qquad \|c_{e,j}\|_2\le1.
\]

The same atom identity addresses both a selected-event list and a final latent summary. Learned token projections and backbone transformations provide role flexibility without independent semantic dictionaries.

## 2. Item-only queries and regular sparse coding

For item/ad \(i\):

\[
y_i=\operatorname{ItemEncoder}(\mathrm{features}_i),\qquad
q_{i,e}=F_e(y_i),\qquad
\pi_e(i)=\operatorname{softmax}_e(g(y_i)).
\]

Use the existing item feature representation and candidate normalization. All experts run. Normalize each query only for sparse coding:

\[
\bar q_{i,e}=q_{i,e}/\max(\|q_{i,e}\|_2,\epsilon),
\]

\[
\gamma^*_{i,e}=\arg\min_{\gamma\ge0}
\frac12\|\bar q_{i,e}-C_e\gamma\|_2^2+\lambda\|\gamma\|_1.
\]

Approximate with eight projected-gradient steps, retain at most 16 positive coefficients, and denote their support by \(\mathcal A(i,e)\). Normalized coefficients can be retained for diagnostics:

\[
\alpha_{i,e,j}=
\frac{\widetilde\gamma_{i,e,j}}
{\sum_{k\in\mathcal A(i,e)}\widetilde\gamma_{i,e,k}}.
\]

Use deterministic evaluation ties and a most-compatible single-atom fallback for all-zero coefficients. Penalize unnormalized coefficients, not the normalized weights. Sparse coding remains regular nonnegative L1 coding.

The same active set addresses selected original events and individual latent-summary tokens. In the adopted scorer, coefficients determine hard support; alpha does not pool, scale, or bias summary tokens. Coefficient magnitudes still train through the sparse objective. At fixed queries, support, and prepared memory, changing only alpha leaves scores unchanged. Ads with identical support can still read the shared memory differently through their actual expert queries.

## 3. User-only sequence computation

### 3.1 Preserve originals and append all latent tokens

Preserve original-event memory tokens derived directly from X:

\[
m^{\rm raw}_{u,t}=P_{\rm raw}x_{u,t}\in\mathbb R^{d_m},
\qquad d_m=32.
\]

This pointwise projection does not contextualize events. Scorer K/V are learned projections of the common-width memory tokens and may be cached during user preparation.

Keep recent positions \(J_u\), the latest at most R valid events.

Initialize each latent as a virtual candidate in the backbone's hidden space:

\[
z^{(0)}_{u,e,j}=P_e c_{e,j}+p_{\rm target}(u)
\in\mathbb R^{d_h}.
\]

Here \(p_{\rm target}(u)\) is the candidate-after-prefix position encoding under the backbone's existing convention. Every latent has the same logical next-candidate position. HSTU time biases likewise use one prefix/request-time policy shared by all latents, with no held-out timestamp leakage.

The logical input is:

\[
[H^{(0)};Z^{(0)}]=[X;\{z^{(0)}_{u,e,j}\}_{e,j}].
\]

Projection of static atoms can be cached for a fixed checkpoint, but latent states after reading history are user-dependent.

### 3.2 Apply independent-target masking in every block

For allowed query-to-key interactions, use:

\[
\mathcal M=
\begin{bmatrix}
\operatorname{Causal}_L & 0\\
\mathbf 1_{C_{\rm total}\times L} & I_{C_{\rm total}}
\end{bmatrix}.
\]

Thus:

- Each history event reads itself and preceding history.
- Each latent reads every valid history event and itself.
- History cannot read latents.
- A latent cannot read any other latent.

Do not use an ordinary triangular suffix mask, which makes later latents depend on earlier latent ordering. Padding is never an eligible key. All latent tokens are available simultaneously as independent virtual candidates.

Run the existing SASRec/HSTU block transformations with this mask:

\[
H^{(\ell+1)}=F_\ell^{\rm causal}(H^{(\ell)}),
\]

\[
z^{(\ell+1)}_{u,e,j}
=G_\ell(z^{(\ell)}_{u,e,j},H^{(\ell)}).
\]

History representations are independent of the latent/ad values. This matches the dependency pattern of a causal candidate-last ranker: candidate awareness resides in the candidate token's evolving readout. At the first block, each latent reads original event representations; later blocks expose progressively contextualized history.

Compute history once per layer and share it across all latent reads. Match target position/time biases and attention normalization to a separate one-latent reference; appending tokens must not silently change HSTU's sequence-length scaling. Do not assume a dense masked implementation avoids latent-to-latent computation.

### 3.3 Reuse first-block attention to select original events

Let \(\mathcal W^{(1)}_{u,h,e,j,t}\) be the first sequence block's actual latent-to-history attention coefficient for head h, before attention dropout and value aggregation. It must use the same query/key projections, score transformation, mask, biases, and scaling as the latent's differentiable read.

For SASRec these are masked softmax coefficients. HSTU uses its native pointwise attention coefficients; they need not be nonnegative or sum to one. Restrict selection candidates to valid history positions, excluding latent self-attention and padding.

Use a fixed parameter-free head reduction initially:

\[
r^{\rm select}_{u,e,j,t}
=\frac1{h_1}\sum_{h=1}^{h_1}\mathcal W^{(1)}_{u,h,e,j,t},
\qquad
I_{u,e,j}=\operatorname{TopB}_{t\ \mathrm{valid}}r^{\rm select}_{u,e,j,t}.
\]

Select the largest scores, with deterministic ties in evaluation. Preserve original X-derived memory tokens for those positions, not first-block output states or backbone K/V. The backbone's selection keys and the scorer's original-event memory projections have different responsibilities.

This removes a standalone selector query and separate selector dictionary. The first-block query/key parameters learn through the latent's continuous attention output and final summary even though hard top-b indices remain nondifferentiable. This learning path does not guarantee optimal event selection; measure coverage later.

Selection is extracted during the first block, not in a separate pre-encoder scoring pass. It does not prune any event from later sequence blocks.

### 3.4 Use final latent states as full-history summaries

Continue each latent through all D blocks. Apply the backbone's configured output postprocessing to its final state and project to scorer value width:

\[
\widehat z^{(D)}_{u,e,j}
=\operatorname{OutputPostproc}(z^{(D)}_{u,e,j}),\qquad
s_{u,e,j}=W^{\rm sum}_e\widehat z^{(D)}_{u,e,j}\in\mathbb R^{d_v}.
\]

This final latent representation is the summary. Do not add a separate post-encoder latent-attention pooling stage as the default. Every valid event is accessible at every latent read; recent/selected events are not excluded. These are learned contextual vectors, not necessarily convex combinations or normalized attention sums.

Cache one summary per expert/atom and selection references from the same atom. Deduplicate physical original-event storage across recent events and all selections, using original event positions as identities. Repeated item interactions remain distinct. Physical deduplication does not remove the declared recent/selected overlap from the scored sequence. Empty histories yield masked memory and zero attention outputs, including masking latent-only self representations.

## 4. One concatenated memory and joint expert-query cross-attention

### 4.1 Gather three unpooled memory segments

Gather active atom identities from both experts:

\[
\mathcal A(i)=\{(e,j):j\in\mathcal A(i,e)\},\qquad
K_i=|\mathcal A(i)|\le E K_{\rm act}=32.
\]

Build one event union across every activated latent, not a different union per query expert:

\[
S_{u,i}=\operatorname{Unique}
\left(\bigcup_{(e,j)\in\mathcal A(i)}I_{u,e,j}\right),
\qquad U_{u,i}=|S_{u,i}|\le\min(L,E K_{\rm act}b).
\]

Deduplicate by original event position across both experts' active lists. Sort selected original events chronologically. Each summary retains its expert/atom identity; do not deduplicate distinct summaries merely because their values happen to match.

Gather:

\[
M^{\rm recent}_{u,i}
=[m^{\rm raw}_{u,t}]_{t\in J_u},\qquad
M^{\rm selected}_{u,i}
=[m^{\rm raw}_{u,t}]_{t\in S_{u,i}},
\]

\[
M^{\rm latent}_{u,i}
=[s_{u,e,j}]_{(e,j)\in\mathcal A(i)}.
\]

Do not average or alpha-weight these summary tokens. Concatenate along the memory-token axis:

\[
\boxed{\mathcal T_{u,i}
=\operatorname{Concat}_{\rm tokens}
[M^{\rm recent}_{u,i};M^{\rm selected}_{u,i};M^{\rm latent}_{u,i}]}
\in\mathbb R^{T_{u,i}\times32},
\]

\[
T_{u,i}=R_u+U_{u,i}+K_i,\qquad R_u=\min(R,L).
\]

Pad candidate-specific lengths with an explicit mask. At current budgets, U<=min(L,512), K<=32, and T<=264 for L<=200.

Preserve literal concatenation: the selected union is deduplicated internally across experts, but may overlap the recent segment. An event appearing in both contributes two memory positions and therefore affects joint attention multiplicity. This behavior must be explicit and tested; cross-segment removal is not silently applied. Recent/selected events remain accessible in user-side latent reads as well.

### 4.2 Use ad experts as the query sequence

Stack the actual expert queries:

\[
Q_i=[q_{i,1};q_{i,2}]\in\mathbb R^{2\times32}.
\]

Project memory and queries for a single cross-attention operation:

\[
\widehat Q_i=Q_iW_Q,\qquad
K_{u,i}=\mathcal T_{u,i}W_K,\qquad
V_{u,i}=\mathcal T_{u,i}W_V,
\]

\[
A_{u,i}
=\operatorname{softmax}_{\rm memory}
\left(\frac{\widehat Q_iK_{u,i}^{\top}}{\sqrt d}
+\mathrm{padding\ mask}\right),
\qquad
O_{u,i}=A_{u,i}V_{u,i}.
\]

Use one attention head initially. There is one softmax per expert query over the entire concatenated memory, not separate normalizations for each segment. Both query experts may read originals and summaries selected by either expert. No query-to-query self-attention is added.

The default attention has no pi prior, alpha prior, or coefficient-based token scaling. The actual query-memory compatibility learns the read weights. Cross-attention aggregates memory only at this stage, returning one output token per expert query.

| Tensor | Shape |
| --- | --- |
| Concatenated memory | [B,N,T_max,32] |
| Expert query sequence | [B,N,2,32] |
| Attention weights, one head | [B,N,2,T_max] |
| Updated expert query outputs | [B,N,2,32] |

The single common-width token projection and joint K/V projections are part of the new scorer. Since K/V projection is pointwise, user preparation can cache projected original/latent K/V, and pair scoring can concatenate gathered caches with equivalent results.

### 4.3 Score expert outputs, then apply pi

Each expert scores its own query and updated output:

\[
s_e(u,i)=f_e(q_{i,e},o_{u,i,e}).
\]

Use an initial head with query, output, and their value-space product:

\[
v^q_{i,e}=W_{vq}q_{i,e},\qquad
f^{\rm score}_{u,i,e}
=[q_{i,e};o_{u,i,e};v^q_{i,e}\odot o_{u,i,e}].
\]

With width 32 this is 96 features, followed by 96 -> 64 -> 1. Replace the obsolete 227-feature head and its three pooled vectors/masks. Padding and all-empty cases are handled by the joint attention mask.

Apply the adopted item-only expert weights only after expert scores:

\[
\boxed{s(u,i)=\sum_{e=1}^{2}\pi_e(i)s_e(u,i)}.
\]

Pi determines expert contribution to the final logit, not attention-memory preference. Recommendation output remains [B,N].

~~~mermaid
flowchart TD
    X[Original history X] --> Raw[Original tokens and recent list]
    X --> Blocks[Shared causal history with all independent latent tokens]
    C[One dictionary per expert] --> Init[Initialize 1024 virtual candidates]
    Init --> Blocks
    Blocks --> First[First-block latent-to-history coefficients]
    First --> Lists[Top 16 original positions per latent]
    Blocks --> Summaries[Final latent states projected to width 32]
    Ad[Ad embedding] --> Queries[Two expert query tokens]
    Ad --> Pi[Item-only expert weights pi]
    Queries --> Coding[Sparse active atoms for each expert]
    Coding --> Union[One selected-event union across both experts]
    Lists --> Union
    Union --> Selected[Gather original tokens]
    Raw --> Selected
    Raw --> Recent[Recent original tokens]
    Coding --> Active[Gather individual active summaries]
    Summaries --> Active
    Recent --> Concat[Concatenate all three token segments]
    Selected --> Concat
    Active --> Concat
    Concat --> CA[One joint cross-attention memory]
    Queries --> CA
    CA --> Outputs[Two updated expert query tokens]
    Outputs --> Scores[One score per expert]
    Queries --> Scores
    Scores --> Mix[pi-weighted sum of expert scores]
    Pi --> Mix
    Mix --> Logits[Logits B by N]
~~~

## 5. Training, integration, and versioning

Keep one canonical dictionary per expert registered with the scorer. Token initialization uses its atoms without copying or detaching them; all histories and latent tokens pass through the same sequence-block parameters. The first-block attention coefficients simultaneously support the differentiable latent read and hard event selection. No selector-only projection or separate selection loss is needed for this initial design.

The bridge must capture X, run history plus latent streams, expose first-block selection references and final latent outputs, then pack original memory tokens, optional projected K/V, and all latent summaries. Preparing raw memory entirely before the first block is no longer sufficient because selection comes from that block.

Keep sampled-prefix supervision. Build every user memory from its available prefix, reuse it for positive/negative scores, aggregate recommendation and sparse reconstruction/L1 losses once per logical batch, and preserve the dictionary norm projection after updates. Maintain gradient paths through latent initialization, every latent block, first-layer Q/K, final summary projections, actual query/memory projections, and final expert scoring/gates. Sparse support is hard routing: alpha values have no differentiable recommendation-read path in this default; the sparse auxiliary objective still trains coefficients and coding.

Use exact prepared-state chunked catalog scoring. Separate cache dependencies: ordinary history K/V depend on history/backbone/input processing; final latent memories also depend on the dictionary, token projections, and summary projections. Dictionary-only changes should not change the history stream when backbone parameters are fixed. Advance the checkpoint/prepared-state architecture version beyond the current preconditioning implementation; require explicit checked warm-start mappings. Preserve DotProduct/MoL behavior.

## 6. Validation, cost, and experiments

Before benchmarks, require:

1. Batched independent latents match separate single-latent passes with matched logical position/time and normalization.
2. History outputs remain unchanged when only latent values/order change; permuting latents permutes their outputs and lists.
3. First-block selection matches its actual history-attention coefficients and gathers original X-derived K/V.
4. Every latent can access all valid history events in every block; no selected-event pruning or cross-latent reads.
5. Isolated final-summary losses produce gradients through token projections, dictionary atoms, and first-block Q/K without sparse auxiliary loss.
6. User-only invariance, causal prefixes, empty/short histories, overlap/deduplication, joint-memory normalization, pi-only final mixing, sparse support behavior, slicing, strict serialization, cache invalidation, exact retrieval, and baseline compatibility all pass for both backbones.

User attention interaction work with an efficient independent-stream implementation is approximately:

\[
O\left(D\left[L^2+C_{\rm total}L+C_{\rm total}\right]d_a\right),
\]

plus projections and pointwise transforms on all \(L+C_{\rm total}\) tokens. First-layer selection reuses computed attention scores, adding reduction/top-b/reference work rather than a second all-history relevance matmul. There is no required \(C_{\rm total}^2\) interaction or separate final pooling scan, though a naive dense backend can still pay for the masked square.

Prepared token memory scales as \(O(Pd_m+C_{\rm total}(b+d_m)+R)\) plus metadata, where P is the packed original count. Optional per-user K/V caches add \(O((P+C_{\rm total})(d+d_v))\). After cached projection, pair attention costs approximately \(O(E T_{u,i}(d+d_v))\), plus gather/union/concatenation, expert MLPs, and final pi mixing. Without cached K/V, also charge \(O(T_{u,i}d_m(d+d_v))\) pointwise projections. The shared union spans up to 32 active lists of 16 events and can cover all L<=200 events. One joint normalization spans recent, union, and active-summary tokens; reduced pair cost is not guaranteed.

Compare matched-supervision SASRec/HSTU and MoL baselines, a full-history actual-ad-token reference, memory-segment ablations, concatenated versus separately pooled evidence under matched capacity, uniform versus learned final expert mixing, and one versus multiple experts. Sweep latent/event/recent budgets after correctness. Evaluate selection coverage and actual-ad read approximation as diagnostics; do not infer ranking gains from structural similarity alone. Benchmark execution remains deferred until explicitly requested.

## References for context

- [HSTU, Appendix H](https://arxiv.org/html/2402.17152v3#A8): independent target masking and history computation reuse.
- [SIM](https://arxiv.org/abs/2006.05639): history search followed by target-aware attention.
- [RAILS / MoL](https://arxiv.org/abs/2407.15462): the learned-similarity baseline.

This is a research specification, not a novelty determination or a measured improvement.
