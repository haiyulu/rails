# LCAR: Latent Cross-Attention Retrieval

## Goal and design status

**LCAR (Latent Cross-Attention Retrieval)** combines shared latent dictionaries, multi-layer history reads, and item-expert queries that cross-attend to preserved events and latent summaries. Its backbone variants are **LCAR-HSTU** and **LCAR-SASRec**.

Approximate the multi-layer history read of a causal ranking model while preparing user memory before the target ad is known. Preserve original event details and amortize ranking-style history interaction through reusable latent tokens. Improve over SASRec/HSTU + MoL; accuracy remains an experimental question.

**Adopted design:** one dictionary per item expert supplies appended virtual-candidate tokens. First-block attention selects original events; final latent states supply summaries. At scoring, concatenate recent originals, one global-history token projected from the last valid history state, one selected-event union across all four experts, and every activated latent summary into a single unpooled memory. Add learned source embeddings for these four groups. The four ad-expert query tokens cross-attend to that shared memory. Item-dependent expert weights pi combine final expert scores only. History uses ordinary causal contextualization and cannot attend to the appended latents.

**Four-output scoring revision:** expose recent original events, the last valid contextualized history state, all latent-selected original-event lists with their preserved originals, and all final latent summaries. Project output 2 from [B,50] to one global token [B,1,32] and include it directly in the attention memory. The adopted score is the pi-weighted expert mixture after this four-source read. This supersedes the earlier proposal to use output 2 only through a separate added dot-product score. The executable configs now implement version 4 with four experts, 128 atoms each, and four memory groups. Historical version-3 benchmark checkpoints used two experts, 1,024 latents, and three groups; those results are not version-4 results.

LCAR is implemented in the independent [lcar package](../../rails/similarities/lcar/README.md), with external native-block adapters for SASRec and HSTU and an optional independent encoder wrapper. It can also be used as a standalone add-on to an unchanged host or extended through a registered third-backbone adapter. Correctness checks cover native single-candidate equivalence, first-block selection, joint attention, gradients, caches, exact retrieval, and training integration. Benchmarks remain deferred. Existing SVG files still depict the earlier preconditioning design; the text schematics below describe LCAR. See [the engineering plan](../new_similarity_scoring_plan.md).

LCARSimilarity (also exported as LCAR), create_lcar_interaction_module, and the sasrec-lcar.gin/hstu-lcar.gin configs use the new name. The superseded ExpertHistory implementation/configuration paths have been removed. Historical checkpoints are preserved but cannot be loaded as version 4.

| Model component | Computation | Purpose |
| --- | --- | --- |
| **Joint actual-ad memory interaction** | A sequence of ad-expert queries attends to one source-tagged concatenation of recent originals, a global-history token, selected originals, and activated latent summaries. | Let actual ads choose among global context, individual details, and contextual-memory tokens with one attention normalization. |
| **Shared ad dictionaries and multi-layer latent reads** | Append every dictionary atom as an independent virtual candidate. Reuse its first-block attention for event selection and its final state for summarization. | Approximate how an actual ad reads history through a causal ranker's sequence blocks. |
| **Item-expert-specific interaction** | Each item expert owns its query, dictionary, selected-event lists, and latent summaries; combine scores after interaction. | Let experts address different ad interests and user evidence. |

There are four user-side outputs and four cross-attention memory segments. Preserve individual memory tokens until joint cross-attention. The last valid history state supplies one global token; each expert query returns one updated vector after reading the entire four-segment memory.

## Architecture walkthrough with dimensions

B = users, L = valid history events, N = candidate ads. The scoring diagrams show shared-catalog candidates broadcast across users. Padding and masks handle variable event and active-latent counts.

### Step 1. Prepare all user memories before knowing the ad

~~~text
Original history X                       Shared expert dictionaries
[B,L,50]                                 [4,128,32]
     │                                         │ project 32 → 50
     │                                         ▼
     │                                   Shared latent bases
     │                                   [512,50]
     │                                         │ per-user target position
     └────────────────────┬────────────────────┘
                          ▼
                 Shared SASRec / HSTU
                Causal history + independent latents
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                      ▼
 First-block attention  Final history states   Final latent states
        │               [B,L,50]               [B,512,50]
        │                 │ last valid           │ project 50 → 32
        ▼                 ▼                      ▼
 Selected positions     User vector h_u        Individual summaries
 [B,4,128,16]           [B,50]                  [B,4,128,32]
 (output 3)             (output 2)              (output 4)

 Preserve original-event tokens separately:
 X → project 50 → 32 → original memory [B,P,32], P ≤ L
 Recent references [B,R] address this memory (output 1).
~~~

History attends causally to history. Each latent attends to all valid history and itself, independently of other latents. The same dictionary atom supplies its selected-event list and final summary. The full history continues through every block; selection only determines preserved original-event references for scoring. Original storage includes the recent events and all latent selections, deduplicated by original position.

The dictionary and projected initial bases are shared model parameters/cache entries, not B parameter copies. Latent states become user-specific after target-position initialization and history reads.

| User-side output | Shape / representation | Consumer |
| --- | --- | --- |
| 1. Recent original events | Up to R=32 tokens per user, projected to [B,R,32], or references into packed originals | Joint memory attention |
| 2. Last valid history state h_u | [B,d_h]=[B,50], after native output postprocessing; zero for an empty history | Project 50 → 32 into one global-history memory token |
| 3. Latent-selected original events | All-latent references/masks [B,4,128,16] plus packed original tokens [B,P,32] | Gather the activated-latent union during pair scoring |
| 4. Final latent summaries | [B,4,128,32] | Gather individual activated summaries during pair scoring |

All four outputs are target-independent. Output 3 is not yet an actual-ad-specific union: that union is formed only after item activation in Step 2. Output 2 requires one gather from the existing history output, with no second backbone pass.

### Step 2. Gather and concatenate individual tokens for an actual ad

~~~text
Candidate ad embedding [N,50]
                  │
           Four item experts
                  │
     ┌────────────┼─────────────────┐
     ▼            ▼                 ▼
 Query tokens   Sparse activation   Expert gates π
 [N,4,32]       Up to 16/expert     [N,4]
                  │
                  ▼
           Gather user memory
                  │
     ┌────────────┼────────────────┬──────────────────┐
     ▼            ▼                ▼                  ▼
 Recent         Global           One selected-      Individual
 originals      history token    event union        active summaries
                P_g h_u          across all experts
 [B,N,R,32]     [B,N,1,32]       [B,N,U,32]          [B,N,K,32]
     │            │                │                  │
     └────────────┴────────────────┼──────────────────┘
                                   ▼
                     CONCATENATE + SOURCE EMBEDDINGS
                             [B,N,T,32]
                         T = R + G + U + K
~~~

Here R ≤ 32 is the actual recent-event count, G=1 for a nonempty history and G=0 otherwise, K ≤ 64 is the total active-latent count across all four experts, and U ≤ min(L,1024) is the selected union size. T ≤ 297 for ML-1M. The selected union is deduplicated across all four experts by event position and ordered chronologically. It may overlap the recent segment as specified below. Keep every activated summary as its own token; no segment is pooled or alpha-weighted before concatenation. All four segment widths are 32. The global token is prepared once per user and reused across candidates, without storing N copies in the user cache.

### Step 3. Run joint cross-attention, then mix expert scores

~~~text
Expert query sequence               Concatenated memory
[B,N,4,32]                          [B,N,T,32]
       │ Q                                │ K,V
       └────────────────┬─────────────────┘
                        ▼
               ONE CROSS-ATTENTION
          One normalization over all T tokens
                        │
                        ▼
             Updated expert query tokens
                   [B,N,4,32]
                        │
              One score head per expert
                        ▼
                Expert scores [B,N,4]
                        │
                 Weighted sum using π
                        ▼
                 FINAL SCORES [B,N]
~~~

All four expert queries can read the entire shared memory, including originals and summaries originating from any expert. Pi weights final expert scores only: score(u,i) = sum_{e=1}^4 π_e(i) score_e(u,i). There is no pi attention prior. Cross-attention is the first stage that aggregates the concatenated memory into expert output vectors.

### HSTU versus HSTU + LCAR

The comparison shows the adopted four-expert, 128-atoms-per-expert design. HSTU blocks are shown as a stack, with their complete input and output sequences labeled. B denotes users, N candidate items, and L<=200 valid history events. The four-source LCAR scorer is implemented as version 4; historical version-3 benchmark results describe the earlier model.

~~~mermaid
flowchart LR
    subgraph BASE["HSTU — original retrieval"]
        direction TB
        BI["History IDs [B,L]"]
        BX["Lookup + position<br/>INPUT: [x₁, …, xL]<br/>[B,L,50]"]
        BH["HSTU stack · 8 blocks"]
        BO["OUTPUT: [h₁, …, hL]<br/>[B,L,50]"]
        BU["Last valid state + normalization<br/>User vector [B,50]"]
        BC["Candidate IDs [N]<br/>Lookup + normalization<br/>Item vectors [N,50]"]
        BS["Dot-product scores<br/>[B,N]"]

        BI --> BX --> BH --> BO --> BU
        BU --> BS
        BC --> BS
    end

    subgraph LCAR["HSTU + LCAR — 4 experts × 128 atoms"]
        direction TB
        LI["History IDs [B,L]"]
        LX["Same lookup + position<br/>Original events [B,L,50]"]

        LD["Four shared dictionaries<br/>[4,128,32]"]
        LB["Project 32 → 50<br/>Shared latent bases [512,50]"]

        LIN["HSTU INPUT SEQUENCE<br/>[x₁, …, xL | z₁, …, z₅₁₂]<br/>[B,L+512,50]"]
        LH["HSTU stack · 8 blocks<br/>Same native block equations"]
        LO["HSTU OUTPUT SEQUENCE<br/>[h₁, …, hL | z₁ᴰ, …, z₅₁₂ᴰ]<br/>[B,L+512,50]"]

        REC["① Recent originals<br/>Project 50 → 32<br/>[B,R,32], R ≤ 32"]
        GLOBAL["② Last valid history state<br/>Normalize [B,50]<br/>Project → global token [B,1,32]"]
        SELECT["③ Selected-original cache<br/>References [B,4,128,16]<br/>Preserved originals at width 32"]
        SUMMARY["④ All latent summaries<br/>Normalize + project 50 → 32<br/>[B,4,128,32]"]

        ITEM["Candidate IDs [N]<br/>Lookup + normalization<br/>Item vectors [N,50]"]
        Q["Four item-expert queries<br/>[N,4,32]"]
        ACTIVE["Sparse activation<br/>≤16 of 128 atoms per expert<br/>≤64 active latents total"]
        GATHER["Gather activated evidence<br/>Selected union [B,N,U,32], U ≤ L<br/>Active summaries [B,N,K,32], K ≤ 64"]

        MEMORY["SCORING MEMORY SEQUENCE<br/>[recent R | global 1 | selected U | summaries K]<br/>[B,N,T,32] + four source embeddings<br/>T ≤ 297"]
        CA["One joint cross-attention<br/>Four queries read the same memory<br/>Output [B,N,4,32]"]
        HEADS["Four expert scoring heads<br/>Expert scores [B,N,4]"]
        PI["Item-only expert weights π<br/>[N,4]"]
        SCORE["π-weighted expert mixture<br/>Final scores [B,N]"]

        LI --> LX --> LIN
        LD --> LB --> LIN
        LIN --> LH --> LO

        LX --> REC
        LX --> SELECT
        LH -. "First-block selections" .-> SELECT
        LO --> GLOBAL
        LO --> SUMMARY

        ITEM --> Q --> ACTIVE
        ITEM --> PI
        ACTIVE --> GATHER
        SELECT --> GATHER
        SUMMARY --> GATHER

        REC --> MEMORY
        GLOBAL --> MEMORY
        GATHER --> MEMORY
        MEMORY -->|"Keys / values"| CA
        Q -->|"Queries"| CA
        CA --> HEADS
        Q --> HEADS
        HEADS --> SCORE
        PI --> SCORE
    end
~~~

For nonempty histories, T=R+1+U+K<=32+1+200+64=297. All four expert queries share one recent segment, one global token, and one selected-event union. Dictionary parameters are shared across users; the latent output states are user-specific. The smaller dictionaries reduce latent preparation/storage budgets, while unchanged activation budgets preserve the maximum scoring-memory length.

## 1. Notation and computation boundaries

Let \(X=(x_1,\ldots,x_L)\) be original event embeddings captured after input preprocessing but before the first sequence block. Preserve positions, IDs, masks, and available time/action payloads. Raw/original events mean these pre-block representations, not contextualized history outputs.

| Symbol | Meaning | Initial setting |
| --- | --- | --- |
| \(L\) | Valid events in a causal prefix | Existing limit; ML-1M at most 200 |
| \(D,d_h\) | Sequence blocks and hidden width | SASRec: 2 blocks; HSTU: 8; width 50 |
| \(E\) | Item experts | 4 |
| \(M\) | Dictionary atoms per expert | 128 |
| \(C_{\rm total}=EM\) | Appended latent tokens per user prefix | 512 |
| \(K_{\rm act}\) | Maximum active latents per ad/expert | 16 |
| \(E K_{\rm act}\) | Maximum active summaries / selected lists per ad | 64 |
| \(b\) | Original events selected per latent | At most 16 |
| \(R\) | Recent original events | At most 32 |
| \(d_m,d,d_v\) | Memory token, query/key, and value widths | 32 |
| \(h_1\) | First sequence block's attention heads | Existing backbone setting; SASRec 1, HSTU 2 |

The adopted four-expert setting prepares all 512 latent tokens before any target is known. Each expert retains 128 atoms and activates up to 16; each atom retains up to 16 original events. This reduces the prepared latent count by 75% relative to the previous four-expert, 512-atoms-per-expert plan. The shared native history stream is still computed once per layer. With latent_chunk_size=128, all 512 latents require four chunks per layer. The maximum active-summary count remains 64 and the ML-1M scoring-memory bound remains 297 tokens, because activation and event-selection budgets have not changed. These are configured budgets, not measured runtime improvements.

The four item experts and four memory-source types are different axes. Increasing experts does not create additional global-history tokens or recent segments: there remains one global token and one recent segment per user, and one selected-event union shared by all four queries.

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

### 3.5 Expose the last valid contextualized history state (output 2)

Apply the native output postprocessing to the final history stream and gather its last valid event, not the last padded slot and not a latent token:

\[
\widehat H^{(D)}_u=\operatorname{OutputPostproc}(H^{(D)}_u),\qquad
h_u=\begin{cases}
\widehat H^{(D)}_{u,L_u}, & L_u>0,\\
0\in\mathbb R^{d_h}, & L_u=0.
\end{cases}
\]

This formula uses one-based event positions; implementation indexing is L_u-1. Gather from already-postprocessed history states without applying output postprocessing twice. The result has shape [B,50] for both ML-1M backbones.

At fixed weights, inputs, masks, and position/time policies, h_u matches the native retrieval model's last-history output as if no latents were present. Dictionary values/order do not affect it, because history cannot read latents. Training through latent reads can still update the shared backbone weights. Preserve h_u's autograd path, zero empty-history outputs, and slice it alongside the rest of the user state.

## 4. One concatenated memory and joint expert-query cross-attention

### 4.1 Gather four unpooled memory segments

Gather active atom identities from all four experts:

\[
\mathcal A(i)=\{(e,j):j\in\mathcal A(i,e)\},\qquad
K_i=|\mathcal A(i)|\le E K_{\rm act}=64.
\]

Build one event union across every activated latent, not a different union per query expert:

\[
S_{u,i}=\operatorname{Unique}
\left(\bigcup_{(e,j)\in\mathcal A(i)}I_{u,e,j}\right),
\qquad U_{u,i}=|S_{u,i}|\le\min(L,E K_{\rm act}b).
\]

Deduplicate by original event position across all four experts' active lists. Sort selected original events chronologically. Each summary retains its expert/atom identity; do not deduplicate distinct summaries merely because their values happen to match.

Gather:

\[
g_u=P_g h_u\in\mathbb R^{32},\qquad M^{\rm global}_{u,i}=[g_u].
\]

P_g is a learned affine projection from native width 50 to memory width 32. It runs once per user. The global slot is present only for nonempty histories; its mask must exclude empty-history projection biases and source embeddings.

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
[M^{\rm recent}_{u,i};M^{\rm global}_{u,i};M^{\rm selected}_{u,i};M^{\rm latent}_{u,i}]}
\in\mathbb R^{T_{u,i}\times32},
\]

\[
T_{u,i}=R_u+G_u+U_{u,i}+K_i,\qquad
G_u=\mathbf1[L_u>0],\qquad R_u=\min(R,L_u).
\]

Add four learned source embeddings, one each for recent, global, selected, and summary tokens, after their width-32 projections and before scorer K/V projection:

\[
\widetilde m_t=m_t+\tau_{\operatorname{source}(t)},\qquad
\tau\in\mathbb R^{4\times32}.
\]

The same original event in the recent and selected segments shares its base projection but receives the corresponding source embedding in each occurrence. Source embeddings do not pool tokens, select experts, or add a separate attention normalization.

Pad candidate-specific lengths with an explicit mask. At current budgets, U<=min(L,1024), K<=64, and T<=297 for L<=200. Padding is a batching representation, not evidence; a packed implementation with equivalent boundaries/masks represents the same attention.

Preserve literal concatenation: the selected union is deduplicated internally across experts, but may overlap the recent segment. An event appearing in both contributes two memory positions and therefore affects joint attention multiplicity. This behavior must be explicit and tested; cross-segment removal is not silently applied. Recent/selected events remain accessible in user-side latent reads as well.

### 4.2 Use ad experts as the query sequence

Stack the actual expert queries:

\[
Q_i=[q_{i,1};q_{i,2};q_{i,3};q_{i,4}]\in\mathbb R^{4\times32}.
\]

Project memory and queries for a single cross-attention operation:

\[
\widehat Q_i=Q_iW_Q,\qquad
K_{u,i}=\widetilde{\mathcal T}_{u,i}W_K,\qquad
V_{u,i}=\widetilde{\mathcal T}_{u,i}W_V,
\]

\[
A_{u,i}
=\operatorname{softmax}_{\rm memory}
\left(\frac{\widehat Q_iK_{u,i}^{\top}}{\sqrt d}
+\mathrm{padding\ mask}\right),
\qquad
O_{u,i}=A_{u,i}V_{u,i}.
\]

Use one attention head initially. There is one softmax per expert query over the entire concatenated memory, not separate normalizations for each segment. All four query experts may read originals and summaries selected by any expert. No query-to-query self-attention is added.

The default attention has no pi prior, alpha prior, or coefficient-based token scaling. The actual query-memory compatibility learns the read weights. Cross-attention aggregates memory only at this stage, returning one output token per expert query.

| Tensor | Shape |
| --- | --- |
| Concatenated memory | [B,N,T_max,32] |
| Expert query sequence | [B,N,4,32] |
| Attention weights, one head | [B,N,4,T_max] |
| Updated expert query outputs | [B,N,4,32] |

The original-event projection, global-history projection, and joint K/V projections are learned. Since K/V projection is pointwise, user preparation can cache base original, global, and summary K/V. Apply each projected source offset exactly once when assembling typed memory; affine projection biases must not be added twice. This is equivalent to directly projecting the source-tagged concatenation.

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
\boxed{s(u,i)=\sum_{e=1}^{4}\pi_e(i)s_e(u,i)}.
\]

Pi determines expert contribution to the final logit, not attention-memory preference. Recommendation output remains [B,N].

### 4.4 Use all four outputs through the joint memory read

Output 2 reaches the scoring function through g_u=P_g h_u in the same attention memory as outputs 1, 3, and 4. The global token supplies the native retrieval-style contextual representation; recent originals preserve fresh details; selected originals expose relevant older details; active summaries supply prototype-specific contextual reads. All four ad-query experts may attend to every source.

Keep the 96 -> 64 -> 1 expert heads, pi-only final mixing, and sampled-softmax temperature 1. Do not concatenate h_u into a larger score head or add a separate dot-product term in this adopted design. Start with one scorer attention head and the four learned source embeddings, without additional segment gates or source-count priors. Measure per-source attention mass, especially use of the single global token, before considering further gating changes.

~~~text
Output 1: recent originals             [B,R,32]
Output 2: last history state [B,50] → P_g → [B,1,32]
Output 3: active selected-event union  [B,N,U,32]
Output 4: active latent summaries      [B,N,K,32]
                            │
             Concatenate + source embeddings
                     [B,N,T,32]
                            │ actual-ad queries [N,4,32]
                            ▼
                 One joint cross-attention
                            │ expert heads + pi
                            ▼
                       Logits [B,N]
~~~

~~~mermaid
flowchart TD
    X[Original history X] --> Raw[Original tokens and recent list]
    X --> Blocks[Shared causal history with all independent latent tokens]
    C[One dictionary per expert] --> Init[Initialize 512 virtual candidates]
    Init --> Blocks
    Blocks --> First[First-block latent-to-history coefficients]
    First --> Lists[Top 16 original positions per latent]
    Blocks --> Last[Output 2: last valid postprocessed history state B by 50]
    Blocks --> Summaries[Final latent states projected to width 32]
    Ad[Ad embedding] --> Queries[Four expert query tokens]
    Ad --> Pi[Item-only expert weights pi]
    Queries --> Coding[Sparse active atoms for each expert]
    Coding --> Union[One selected-event union across all four experts]
    Lists --> Union
    Union --> Selected[Gather original tokens]
    Raw --> Selected
    Raw --> Recent[Recent original tokens]
    Coding --> Active[Gather individual active summaries]
    Summaries --> Active
    Last --> Global[Project global history token to width 32]
    Recent --> Concat[Concatenate all four token segments]
    Global --> Concat
    Selected --> Concat
    Active --> Concat
    Concat --> Types[Add recent global selected summary source embeddings]
    Types --> CA[One joint cross-attention memory]
    Queries --> CA
    CA --> Outputs[Four updated expert query tokens]
    Outputs --> Scores[One score per expert]
    Queries --> Scores
    Scores --> Mix[pi-weighted sum of expert scores]
    Pi --> Mix
    Mix --> Logits[Logits B by N]
~~~

## 5. Training, integration, and versioning

Keep one canonical dictionary per expert registered with the scorer. Token initialization uses its atoms without copying or detaching them; all histories and latent tokens pass through the same sequence-block parameters. The first-block attention coefficients simultaneously support the differentiable latent read and hard event selection. No selector-only projection or separate selection loss is needed for this initial design.

The bridge must capture X, run history plus latent streams, expose first-block selection references, the last valid postprocessed history state h_u, and final latent outputs, then pack original memory tokens, optional projected K/V, and all latent summaries. The integrated encode_with_history API already returns (current_user, UserMemory); current_user is output 2. Prepared memory now contains that vector, its projected global token/mask, and optional global K/V, so standalone preparation and chunked scoring use the same four-source memory. Preparing raw memory entirely before the first block is no longer sufficient because selection comes from that block.

Use full-available-history next-item supervision (`full_prefix`) by default. The dataset-provided next item is the label; mask reserved output slots and target timestamps. Retain seeded `sampled_prefix` as an explicit study option, and `all_positions` for the original baselines. Build every user memory from its available prefix, reuse it for positive/negative scores, aggregate recommendation and sparse reconstruction/L1 losses once per logical batch, and preserve the dictionary norm projection after updates. Maintain gradient paths through latent initialization, every latent block, first-layer Q/K, final summary projections, actual query/memory projections, and final expert scoring/gates. Sparse support is hard routing: alpha values have no differentiable recommendation-read path in this default; the sparse auxiliary objective still trains coefficients and coding.

Use exact prepared-state chunked catalog scoring. Separate cache dependencies: ordinary history K/V and h_u depend on history/backbone/input processing; the global token also depends on P_g; final latent memories also depend on the dictionary, token projections, and summary projections. Typed K/V depend on source embeddings and joint projections. Dictionary-only changes should not change the history stream or global token at fixed native/global weights. Use format version 4 for four-source scoring; version 3 denotes historical three-source results. Strict loading rejects earlier formats; no automatic warm-start mapping is supplied. Preserve DotProduct/MoL behavior.

## 6. Validation, cost, and experiments

Before benchmarks, require:

1. Batched independent latents match separate single-latent passes with matched logical position/time and normalization.
2. History outputs remain unchanged when only latent values/order change; permuting latents permutes their outputs and lists.
3. First-block selection matches its actual history-attention coefficients and gathers original X-derived K/V.
4. Every latent can access all valid history events in every block; no selected-event pruning or cross-latent reads.
5. Isolated final-summary losses produce gradients through token projections, dictionary atoms, and first-block Q/K without sparse auxiliary loss.
6. User-only invariance, causal prefixes, empty/short histories, overlap/deduplication, joint-memory normalization, pi-only final mixing, sparse support behavior, slicing, strict serialization, cache invalidation, exact retrieval, and baseline compatibility all pass for both backbones.
7. Output 2 equals the last valid native postprocessed history state under matched weights/masks/position-time policies; dictionary perturbations leave it unchanged. Empty histories mask the global slot, user slicing preserves alignment, and a loss through the global token reaches P_g and the native backbone. Four-source attention matches a direct source-tagged concatenation under candidate chunking; source offsets appear exactly once.

Exposing output 2 adds O(B*d_h) retained values, a last-valid gather, and one 50 -> 32 projection per user, without another encoder pass. Cache one global token and its optional K/V per user, plus four model-level source embeddings/projected offsets. Pair attention reads one additional valid token for nonempty histories; no separate dot-product branch is added.

User attention interaction work with an efficient independent-stream implementation is approximately:

\[
O\left(D\left[L^2+C_{\rm total}L+C_{\rm total}\right]d_a\right),
\]

plus projections and pointwise transforms on all \(L+C_{\rm total}\) tokens. First-layer selection reuses computed attention scores, adding reduction/top-b/reference work rather than a second all-history relevance matmul. There is no required \(C_{\rm total}^2\) interaction or separate final pooling scan, though a naive dense backend can still pay for the masked square.

Prepared token memory scales as \(O((P+1)d_m+d_h+C_{\rm total}(b+d_m)+R)\) plus metadata, where P is the packed original count. Optional per-user K/V caches add \(O((P+1+C_{\rm total})(d+d_v))\). After cached projection, pair attention costs approximately \(O(E T_{u,i}(d+d_v))\), plus gather/union/concatenation, expert MLPs, and final pi mixing. Without cached K/V, also charge \(O(T_{u,i}d_m(d+d_v))\) pointwise projections. The shared union spans up to 64 active lists of 16 events and can cover all L<=200 events. One joint normalization spans recent, global, union, and active-summary tokens, with T<=297; reduced pair cost is not guaranteed.

Compare matched-supervision SASRec/HSTU and MoL baselines, a full-history actual-ad-token reference, memory-segment ablations, concatenated versus separately pooled evidence under matched capacity, uniform versus learned final expert mixing, and one versus multiple experts. Sweep latent/event/recent budgets after correctness. Evaluate selection coverage and actual-ad read approximation as diagnostics; do not infer ranking gains from structural similarity alone. Benchmark execution remains deferred until explicitly requested.

## References for context

- [HSTU, Appendix H](https://arxiv.org/html/2402.17152v3#A8): independent target masking and history computation reuse.
- [SIM](https://arxiv.org/abs/2006.05639): history search followed by target-aware attention.
- [RAILS / MoL](https://arxiv.org/abs/2407.15462): the learned-similarity baseline.

This is a research specification, not a novelty determination or a measured improvement.
