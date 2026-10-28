# Inference FLOPs comparison: SASRec, SASRec+MoL, HSTU, HSTU+MoL

## Goal and comparison scope

Compare the four trained ML-1M models using the existing uv environment and checkpoints from `playground/benchmark_runs/20261026-201653/`. Use each model's `best.pt` and its derived Gin configuration; no retraining or architecture changes are required. Dense FLOPs depend on architecture and tensor shapes, not the selected checkpoint's accuracy.

Report three requested measurements:

1. **Entire-model inference:** raw history/user/candidate IDs to a matrix of candidate scores, with clearly separated cached-item and uncached-item totals.
2. **User model:** every computation depending only on the user/history, including output projections and query-only gate computation.
3. **Strict user–item scoring:** only computations combining the prepared user and item representations, or operating on quantities already dependent on both.

Track candidate-only preprocessing as a fourth accounting bucket so the entire-model totals reconcile. It is never part of strict user–item scoring.

Exclude data loading, host/device transfers, checkpoint loading, compilation, ranking/top-k, seen-item filtering, index construction, and approximate retrieval. These are outside the neural forward pass being compared. Do not present FLOPs as a latency measurement.

## 1. Computation boundaries

Let U denote user-only computation, I candidate-only preparation, and S strict joint scoring. Classify operations by their input dependencies, regardless of which Python class currently contains them.

| Operation | Bucket | Frequency |
|---|---|---|
| History item lookup, embedding scaling, positional embedding addition | U | Once per user history |
| SASRec/HSTU backbone, relative/time-bias arithmetic, output normalization, latest-position selection | U | Once per user history |
| MoL query projection, reshape, user-ID component lookup/concatenation, component normalization | U | Once per user |
| Query-only gate MLP `f_q(h)` | U | Once per user |
| Candidate embedding lookup and baseline candidate normalization | I | Once per candidate |
| MoL candidate projection and component normalization | I | Once per candidate |
| Item-only gate MLP `f_i(e)` | I | Once per candidate |
| Baseline query–candidate dot product | S | Once per user–candidate pair |
| MoL component dot products and temperature scaling | S | Once per user–candidate pair |
| Joint gate MLP `f_qi(z)` applied to the pair's logits | S | Once per user–candidate pair |
| Combining prepared query/item gate outputs, gate activation, softmax, weight renormalization, weighted-logit reduction | S | Once per user–candidate pair |

Lookups, concatenations, reshapes, and selections can have zero arithmetic FLOPs while still belonging to U or I. Their memory costs must not be mistaken for FLOPs.

For the ML-1M MoL models:

```text
User preparation U:
  history IDs → positional preprocessing → SASRec/HSTU → h [B, 50]
  h → SwiGLU(50 → 512) → Linear(512 → 448) → [B, 7, 64]
  append user-ID embedding [B, 1, 64] → normalize → Q [B, 8, 64]
  h → query-only gate MLP (50 → 128 → 32) → a [B, 32]

Candidate preparation I, reusable across users:
  item IDs → e [N, 50]
  e → Linear(50 → 256) → reshape/normalize → V [N, 4, 64]
  e → item-only gate MLP (50 → 128 → 32) → b [N, 32]

Strict joint scoring S:
  Q, V → 32 component-pair dot products per pair → z [B, N, 32]
  z / temperature → joint gate MLP (32 → 128 → 32)
  combine with a[:, None, :] * b[None, :, :]
  → SiLU → softmax → implementation's renormalization
  → weighted sum of the 32 scaled logits → scores [B, N]
```

In particular, neither the 50→7×64 query projection nor `f_q(h)` belongs to S. Neither the item projection nor `f_i(e)` belongs to S. The gate does not consume flattened 512-dimensional query components as its query-only input: it consumes the original 50-dimensional h.

### Gate equations and counting boundary

Prepare the following outside S:

\[
Q=F_q(h,\mathrm{userID}),\quad a=f_q(h),\qquad
V_i=F_i(e_i),\quad b_i=f_i(e_i).
\]

Only the following computation belongs to S:

\[
z_{i,(r,t)}=\langle Q_r,V_{i,t}\rangle/\tau,
\quad c_i=f_{qi}(z_i),
\]
\[
g_i=\operatorname{SiLU}(a\odot b_i+c_i),
\quad p_i=\operatorname{softmax}(g_i),
\]
\[
\widetilde p_i=p_i/\max\left(\sum_\ell p_{i,\ell},\epsilon\right),
\qquad s(h,i)=\sum_\ell\widetilde p_{i,\ell}z_{i,\ell}.
\]

The extra renormalization is included because the current combiner's configured dropout rate is positive: its renormalization branch executes even in evaluation mode, where dropout itself is an identity. Training-only MI/UID regularization losses are excluded.

## 2. Whole-model totals and cache assumptions

Use a shared catalog of N candidates for B users. Prepare each candidate once, rather than duplicating candidate-only work B times.

\[
F_{\mathrm{cached}}(B,L,N)=F_U(B,L)+F_S(B,N),
\]
\[
F_{\mathrm{uncached}}(B,L,N)=F_U(B,L)+F_I(N)+F_S(B,N).
\]

- **Primary serving total:** cached-item inference. Candidate embeddings, component vectors, and item-only gate outputs are already materialized. User preparation is still charged once for every request.
- **Secondary entire-model total:** uncached-item inference, including candidate-only preparation once for the requested shared candidate set. Also report the full-catalog preparation cost separately as an offline cost.
- For B=1, report FLOPs per request. For larger B, report both total batch FLOPs and FLOPs per user by dividing the whole batch total by B. Candidate preparation is amortized across B users only in this shared-catalog batch scenario.
- Report S both as total scoring FLOPs and as FLOPs per pair, `F_S / (B*N)`. Do not allocate U or I into this per-pair value.
- Use a full history forward pass for all four models. HSTU's incremental history-state cache is disabled; item-side materialization is a distinct cache assumption.

The requested headline table uses B=1, valid history length L=200, and the full catalog. Include cached and uncached totals so “entire model” is not ambiguous.

## 3. Workloads and FLOP convention

### Fixed workloads

- Use the existing Python 3.12 uv environment on GPU 0. Evaluate all models in FP32 under `eval()` and inference mode; no backward pass, dropout sampling, or training losses.
- Primary workload: B=1, L=200, N equal to the actual dataset catalog size. Derive N from `get_reco_dataset(...).all_item_ids`; the current prepared ML-1M catalog contains **3,883** items. Do not substitute the embedding table's maximum ID or a paper catalog count.
- Run a controlled sweep with B in `{1, 32}`, L in `{10, 50, 100, 200}`, and N in `{1, 100, 1000, full_catalog}`. Use identical deterministic input fixtures across the four models, seed 42, valid catalog IDs/user IDs, and monotonically increasing timestamps.
- Preserve the deployed input width **T=211**, corresponding to the current 200-history/11-extra-position configuration. Set `past_lengths=L`, fill only the valid prefix, and zero-pad the rest. Report L and T separately.
- Count the padding work actually performed: SASRec operates on padded tensors, while HSTU has jagged pointwise work but still uses padded attention matrices in this implementation. Do not assume jagged tensors or a causal mask eliminate dense attention arithmetic. Do not halve dense attention FLOPs just because attention is causal.
- Prepare the whole candidate set at once and score it without chunking or approximate selection. This isolates FLOPs from cache-miss/chunking/retrieval-policy effects.

### Counting convention

Report logical forward arithmetic FLOPs, not hardware instruction counts, Tensor Core throughput, or profiler kernel durations.

- One multiply–accumulate is **2 FLOPs**. A matrix product `(m,k) @ (k,n)` costs `2*m*k*n`. A D-dimensional dot product costs `2*D` by this convention. Apply the same convention to batched products, einsums, attention contractions, and pointwise Conv1d projections.
- Count floating-point addition, subtraction, multiplication, and division as 1 FLOP each. Include biases, residual additions, temperature scaling, attention-mask multiplication, and normalization arithmetic.
- Report **matrix/convolution FLOPs**, **other arithmetic FLOPs**, and their sum. Separately report counts of special-function evaluations such as exp, sigmoid, GELU, sqrt, rsqrt, and log; do not silently omit them or invent hardware-equivalent FLOP prices.
- Use explicit reference decompositions for fused normalization/activation operations. Stable softmax over E values costs `3*E-1` arithmetic FLOPs plus E exp evaluations, with maximum comparisons reported outside FLOPs. The additional sum/divide renormalization costs `2*E-1` arithmetic FLOPs. SiLU costs one multiplication and one sigmoid evaluation per value. L2 normalization over D values costs `3*D-1` arithmetic FLOPs plus one sqrt evaluation, excluding clamp comparisons. For non-affine LayerNorm, use a two-pass centered-variance decomposition: `5*D+1` arithmetic FLOPs and one rsqrt per row; count any affine scale/bias separately if present.
- Embedding lookup, gather/scatter, reshape, transpose, concatenation, padding stores, casts, comparisons, and integer indexing/hash/cumulative-sum operations contribute no arithmetic FLOPs. List them as non-arithmetic operations where useful for coverage.
- Count broadcast arithmetic at its output shape. Broadcasting a cached gate vector is free; multiplying query and item gate vectors produces B*N*32 multiplications.
- State that canonical logical decompositions may differ from arithmetic performed by a particular fused GPU kernel. Report all special-function counts alongside the arithmetic totals.

### Analytic scoring sanity checks

For the supplied ML-1M configurations:

- SASRec and HSTU baseline scoring both require **100 matrix FLOPs per pair**: `2*50`.
- Both MoL models have the same strict scoring dimensions, regardless of backbone or learned weights.
- MoL component contractions require **4,096 matrix FLOPs per pair**: `2*8*4*64`.
- The joint-only gate MLP requires **16,384 matrix FLOPs per pair**: `2*(32*128 + 128*32)`.
- Thus, strict MoL scoring has **20,480 matrix FLOPs per pair**, before bias, activation, scaling, mixing, softmax, renormalization, and final reduction arithmetic. The expected arithmetic total under the convention above is **21,117 FLOPs per pair**, plus **160 sigmoid** and **32 exp** evaluations. Verify this against the explicit scorer; if the implementation changes, update the derivation rather than forcing the measured count.
- The corresponding **matrix-only** MoL/baseline scoring ratio is **204.8×**. This is not the whole-model ratio and does not imply the same latency ratio.

These are formula checks, not completed benchmark measurements.

## 4. Implementation approach

Add a benchmark runner under `playground/` with a CLI such as:

```bash
.venv/bin/python playground/benchmark_inference_flops.py \
  --checkpoint-root playground/benchmark_runs/20261026-201653 \
  --output-dir playground/inference_flops_runs/<run-id>
```

The runner should:

1. Load each existing configuration/checkpoint through shared model-construction helpers. Filter training-only benchmark Gin bindings as the checkpoint evaluator already does. Keep production model code and checkpoint parameters unchanged.
2. Implement benchmark-only `prepare_user`, `prepare_items`, and `score_prepared` adapters. Reuse the original projection/gating modules and weights. For MoL, cache both component embeddings and the query-only/item-only gate outputs at their respective boundaries. Simply calling `MoLSimilarity.forward` inside S is incorrect because that would recompute and charge U/I work.
3. Implement an explicit eager score reference matching the current gating/combiner formulas. Avoid compiled/fused wrappers hiding operations from FLOP accounting; separately compare its scores with the existing original inference path.
4. Instrument U, I, and S separately, then instrument their composed forward pass. Use operator/shape tracing and analytical contraction formulas as the authoritative accounting ledger. Use PyTorch's installed FLOP counter/profiler as a cross-check for supported contractions, not as the sole total.
5. Add explicit rules for attention, pointwise Conv1d, normalization, and custom FBGEMM operations. FBGEMM jagged layout conversions are data movement, not matrix FLOPs. Decompose or explicitly account for fused attention/normalization rather than silently counting unsupported operations as zero.
6. Record every floating-point operation in a per-stage ledger, including its input/output shapes, FLOP rule, arithmetic count, special-function counts, and module attribution. Prevent double counting between high-level handlers and their traced sub-operations. Fail on an unclassified floating-point operation.
7. Preserve the actual backbone inference behavior, including normalization over the sequence before latest-position selection and any intermediate work executed by the existing attention implementation. Any optimized alternative belongs in a separate future experiment.
8. Calculate cached/uncached whole-model totals, batch/per-user/per-pair values, and relative ratios versus SASRec. Record checkpoint/config/source hashes, environment versions, dimensions, input fixtures, and cache assumptions in a manifest.

No training interfaces, saved checkpoint format, or production scoring API need to change. The split representation/adapters are benchmark-local.

## 5. Validation and acceptance criteria

- **Numerical equivalence:** for all four models and all workload shapes, composed prepared-stage scores must match the original full forward inference scores with `rtol=1e-4, atol=1e-5`. Both paths must use the same normalization and candidate embeddings. Verify cache preparation for a candidate subset and the full catalog.
- **Boundary isolation:** the S adapter accepts prepared representations only. Hooks must show zero calls to the history encoder, query/item component projection modules, and query-only/item-only gate MLPs during S. Verify those modules execute once at the appropriate U/I preparation boundary, not once per candidate pair.
- **Reconciliation:** a separately counted uncached composed forward must equal U+I+S under the same logical rules; cached forward must equal U+S. Matrix FLOP totals must reconcile exactly. Any profiler discrepancy must be explained in the coverage report.
- **Closed-form checks:** baseline score matrix FLOPs must equal `100*B*N`; MoL score matrix FLOPs must equal `20,480*B*N`. Check the full arithmetic/special-function formula above, including inference renormalization and the final weighted sum.
- **Scaling:** changing N must not change U. I must scale with N, independent of B for the shared catalog. S must scale with B*N, independent of L. Changing L must not change S. Do not require user FLOPs to scale linearly with L when padded attention width T is fixed.
- **Cross-model invariants:** SASRec and HSTU strict baseline scoring costs must match; SASRec+MoL and HSTU+MoL strict scoring costs must match. Their user-model and whole-model totals may differ.
- **Cache equivalence:** materialized candidate projections and item-only gate outputs must give the same scores as on-demand candidate preparation. User-only gate outputs must be reused across candidates without changing the result.
- **Completeness:** every measured case must have a fully classified arithmetic ledger and explicit special-function counts. An unsupported profiler operation must not disappear from the total.

## Deliverables

Write results into a new inference-FLOPs run directory, separate from accuracy-benchmark outputs:

- `comparison.md` and `comparison.csv`: four-model headline comparison at B=1/L=200/full catalog, including user FLOPs, candidate preparation FLOPs, strict pair FLOPs, total scoring FLOPs, cached whole-model FLOPs, uncached whole-model FLOPs, special-function counts, and ratios versus SASRec.
- `sweep.csv`: the B/L/N/T sweep with batch and per-user values.
- `operation_ledger.json`: per-operation counts, shapes, dependency bucket, and counting rules.
- `validation.json` and `coverage.md`: score-equivalence errors, boundary-hook evidence, formula/reconciliation checks, and profiler coverage.
- `manifest.json`: the exact models, checkpoint/config/source hashes, environment, inputs, and counting/cache conventions.

Keep this experiment focused on inference compute accounting. Latency, memory bandwidth, approximate retrieval quality, incremental history caching, architecture resizing, and additional datasets are outside this plan.
