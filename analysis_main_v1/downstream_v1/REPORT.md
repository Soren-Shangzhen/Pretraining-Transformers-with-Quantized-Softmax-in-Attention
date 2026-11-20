# Downstream evaluation — 124M @ 2.5B suite

Executing `downstream_eval_instructions_124M_2p5B_v2.md` (2026-09-18). Objects: the 15 final checkpoints
(2500.1M tokens) of the 124M @ 2.5B long-horizon suite. Purpose: answer whether reporting NLL alone is
sufficient, not to re-rank conditions.

Output root: `analysis_main_v1/downstream_v1/`. Machine: gpu-host-A (RTX 5090), venv `<CLOUD_ENV>`.

---

## 1. Pre-declarations (written before any downstream result was produced)

Written 2026-09-20, after the §8.0 gate passed and before any downstream corpus, probe or benchmark was run.

### 1.1 Effect-size tiers (from the 2.5B endpoint ΔNLL vs Exact, report Table 23)

| Tier | Conditions | ΔNLL | Expectation |
|---|---|---|---|
| Pathological | MinMax-Weight K4 (+0.890), MinMax-Weight K16 (+0.447), LERP K1 (+0.346), MinMax-Weight K32 (+0.160), QRM-Weight K2 (+0.130) | 0.13–0.89 | should be clearly detectable |
| Intermediate | MinMax-Prob K4 (+0.061), QRM-Weight K4 (+0.043), MinMax-Weight K64 (+0.038), QRM-Prob K4 (+0.019) | 0.02–0.06 | possibly detectable, task dependent |
| Good | LERP K2 (+0.0097), LERP K4 (+0.005), QRM-Weight K16 (+0.004), LERP K32 (+0.002), LERP K16 (+0.0003) | <0.01 | **expected not detectable; that is the expected result** |

Plus Exact, 15 runs total.

### 1.2 Predictions

- **P1.** The pathological tier shows detectable degradation relative to Exact on layers 2 and 3.
- **P2.** The good tier shows no detectable difference from Exact on all three layers; on metrics with a
  declared equivalence margin, the paired CI lies entirely within ±0.01 nats.
- **P3.** Pathological conditions degrade on retrieval-type probes *more* than on ordinary LM NLL, i.e.
  `A = ΔNLL_probe − ΔNLL_LM > 0`. This is the one falsifiable mechanistic prediction.
- **P4.** MinMax-Weight retrieval degradation grows with retrieval distance: the slope of paired target-NLL
  degradation against `log2(distance)` is declared positive in advance.

- **P5** (added 2026-09-20, before any probe data was generated; motivated by the layer-1 exact-forward
  reversal in §4.3). If models learn a score geometry matched to their own operator, then the exact-forward
  reversal should be larger on retrieval probes than on ordinary LM NLL, because precise retrieval depends most
  on within-row resolution. Formally, with `R(c, task) = ΔNLL_exact_forward(c, task) − ΔNLL_native_forward(c, task)`,
  the prediction is `R(c, probe A/B) > R(c, LM)`.

If P3, P4 or P5 fail, they are reported as failed ("consistent with broad quality degradation rather than
selective retrieval failure"), without re-parameterising the metric.

### 1.3 Practical equivalence margin

- **LM NLL-type metrics (layer 1, probe target-NLL): ±0.01 nats.** This is the resolution already declared in
  the main results (a contrast counts as separated only when its CI lower bound exceeds +0.01 nats).
- **Accuracy / margin metrics: no equivalence margin is declared.** Those results may only be reported as "no
  detectable difference", never as equivalence.

Decision rule: CI entirely within ±0.01 → "within the pre-declared practical equivalence margin"; CI covering 0
but exceeding ±0.01 → "no detectable difference" only; CI excluding 0 → "detectable difference" with point
estimate and CI.

### 1.4 Statistical rules

Paired per example on identical examples in identical order; bootstrap on the differences; 4000 resamples; 95%
percentile intervals. Bootstrap kind: examples (probes, multiple choice), circular moving-block with block
length 16 (continuous token streams), document-cluster (C4). Detectability is decided solely by the paired CI
excluding 0. Split-half noise scale is descriptive only, never a second significance gate. The 14
condition-vs-Exact comparisons within a layer form one confirmatory family with Benjamini–Hochberg q-values;
sub-group analyses (BLiMP per paradigm) are exploratory.

### 1.5 Benchmark selection — superseded on 2026-09-20

The instructions' original rule excluded HellaSwag, Winogrande, ARC-Challenge and MMLU a priori because "at
124M they sit near chance, and their variance is far larger than the effects studied here".

**That rule is withdrawn.** It conflates task difficulty with statistical resolution. Headroom above chance
says nothing on its own about whether a *paired* comparison can resolve an effect: HellaSwag has ~10,000
validation items, so a single-model accuracy SE is about 0.45 pp and a 4 pp margin above chance is not "at the
variance scale", while SciQ has ~1,000 test items, so large headroom does not by itself buy resolution. Both
must be judged from measured per-example scores together with sample size.

It is replaced by the measured screen in §1.7, run on Exact only, before any condition-vs-Exact comparison.

### 1.6 Claims retired from this report

The following assertions were made without evidence and must not appear in any section:

- "adversarial distractors leave small models at chance rather than partially informed";
- "ARC-Challenge falls below chance because its distractors look more like high-frequency text";
- WinoGrande "contributes only noise" — the permitted wording is "the current model shows no measurable
  ability above chance on this task".

### 1.7 Pre-registered inclusion screen for layer 3 (written before stage A results) — superseded on 2026-09-20

**Status.** The screen itself was run and its measurements stand (§6). Its *output* — the hard
Primary / Reference / Unusable banding — is withdrawn as a gate on stage C and kept below, and in §6.4,
only as a record of what was explored. Stage C (§7) does not pre-judge a task's power from an assumed
paired correlation; it reports the realised per-document paired differences, intervals and effect sizes,
and lets each task's resolution show up in its own interval. The natural-text shrinkage measured in §6.3
remains a *sensitivity analysis of stage B only*: it is not used to build any stage-C interval or power
claim, it is not assumed equal across tasks, and neither kappa <= 1 nor kappa = 1 is used as a conversion
rule between corpus NLL and downstream margins.


Stage A measures **Exact only** on nine tasks: LAMBADA, BLiMP, SciQ (harness default and a no-support
variant), PIQA, ARC-Easy, OpenBookQA, HellaSwag, WinoGrande, ARC-Challenge. No condition-vs-Exact comparison is
computed at this stage.

For each task the screen records measured accuracy, accuracy normalised by continuation length, the continuous
main metric (margin or target NLL) with its mean and SD, the number of evaluated **examples** (not requests),
the chance baseline computed from that task's actual option count, the single-model accuracy SE
`sqrt(p(1−p)/n)`, the SE of the continuous metric, and the truncation rate.

**Primary criterion: the paired detection floor, not headroom above chance.** For each task the floor is
approximated as `2 × SD_Exact / sqrt(n)` on the continuous metric (and the analogous quantity on accuracy).
Using the SD of Exact's own per-example scores is deliberately conservative: paired differences between
conditions are usually less variable than raw scores because per-example difficulty cancels, so the true paired
floor will be smaller than the number reported here. The floor is then compared with this report's target
effect sizes: good-tier conditions differ by <0.01 nats of LM NLL, pathological ones by 0.13–0.89 nats.

Three bands, recorded rather than used for hard exclusion:

- **Primary task** — the detection floor is small enough to carry good-tier differences.
- **Reference task** — the floor can only carry pathological-tier differences.
- **Unusable** — Exact does not measure above the chance baseline computed from the actual option count.

Reference-task results are reported in their own table, with the detection floor in the table note, and are
never listed as equal-weight rows beside primary tasks.

### 1.8 Two attributions corrected

- **BLiMP subset.** All BLiMP numbers in this report come from `limit=200` per paradigm: ~13,400 pairs across
  67 paradigms, not the full 67,000-pair benchmark. The earlier description of 26,800 scored requests as "full
  BLiMP" was wrong; 26,800 is the request count for the 200-per-paradigm subset (two requests per pair).
- **SciQ template.** The harness default SciQ template places the support passage in the input, so it mainly
  measures reading comprehension of supplied material, not retained educational knowledge. Treating it as
  evidence of FineWebEdu domain match was wrong. Stage A therefore runs both the default template and a
  no-support variant, and §1.7 decides which enters the main task set.

### 1.9 Deployment configuration

All main results use each run's own training operator (native forward). The exact-forward arm (same weights,
exact softmax forward) is a second formal arm, used only as a descriptive decomposition, never as a causal
split: the pathological weights were learned while compensating for their own operator, so exact forward is a
particular out-of-distribution setting for them.

---

## 2. Gate and sanity checks

### 2.1 §8.0 hard implementation gate — PASS

| Condition | Target | Reproduced | Difference | Tolerance | Result |
|---|---|---|---|---|---|
| Exact | 3.1928 | 3.192772 | −2.81e-05 | 4.5e-05 | PASS |
| MinMax-Weight K4 | 4.0831 | 4.083116 | +1.59e-05 | 1e-03 | PASS |

Full FineWebEdu-3B validation split (976 blocks), batch 1, bf16 autocast outside the fp32 attention kernel,
TF32 off. Artefacts: `results/gate_report.json`, `results/gate_*_per_block.npy`.

### 2.2 Other checks

- **Condition identity (§8.1.1):** all 15 runs pass config-vs-manifest assertions (dataset, K, approximation,
  range_mode, τ, range_grad); `range_grad == full` for every quantized run.
- **Determinism (§8.1.3):** MinMax-Weight K4 evaluated twice is bit-identical per block.
- **exact-forward self-consistency (§8.1.6):** the Exact run gives bit-identical results in native and exact
  mode.
- Remaining checks (pairing alignment, probe validity, truncation rate) are reported with their layers.

---

## 3. Noise scale (§2.5, descriptive only)

Exact run, split-half absolute difference in mean block NLL, 20 repeats, 90th percentile. C4 uses document
clusters for the split. `results/layer1/noise_floor.csv`.

| Corpus | Blocks | Noise floor (p90) |
|---|---:|---:|
| WikiText-103 test | 276 | 0.057 |
| PTB test | 102 | 0.072 |
| C4 validation | 302 | 0.131 |

These are non-paired quantities and are **not** a significance gate; they say only which effects are of the
same magnitude as Exact's own split-to-split fluctuation.

## 4. Layer 1 — zero-shot PPL transfer

Corpora (GPT-2 BPE, `encode_ordinary` + EOT per document, non-overlapping 1024-token blocks, identical to the
training pipeline):

| Corpus | Source | Blocks | Notes |
|---|---|---:|---|
| WikiText-103 test | `Salesforce/wikitext`, `wikitext-103-raw-v1` | 276 | held out for all 15 runs (all trained on FineWebEdu) |
| PTB test | canonical `ptb.test.txt` (sha256 `dd65dff3…`) | 102 | HF script datasets are unsupported in `datasets` 5.x; source and hash recorded in `data/layer1/ptb_test_meta.json` |
| C4 validation | `allenai/c4` en validation, fixed-seed document sample | 302 | 717 documents; block→document map saved for cluster bootstrap |

All 15 runs × 3 corpora × {native, exact} = 90 evaluations. Tables: `results/layer1/ppl_transfer.csv`,
`rank_consistency.csv`, raw per-block rows in `per_block_rows.csv.gz`.

### 4.1 Δ NLL vs Exact, native forward

| Tier | Condition | FineWebEdu (main) | WikiText-103 | PTB | C4 |
|---|---|---:|---:|---:|---:|
| pathological | MinMax-Weight K4 | +0.890 | +1.215 | +1.122 | +0.865 |
| pathological | MinMax-Weight K16 | +0.447 | +0.588 | +0.526 | +0.435 |
| pathological | LERP K1 | +0.346 | +0.653 | +0.702 | +0.390 |
| pathological | MinMax-Weight K32 | +0.160 | +0.209 | +0.176 | +0.156 |
| pathological | QRM-Weight K2 | +0.130 | +0.180 | +0.135 | +0.128 |
| intermediate | MinMax-Prob K4 | +0.061 | +0.097 | +0.091 | +0.067 |
| intermediate | QRM-Weight K4 | +0.043 | +0.066 | +0.049 | +0.044 |
| intermediate | MinMax-Weight K64 | +0.038 | +0.045 | +0.068 | +0.038 |
| intermediate | QRM-Prob K4 | +0.019 | +0.027 | −0.020 | +0.021 |
| good | LERP K2 | +0.0097 | +0.005 | −0.016 | +0.013 |
| good | LERP K4 | +0.005 | +0.002 | −0.018 | +0.005 |
| good | QRM-Weight K16 | +0.004 | −0.002 | +0.023 | +0.005 |
| good | LERP K32 | +0.002 | −0.006 | −0.039 | +0.001 |
| good | LERP K16 | +0.0003 | −0.008 | −0.058 | −0.001 |

### 4.2 What layer 1 answers

- **Ordering is preserved out of domain.** Spearman ρ between each corpus's Δ ordering and the FineWebEdu
  ordering is 0.996 (WikiText), 0.952 (PTB) and 0.996 (C4), all with p < 1e-3.
- **Pathological gaps are amplified out of domain**, most on the most distant domains: MinMax-Weight K4 goes
  from +0.890 (FineWebEdu) to +1.215 (WikiText) and +1.122 (PTB). All 15 pathological comparisons are
  detectable on all three corpora.
- **The intermediate tier is detectable everywhere** (12/12 comparisons).
- **The good tier shows differences of at most 0.058 nats, with signs inconsistent across corpora and
  conditions (8 of 15 negative, i.e. the quantized model ahead of Exact), all below each corpus's own
  split-half fluctuation.** Under a single seed the paired interval measures evaluation-set sampling only and
  cannot separate an operator effect from a seed effect (main report §5.3). The largest good-tier effect,
  LERP K16 on PTB at −0.058, is 0.80 of that corpus's split-half scale; every other good-tier effect is below
  0.55 of its corpus scale.
- **The ±0.01 equivalence margin is reported but not over-read out of domain.** That margin was derived as a
  separation threshold inside the FineWebEdu training domain, so applying it unchanged to a far domain such as
  PTB is not appropriate. Good-tier results are therefore given both in absolute nats and as a fraction of the
  corpus's own split-half scale.

| Good-tier condition | WikiText Δ (×scale) | PTB Δ (×scale) | C4 Δ (×scale) |
|---|---:|---:|---:|
| LERP K2 | +0.005 (0.09) | −0.016 (−0.22) | +0.013 (0.10) |
| LERP K4 | +0.002 (0.04) | −0.018 (−0.25) | +0.005 (0.03) |
| LERP K16 | −0.008 (−0.15) | −0.058 (−0.80) | −0.001 (−0.01) |
| LERP K32 | −0.006 (−0.10) | −0.039 (−0.54) | +0.001 (0.01) |
| QRM-Weight K16 | −0.002 (−0.04) | +0.023 (0.31) | +0.005 (0.04) |

Scale = that corpus's split-half noise floor from §3 (0.057 / 0.072 / 0.131).

### 4.3 Post-hoc observation: gap amplification scales with domain distance

Not anticipated by the instructions; marked **post-hoc observation**, not a pre-registered test.

The pathological gaps are not amplified uniformly out of domain. They grow most on the corpora furthest from
the FineWebEdu training domain and are essentially unchanged on the nearest one:

| Condition | WikiText vs FineWebEdu | PTB vs FineWebEdu | C4 vs FineWebEdu |
|---|---:|---:|---:|
| LERP K1 | +89% | +103% | +13% |
| MinMax-Weight K4 | +37% | +26% | −3% |
| MinMax-Weight K16 | +32% | +18% | −3% |

C4 is the closest general web domain to FineWebEdu and shows no amplification at all; the two distant domains
show large amplification. Read descriptively: the in-domain NLL gap understates these conditions' deficit on
text unlike their training distribution.

### 4.4 exact-forward arm (§6), layer 1, all 15 conditions — a main layer-1 result

Same weights, forward replaced by exact softmax. `results/layer1/ppl_transfer.csv` (`forward_mode=exact`).

| Condition | mean Δ native | mean Δ exact |
|---|---:|---:|
| LERP K1 | +0.581 | +4.773 |
| LERP K2 | +0.001 | +1.723 |
| LERP K4 | −0.004 | +0.171 |
| MinMax-Weight K4 | +1.067 | +3.267 |
| MinMax-Weight K16 | +0.516 | +1.036 |
| MinMax-Weight K32 | +0.180 | +0.218 |
| MinMax-Prob K4 | +0.085 | +0.175 |
| QRM-Weight K2 | +0.148 | +0.256 |
| QRM-Weight K4 | +0.053 | +0.062 |
| QRM-Prob K4 | +0.009 | +0.005 |
| LERP K16 / K32, QRM-Weight K16, MinMax-Weight K64 | ≈0 | ≈0 |

- **Switching to exact softmax does not recover the gap; for 37 of 42 comparisons it makes things worse.** The
  effect is dramatic for coarse grids: LERP K1 goes from +0.58 to +4.77 and LERP K2 from +0.00 to +1.72.
- **Reading:** each model has learned a score geometry matched to its own operator. Model and operator behave
  as one joint system, and these operators are not drop-in replacements for softmax at inference time. The
  gaps are therefore not primarily a deployment-time kernel artefact that a better inference implementation
  would remove.
- **This direction was outside the pre-specification.** The instructions' §6 anticipated only "recovers" or
  "does not recover"; "worse than native" is an observation beyond that pre-specification and is reported as
  such. As required by §6 it stays descriptive: no causal split is claimed, and for the pathological
  conditions exact forward is itself an out-of-distribution setting.
- Under exact forward the cross-corpus rank correlation with FineWebEdu drops to 0.78–0.81, another sign that
  exact forward is a different regime rather than a correction.

**One-sentence conclusion for layer 1:** condition ordering transfers out of domain essentially perfectly
(ρ ≥ 0.95) and pathological gaps grow with domain distance, good-tier differences stay at or below each
corpus's own split-half fluctuation with inconsistent sign, and exact-forward evaluation makes coarse-grid
conditions much worse rather than recovering them, indicating that each model's geometry is matched to its own
operator.

---

## 5. Layer 2 — mechanism probes

Data: `data/probes/` (seed 20260920, shared by every condition, hashes in `probes_meta.json`). Token pool:
12,320 tiktoken tokens matching `^ ?[A-Za-z]{3,10}$`, unique lowercase form, ids in [1000, 20000), each
re-tokenizing to itself, no byte-level or control tokens (`token_pool.json`).

| Probe | Construction | Samples |
|---|---|---:|
| A induction | random sequence, inserted bigram (A,B), A recurs at distance d, predict B; **matched no-induction control**; distances 64/128/256/512/896 as a matched family (same content, distance varies) | 4000 (400 families × 5 distances × 2 arms) |
| B key-value | `k1:v1 … kn:vn` then a query for `k_i`, predict `v_i`; n ∈ {16,32,64}; queried key near (last quarter) or far (first quarter) | 2400 |
| C prefix copy | random segment S of length 32/64/128, `S <newline> S`, score every token of the second copy | 900 |

Sanity checks (`sanity_checks.json`): Exact induction gain at distance 64 is **+1.224 nats, CI [1.126, 1.328]**
(probe valid); Exact native vs exact forward is bit-identical; sample-id hashes are identical across conditions
by construction. Per-sample rows: `per_sample_rows.csv.gz`.

### 5.1 Instrument resolution comes first

| Probe | Metric | Exact score | Noise scale (p90) | Reference | Usable? |
|---|---|---:|---:|---|---|
| A | induction gain | 0.865 | 0.072 | >0 means induction works | degraded (§5.3) |
| A | target NLL | 11.160 | 0.136 | uniform = 10.825 | no |
| B | target NLL | 10.255 | 0.187 | uniform = 10.825; top-1 0.71% | **no** (§5.2) |
| C | target NLL (per token) | 1.692 | 0.033 | top-1 83% | **yes** |

Diagnostics that established this: `results/layer2/diagnostics/`.

### 5.2 Probe B is withdrawn from the conclusions

Exact's entire lead over a uniform predictor on probe B is 0.570 nats, and its top-1 accuracy is 0.71%. Most
conditions' degradation exceeds that lead (0.83 to 3.00 times it), which places them at or beyond uniform. At
124M this task is not measuring retrieval; the model barely departs from the uniform distribution at all. The
D3 classification rule ("retrieval loss" when NLL and top-1 both degrade, "sharpness effect" when only NLL
does) is itself sound, but applied to a measurement without resolution both classes are just different
appearances of noise. **Probe B therefore contributes no result.** It is recorded in the limitations as "KV
retrieval probing is not viable at this scale", and the good tier's +0.47 on probe B is not an effect that
needs explaining.

### 5.3 Probe A is downgraded, and the reason is about the models, not only the probe

Exact's own induction gain falls from 1.224 nats at d=64 to 0.209 at d=896, retaining 17%. Restricted to the
distances where the baseline keeps at least half its d=64 gain (64, 128, 256), the distance slope for
MinMax-Weight lies between +0.043 (target NLL) and −0.029 (induction gain), i.e. it flips sign with the metric
and is effectively zero.

The correct reading is not merely "this probe cannot test it": **at 124M @ 2.5B the models' own long-range
induction ability is too weak to carry a distance-dependence test.** The baseline capability decays to near
nothing by d≈900, so there is no headroom in which a condition-dependent distance effect could be observed.
This also explains why the geometry statistics that motivated the prediction (top-bucket mass 0.995, effective
buckets 1.03 for MinMax-Weight K4) have no measurable behavioural counterpart in this format at this scale:
the behaviour they would damage is barely present in the reference model. For future work, a distance-
dependence test needs a model with a strong long-range induction baseline, not a different probe alone.

The matched-family construction itself is sound: sequences at different distances are identical outside the
four inserted positions, with no stray occurrences of the induction tokens
(`d2_matched_family_audit.csv`).

### 5.4 Probe C — the one probe with adequate resolution

Per-token confirmed: the scorer averages cross-entropy over target positions and the target count equals the
segment length, so all probe C numbers are per-token means (`d1_probe_c_scale.csv`). Exact sits at 1.69 nats
per token with 83% top-1, far from any uniform ceiling.

| Condition | Tier | Δ L=32 | Δ L=64 | Δ L=128 | slope vs log2(L) [95% CI] |
|---|---|---:|---:|---:|---|
| LERP K1 | pathological | +7.96 | +9.24 | +9.89 | +0.963 [+0.929, +0.997] |
| MinMax-Weight K4 | pathological | +7.13 | +7.06 | +7.34 | +0.106 [+0.064, +0.146] |
| QRM-Weight K2 | pathological | +3.21 | +3.91 | +4.67 | +0.727 [+0.684, +0.770] |
| MinMax-Weight K16 | pathological | +1.91 | +1.49 | +1.24 | −0.337 [−0.375, −0.300] |
| MinMax-Prob K4 | intermediate | +0.82 | +1.41 | +2.69 | +0.939 [+0.897, +0.980] |
| QRM-Prob K4 | intermediate | +0.18 | +0.48 | +1.94 | +0.877 [+0.847, +0.907] |
| QRM-Weight K4 | intermediate | +0.55 | +0.70 | +1.01 | +0.234 [+0.200, +0.268] |
| MinMax-Weight K32 | pathological | +0.50 | +0.40 | +0.26 | −0.119 [−0.145, −0.095] |
| MinMax-Weight K64 | intermediate | +0.24 | +0.42 | +0.36 | +0.057 [+0.028, +0.085] |
| LERP K2 | good | +0.37 | +0.30 | +0.70 | +0.163 [+0.134, +0.190] |
| LERP K4 | good | +0.07 | +0.05 | +0.47 | +0.200 [+0.174, +0.226] |
| LERP K16 | good | −0.15 | −0.01 | +0.22 | +0.185 [+0.161, +0.208] |
| LERP K32 | good | −0.08 | +0.05 | +0.27 | +0.178 [+0.154, +0.201] |
| QRM-Weight K16 | good | −0.00 | −0.05 | −0.07 | −0.033 [−0.056, −0.011] |

Data: `probeC_length_trend.csv`, figure `probeC_length_trend.png`, contrasts `probeC_slope_contrast.json`.

**The surrogate position changes how degradation scales with copied length** [descriptive]. Holding
calibration and K fixed, so that only the surrogate position differs:

| Matched pair (same calibration, same K) | Weight-STE slope | Prob-STE slope | Difference [95% CI] |
|---|---:|---:|---|
| MinMax K4 | +0.106 | +0.939 | −0.833 [−0.889, −0.780] |
| QRM K4 | +0.234 | +0.877 | −0.643 [−0.679, −0.607] |

Both differences exclude 0 and neither pair's CIs overlap. Prob-STE conditions degrade progressively more as
the copied segment lengthens, while their Weight-STE counterparts stay nearly flat. Note that Prob-STE is the
*better* condition in absolute terms at every length here (e.g. MinMax-Prob K4 +2.69 against MinMax-Weight K4
+7.34 at L=128); the contrast is about the slope, not the level.

The pair named in the diagnostic request, MinMax-Weight K16 versus MinMax-Prob K4 (−0.337 against +0.939,
difference −1.275 [−1.324, −1.226]), shows the same direction but **differs in K as well as surrogate
position**, so it is not a clean surrogate-only comparison and the matched pairs above are used instead.

### 5.5 Predictions, final wording

- **P3 (`A = ΔNLL_probe − ΔNLL_LM > 0`): supported on the one probe with adequate resolution.** On probe C
  (sequence copying) A > 0 for 13 of 14 conditions, from +0.02 to +8.68. Probe B's apparent support is
  withdrawn (§5.2) and probe A's target NLL never tested it cleanly. The claim is therefore about sequence
  copying specifically, not "retrieval-type probes" in general.
- **P4 (degradation grows with distance): not testable here.** The original negative slope was an artefact of
  the decaying baseline, and in the corrected range the effect is zero within metric choice. The limiting
  factor is the models' own weak long-range induction at this scale (§5.3), not only probe design.
- **P5 (exact-forward reversal larger on probes than on LM): fails as measured, and the comparison is not
  dimensionally fair.** R(LM) aggregates hundreds of thousands of tokens of ordinary text, while R(probe)
  covers a single target token (A, B) or a few dozen (C); the two are not the same quantity, so the failure
  should not be read as evidence against the underlying idea. As measured: P5 holds in 5 of 18 cells and in 2
  of 12 on probes A and B, with R(LM) usually larger, e.g. LERP K1 R(LM) = +4.19 against +1.31 / +0.92 / +2.20.
- **P2 (good tier indistinguishable): consistent with the LM conclusion, on probe C evidence only.** Good-tier
  per-token effects on probe C run from −0.04 to +0.46, with QRM-Weight K16 slightly *better* than Exact at
  every length. The largest, LERP K2 at +0.46, is small next to the pathological range of +1.2 to +9.9. Probe B
  numbers do not enter this discussion.

### 5.6 Layer 2 limitations

- **The three probes have different target counts** — probe A one token, probe B one value token, probe C
  dozens of tokens. Magnitudes must not be compared across probes; only each probe's paired difference from
  Exact is meaningful.
- **Probe B is not viable at 124M:** Exact's lead over uniform is 0.570 nats with 0.71% top-1, so the probe has
  no resolution and contributes nothing.
- **Probe A cannot support a distance test at this scale** because the reference model's own induction gain
  decays to 17% of its short-distance value by d=896.
- Single seed throughout: paired intervals measure example sampling, not training-seed variance.
- The probes are synthetic; their relation to real capability is not independently validated.

**One-sentence conclusion for layer 2:** only sequence copying had the resolution to measure anything, and on
it the pathological tier degrades by +1.2 to +9.9 nats per token while the good tier stays within −0.04 to
+0.46, supporting P3 on that probe alone; P4 is untestable because these models' own long-range induction is
too weak; and the one new observation worth keeping is that at matched calibration and K, Prob-STE degradation
grows with copied length while Weight-STE degradation does not.

---

*Layer 3 follows; see §9 of the instructions for the stop points.*

## 6. Layer 3, stage A — Exact-only baselines on nine tasks

Stage A runs the Exact 2.5B endpoint (`formal_124m_s1337_exact_2p5b`) and nothing else, through the
`frozenop` lm-eval adapter at batch 1, fp32 attention, TF32 off, context 1024, GPT-2 BPE. No condition was
compared against Exact at this stage. The ten runs issued 102,519 scored requests; the nine after LAMBADA
took 1,059 s for 97,366 requests, i.e. 92 requests/s including dataset setup, and the Stage C cost estimates
below use a conservative 88 requests/s. Truncation was **0.0000 on every task** — no example exceeded the
1024-token context, so the left-truncation path was never taken and cannot confound anything below.

BLiMP is the `limit=200` per-paradigm subset throughout: 13,400 pairs across 67 paradigms, 26,800 requests,
not the full 67,000-pair benchmark (§1.8).

### 6.1 Accuracy channel

Chance is computed from the actual option count of each task, not assumed. `z` is `(acc − chance) / acc_SE`
for the single model; the accuracy floor is `2 × acc_SE` in percentage points.

**Correction to the `acc_norm` column.** These `acc_norm` values were recomputed by stage A's own code,
which divided by the character length of the scored continuation *including* the task's target delimiter.
The locked harness divides by the character length of the raw choice string, delimiter excluded. The two
differ slightly: for HellaSwag, stage A's 0.3010 against the harness's own 0.2999. Stage C uses the
harness value throughout (§7.2), and the `acc` column, which involves no normalisation, is unaffected.

| Task | n examples | options | chance | Exact acc | acc_norm | acc − chance (pp) | z | acc SE (pp) | acc floor (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP (subset) | 13,400 | 2.00 | 0.5000 | 0.7948 | 0.7434 | +29.48 | +84.5 | 0.35 | 0.70 |
| SciQ (default, with support) | 1,000 | 4.00 | 0.2500 | 0.7310 | 0.6750 | +48.10 | +34.3 | 1.40 | 2.80 |
| LAMBADA (OpenAI) | 5,153 | — | 0.0000 | 0.2164 | 0.2164 | +21.64 | +37.7 | 0.57 | 1.15 |
| SciQ (no support) | 1,000 | 4.00 | 0.2500 | 0.5120 | 0.4840 | +26.20 | +16.6 | 1.58 | 3.16 |
| ARC-Easy | 2,376 | 4.00 | 0.2502 | 0.4979 | 0.4588 | +24.77 | +24.2 | 1.03 | 2.05 |
| PIQA | 1,838 | 2.00 | 0.5000 | 0.6083 | 0.6001 | +10.83 | +9.5 | 1.14 | 2.28 |
| HellaSwag | 10,042 | 4.00 | 0.2500 | 0.2825 | 0.3010 | +3.25 | +7.2 | 0.45 | 0.90 |
| WinoGrande | 1,267 | 2.00 | 0.5000 | 0.4341 | 0.4341 | −6.59 | −4.7 | 1.39 | 2.78 |
| ARC-Challenge | 1,172 | 4.00 | 0.2502 | 0.1920 | 0.2270 | −5.82 | −5.1 | 1.15 | 2.30 |
| OpenBookQA | 500 | 4.00 | 0.2500 | 0.1640 | 0.2840 | −8.60 | −5.2 | 1.66 | 3.31 |

Three tasks — WinoGrande, ARC-Challenge, OpenBookQA — place Exact **below** the chance baseline by 4.7–5.2 SE.
The permitted wording (§1.6) is that the current model shows no measurable ability above chance on these
tasks; no claim is made here about why.

### 6.2 Continuous channel, and the units it is measured in

The continuous metric is the gold-minus-best-wrong log-likelihood margin for multiple-choice tasks and the
summed target NLL for LAMBADA. **The margin is normalised by the continuation's character length**, following
the harness `acc_norm` convention, so its natural unit is nats per character, not nats per token. To compare a
floor against this report's per-token LM NLL targets it is multiplied by the task's measured
characters-per-token; LAMBADA's summed NLL is divided by its mean target length instead. Those conversion
factors were measured on the actual scored continuations (`results/layer3/stageA/chars_per_token.json`):

| Task | mean continuation chars | mean continuation tokens | chars/token |
|---|---:|---:|---:|
| LAMBADA | 6.67 | 1.28 | 5.799 |
| BLiMP | 47.57 | 9.35 | 5.091 |
| SciQ (both arms) | 11.20 | 1.98 | 6.398 |
| PIQA | 99.22 | 22.96 | 4.419 |
| ARC-Easy | 23.58 | 4.50 | 5.729 |
| OpenBookQA | 17.78 | 3.39 | 5.603 |
| HellaSwag | 136.37 | 29.56 | 4.605 |
| WinoGrande | 24.82 | 5.61 | 4.293 |
| ARC-Challenge | 30.09 | 5.99 | 5.313 |

LAMBADA's targets are short — 77.8% are a single BPE token, mean 1.276 — so its per-example NLL floor is
close to a per-token floor.

| Task | metric | mean | SD | SE | floor `2·SD/√n` | floor in nats/token |
|---|---|---:|---:|---:|---:|---:|
| BLiMP (subset) | grammatical − ungrammatical margin | +0.0667 | 0.1228 | 0.00106 | 0.0021 | 0.0108 |
| HellaSwag | length-normalised margin | −0.0840 | 0.1671 | 0.00167 | 0.0033 | 0.0154 |
| PIQA | length-normalised margin | +0.0166 | 0.1811 | 0.00422 | 0.0084 | 0.0373 |
| ARC-Easy | length-normalised margin | −0.0639 | 0.3217 | 0.00660 | 0.0132 | 0.0756 |
| LAMBADA | summed target NLL | 5.3629 | 3.6906 | 0.05140 | 0.1028 | 0.0806 |
| SciQ (no support) | length-normalised margin | −0.0723 | 0.4788 | 0.01514 | 0.0303 | 0.1937 |
| SciQ (default) | length-normalised margin | +0.1226 | 0.4828 | 0.01527 | 0.0305 | 0.1954 |
| WinoGrande | length-normalised margin | −0.0081 | 0.0758 | 0.00213 | 0.0043 | 0.0183 |
| ARC-Challenge | length-normalised margin | −0.1876 | 0.3839 | 0.01121 | 0.0224 | 0.1192 |
| OpenBookQA | length-normalised margin | −0.2485 | 0.5002 | 0.02237 | 0.0447 | 0.2507 |

### 6.3 Two corrections the declared floor needs, pulling in opposite directions

The floor declared in §1.7, `2 × SD_Exact / √n`, is **not** conservative in the way that section claimed. It
equals `2 × SD(Δ) / √n` only when `SD(Δ) = SD_Exact`, i.e. when the condition and Exact correlate at ρ = 0.5.
Two corrections apply and they push opposite ways, so neither can be quietly ignored.

**(a) Pairing makes the floor smaller, and by a measured amount.** Writing `s = SD(cond − Exact) / SD(Exact)`,
the declared floor is `s = 1`. The realised `s` was measured on per-example data already collected
(`code/layer3_stageB_pairing.py`, 84 condition × dataset pairs):

| Source | good tier | intermediate | pathological |
|---|---:|---:|---:|
| Layer 1, C4 validation | 0.058 (ρ 0.998) | 0.058 (ρ 0.998) | 0.150 (ρ 0.991) |
| Layer 1, WikiText-103 test | 0.087 (ρ 0.997) | 0.098 (ρ 0.995) | 0.195 (ρ 0.985) |
| Layer 1, PTB test | 0.184 (ρ 0.984) | 0.174 (ρ 0.985) | 0.385 (ρ 0.936) |
| Layer 2, probe A | 0.825 (ρ 0.686) | 0.875 (ρ 0.647) | 0.888 (ρ 0.608) |
| Layer 2, probe B | 0.676 (ρ 0.770) | 0.675 (ρ 0.762) | 0.731 (ρ 0.713) |
| Layer 2, probe C | 1.039 (ρ 0.541) | 2.052 (ρ 0.404) | 1.717 (ρ 0.322) |

(medians per cell). On **natural text** the floor shrinks by 5–17×; on the **constructed probes** it does not
shrink at all, because those items were built so that the operator difference, not item difficulty, dominates
the variance. Layer-3 tasks are natural text, but their metric is already a within-example difference between
two options, so which regime applies is not settled in advance. The tables below therefore report the floor at
`s = 1` (as pre-registered), `s = 0.20` (natural text, pathological tier) and `s = 0.09` (natural text, good
tier), and the band is assigned twice.

**(b) Common-mode cancellation makes the floor effectively larger, by an unknown amount.** A per-token LM NLL
shift of δ that is common to the gold and the distractor continuations cancels exactly in a
gold-minus-best-wrong margin; only the differential part survives. The transfer coefficient κ from LM NLL to
margin is therefore ≤ 1 for every multiple-choice task and unknown before Stage C. All comparisons below use
κ = 1, which makes them an **upper bound on sensitivity**. LAMBADA is the exception: its metric is the target
NLL itself, so κ = 1 holds by construction, which is the one structural argument for keeping it in the set.

Stage C must replace both approximations with the realised paired SD and the realised margin shift, and must
report the band a task actually achieved rather than the one predicted here.

### 6.4 Band assignment

| Task | floor, s=1 | floor, s=0.20 | floor, s=0.09 | band as pre-registered | band, natural-text calibrated |
|---|---:|---:|---:|---|---|
| BLiMP (subset) | 0.0108 | 0.0022 | 0.0010 | reference | **primary** |
| HellaSwag | 0.0154 | 0.0031 | 0.0014 | reference | **primary** |
| PIQA | 0.0373 | 0.0075 | 0.0034 | reference | **primary** |
| ARC-Easy | 0.0756 | 0.0151 | 0.0068 | reference | reference |
| LAMBADA | 0.0806 | 0.0161 | 0.0073 | reference | reference |
| SciQ (no support) | 0.1937 | 0.0387 | 0.0174 | reference, lower edge only | reference |
| SciQ (default) | 0.1954 | 0.0391 | 0.0176 | reference, lower edge only | reference |
| WinoGrande | 0.0183 | 0.0037 | 0.0016 | unusable | unusable |
| ARC-Challenge | 0.1192 | 0.0238 | 0.0107 | unusable | unusable |
| OpenBookQA | 0.2507 | 0.0501 | 0.0226 | unusable | unusable |

All figures are nats per token; the good band is < 0.01 and the pathological band 0.13–0.89.

**Recommendation (withdrawn as a gate on 2026-09-20; retained as a record of the exploration).** The
banding below was not used to select stage-C tasks or to weight any stage-C conclusion. Stage C drops
WinoGrande, ARC-Challenge and OpenBookQA for scope reasons stated in §7.1, not because this screen
called them unusable, and it makes no claim that a below-chance Exact score demonstrates a failure
mechanism or that such a task is worthless.


- **Primary: BLiMP (subset), HellaSwag, PIQA.** Under any pairing at least as strong as the weakest natural-text
  case measured (PTB, s ≈ 0.20–0.39) their floors sit at or below the 0.01-nat good-band target. Their
  ordering — BLiMP best by a factor of ~1.4 over HellaSwag and ~3.5 over PIQA — is stable across every value
  of `s`, because `s` scales all floors alike.
- **Reference: ARC-Easy, LAMBADA, SciQ (both arms).** These reach the good band only under the most favourable
  pairing (`s = 0.09`) and are comfortably inside the pathological band at every `s`. Report them in their own
  table with the floor in the note, never as equal-weight rows beside primary tasks.
- **Unusable: WinoGrande, ARC-Challenge, OpenBookQA.** Exact scores below chance by 4.7–5.2 SE. Note that
  WinoGrande's *floor* is excellent (0.0183 at s=1, the second best in the set) — it is excluded purely on the
  chance criterion, which is the correct call because a margin measured on a model that cannot do the task is
  not a measurement of task ability. They may be run in Stage C as a null-channel check, but no capability
  claim may rest on them.

**Caveat that limits all three primaries.** HellaSwag clears the chance test at z = +7.2 but only by 3.25 pp,
and its `acc_norm` (0.3010) exceeds its `acc` (0.2825); its *accuracy* channel carries almost no information
at this model scale, and only its continuous margin is usable. PIQA's accuracy (0.6083) and BLiMP's (0.7948)
are solid. Across the whole set, the accuracy channel alone can carry only pathological-band differences:
the smallest accuracy floor in the table is 0.70 pp (BLiMP), and a 0.01-nat LM NLL gap will not move accuracy
by anything close to that.

### 6.5 Which SciQ enters the set

Both arms have essentially the same floor (0.0305 vs 0.0303; 0.195 vs 0.194 nats/token), so the choice is not
a power question. They differ in what they measure: with the support passage Exact scores 0.7310, without it
0.5120 — a 21.9 pp drop from removing the retrievable evidence while holding format, options and chance
constant. The default arm is therefore substantially an **in-context retrieval** task, which is exactly the
mechanism the approximation is expected to damage (§5.4), and the no-support arm is the knowledge control.

Recommendation: run **both** in Stage C as a matched pair — together they cost 8,000 requests per condition,
about 91 s, so there is no reason to choose. Report the default arm labelled as in-context retrieval, never as
retained science knowledge (§1.8), and report the difference between the two arms as the retrieval-dependence
contrast.

### 6.6 LAMBADA is reported separately, not as long-range evidence

Per the cross-layer correction, **LAMBADA is withdrawn from the "weak long-range ability" argument.** That
argument now rests only on the layer-2 evidence: probe A's induction gain decaying from 1.224 to 0.209 nats
between the shortest and longest distance, and probe B's Exact headroom over uniform of 0.570 nats with 0.71%
top-1 (§5.2, §5.3).

LAMBADA's Stage A numbers stand on their own, with both channels reported:

- accuracy 0.2164 (SE 0.0057), target NLL 5.3629 nats (SD 3.6906, SE 0.0514), perplexity 213.35 (SE 11.00);
- the model was trained on 2.5B tokens of FineWebEdu, while LAMBADA is book narrative whose target is a
  final word requiring discourse-level resolution — a corpus difference, which §4.3 already records as
  producing gap amplification with domain distance;
- the adapter is not the cause: BLiMP through the same adapter gives 0.7948, inside the published range for
  models of this class; stock GPT-2 through the same adapter gives LAMBADA accuracy 0.32 at perplexity 42.2;
  and scoring is invariant to the tokenisation checks in §2.2.

No claim is made here that the low accuracy is *caused* by domain difference; the corpus difference and the
adapter validation are recorded as the two facts available.

### 6.7 Stage C cost, and the stop

At the measured 88 requests/s, one condition scored on all ten task runs costs 102,519 requests ≈ 0.32 h.
Fifteen conditions on the native forward is ≈ 4.9 h on one GPU; adding the exact-forward arm doubles it to
≈ 9.7 h. Restricting Stage C to the three primary tasks plus the four reference tasks and dropping the three
unusable ones removes 9,221 requests per condition, about 9%, so trimming the task list is not where the cost
is.

**Stage C has not been started.** Stages A and B are complete and this section is the stop point.

---

## 7. Layer 3, stage C — seven configurations × 15 conditions, native forward

### 7.1 Scope, and what stage C does not inherit from stage B

Stage C evaluates **seven fixed configurations** on **all 15 conditions** of the 124M @ 2.5B suite, each
condition using its own trained forward operator (`native`). The exact-forward substitution arm is **not** run
in this round.

| # | Configuration | Task id | Coverage |
|---|---|---|---|
| 1 | BLiMP | `blimp` | `limit=200` per paradigm — **a subset**: 67 paradigms × 200 pairs = 13,400 pairs, 26,800 scored requests, not the full 67,000 pairs |
| 2 | HellaSwag | `hellaswag` | full validation split |
| 3 | PIQA | `piqa` | full validation split |
| 4 | ARC-Easy | `arc_easy` | full test split |
| 5 | LAMBADA OpenAI | `lambada_openai` | full test split |
| 6 | SciQ default | `sciq` | full test split, harness template **with** the supporting passage |
| 7 | SciQ no-support | `sciq_nosupport` | full test split, identical template, question, candidates and scoring, supporting passage removed |

The paradigm and pair counts above are the **actual clean counts** from the run, not nominal figures: every
one of the 67 paradigms contributed exactly 200 pairs.

WinoGrande, ARC-Challenge and OpenBookQA are **not** extended to stage C in this round. Their stage-A Exact
results are retained in §6.1 and remain valid measurements. No claim is made that they are useless or that a
below-chance Exact score demonstrates any particular failure mechanism; they are simply outside this round's
scope.

The stage-B banding is withdrawn as a gate (§1.7, §6.4). Stage C assumes nothing about paired correlation in
advance. Each comparison is built from the realised per-document paired differences, and its resolution is
whatever its own interval says it is. The natural-text shrinkage measured in §6.3 is not used to construct any
stage-C interval or power statement, is not assumed equal across tasks, and no κ conversion between corpus NLL
and downstream margins is applied anywhere in this section.

### 7.2 Manifest, identity and numerics

The full manifest is `results/layer3/stageC/manifest_stageC.json`. It records, per condition: checkpoint path,
full-file SHA-256, file size, a SHA-256 over the frozen training code root, parameter count, architecture,
the operator as stored in the checkpoint config and as effectively constructed, training seed, training data
directory, the final optimiser step, the per-step token budget, and the token count.

All 15 checkpoints are distinct files at the **same** final step **19,074**, with the trainer's own
`tokens_done` field reading **2,500,067,328 tokens** (19,074 × 131,072 tokens per step). "2.5B" is a label;
2,500,067,328 is the measured count. All 15 were trained on the same FineWebEdu corpus with seed 1337, so
every comparison below is a single-seed comparison.

Identity always comes from the manifest plus the checkpoint config, never from a directory name: the loader
asserts the calibration, reconstruction, K, τ and `range_grad` recorded in the checkpoint against the label
before any scoring happens, and refuses to proceed on a mismatch.

Evaluation numerics, identical for all conditions: tiktoken GPT-2 BPE (50,257 tokens), context length 1,024,
batch size 1, bf16 autocast outside the fp32 attention kernel, fp32 logits and `log_softmax`, TF32 disabled,
`model.eval()` with dropout inactive, `torch.inference_mode()`. Harness `lm-eval` 0.4.13 with a custom
`frozenop_stagec` adapter; seeds 20260920 for all harness randomness (the tasks are zero-shot, so no few-shot
sampling occurs; the seed is recorded anyway).

**The native operator is never silently replaced by exact softmax.** The preflight asserts this numerically:
loading QRM-Weight K4 with its own operator and with the exact-softmax forward on the same weights and the
same input produces a maximum logit difference of 6.2461, so the two paths are demonstrably different code.

### 7.3 Preflight

`code/layer3_stageC_preflight.py` ran two conditions (Exact and QRM-Weight K4) over five reduced
configurations before the sweep: **105 of 105 checks passed** (`results/layer3/stageC_preflight/`). It
verified the native operator is in effect and differs from exact softmax; eval mode with dropout inactive;
that every candidate's gold index, candidate ordering and scoring direction are right; that the scored-token
count, character count and token-normalised score are mutually consistent and that nothing was truncated;
that the standard `acc` and `acc_norm` rebuilt from the saved raw scores reproduce the harness's own values
exactly; that both conditions saw the same examples in the same order with byte-identical scored inputs; and
that the two SciQ arms align one-to-one by original document id with identical gold indices and candidate
order but different scored inputs.

One real defect was found and fixed by the preflight. The first implementation rebuilt `acc_norm` by dividing
by the character length of the scored continuation, which includes the task's target delimiter. The locked
harness divides by the character length of the **raw choice string**, delimiter excluded
(`lm_eval.api.task.Task.process_results`: `argmax(lls / [len(c) for c in doc_to_choice(doc)])`). The two
disagreed on SciQ no-support (0.400 rebuilt against 0.350 from the harness). Stage C now records
`cont_chars` (scored continuation, delimiter included), `choice_chars` (raw choice, delimiter excluded) and
`target_delimiter_chars` separately, and uses `choice_chars` for `acc_norm`, reproducing the harness exactly.
The consequence for earlier sections is recorded in §6.1.

### 7.4 Metrics, fixed before any comparison was computed

**Standard metrics** are whatever the locked harness natively produces; no task is given a metric it does not
natively define. The display metric of each configuration was fixed in advance:

| Configuration | Primary display accuracy | Also reported |
|---|---|---|
| BLiMP | paradigm macro-average of standard accuracy | per-paradigm accuracies |
| HellaSwag, PIQA, ARC-Easy | `acc_norm` | `acc` |
| SciQ default, SciQ no-support | `acc` | `acc_norm` |
| LAMBADA | standard `acc` | standard perplexity, and target NLL (below) |

`acc_norm` here is the harness's own quantity and is **length-normalised by characters of the raw choice
string**, not by tokens. It is not renamed and not reinterpreted as a token-normalised score.

**Supplementary continuous metric, multiple choice.** For candidate *j*, with `log P` the total natural-log
probability of the continuation actually scored and `N_j` the number of tokens actually scored (no prompt, no
padding, no unscored tokens):

```
s_j     = log P(y_j | x) / N_j
margin  = s_gold − max_{j ≠ gold} s_j          (larger is better)
```

Every candidate's `s_j`, the gold score, the best wrong score and the margin are stored per document. This
margin is computed **directly from the per-candidate raw total log-probability and the scored token count**;
no character-normalised quantity is ever multiplied by an average chars-per-token to approximate it. The
margin is a supplementary diagnostic: it does not replace the standard accuracy, and a significant change in
margin is not read as a change in answering accuracy.

**BLiMP.** The standard metric stays the harness's sentence-total-probability comparison with its native
scoring and tie handling. The supplementary margin is reported under a separate name:
`margin = log P(good)/N_good − log P(bad)/N_bad`, using the actual scored token counts.

**LAMBADA.** Stored and reported together: the standard harness `acc`; the standard harness `perplexity`,
whose actual definition was verified empirically to be `exp(mean over examples of the total target NLL)` —
not a token-weighted corpus perplexity, and not a full-text perplexity; each example's total target NLL
`−log P(target | context)`; each example's target token count; each example's token-normalised target NLL;
the example-average of both; and the **corpus token-weighted target NLL**, `sum_i NLL_i / sum_i N_i`, which is
the primary continuous metric for this configuration. Its bootstrap redraws examples and recomputes the ratio
of sums in each replicate; it is never computed as an average of per-example ratios. LAMBADA's target-word
perplexity is never called a full-text perplexity, no `chance = 0` z-test is used, and this configuration
alone is not treated as evidence about long-range context ability (§6.6).

### 7.5 Statistical plan, fixed before the comparisons were run

Every comparison is one of the 14 methods against Exact, in the direction **method − Exact**. For accuracy
and margin a positive value means the method is better; for NLL a positive value means the method is worse.
Accuracy differences are in percentage points; continuous metrics are in nats with the aggregation named.

- **Bootstrap.** 4,000 paired resamples, 95% percentile interval, seed 20260920. Within a comparison, the
  method and Exact are always resampled with the *same* document draw — never independently — and the draw is
  shared by all 14 methods in a configuration so their intervals are mutually comparable. For example-average
  metrics the per-document difference is formed first and then resampled. For LAMBADA's corpus token-weighted
  NLL each replicate redraws documents once and recomputes both ratios of sums and their difference. BLiMP
  resamples sentence pairs *within* each of the 67 paradigms and then takes the paradigm macro-average, so its
  interval is conditional on this fixed paradigm set and is not an extrapolation to all grammatical phenomena.
  Resampling is implemented as multinomial document counts, which is exactly equivalent to drawing an index
  vector with replacement for the statistics used here; the implementation was checked against a brute-force
  index bootstrap and against analytic normal intervals.
- **Hypothesis test.** Two-sided paired label-exchange (permutation) test, 20,000 Monte-Carlo draws, seed
  20260921. Each draw swaps the method and Exact records *within a document*, preserving the aggregation
  structure — BLiMP's paradigm macro-average, LAMBADA's ratio of sums, SciQ's per-document support gain. The
  Monte-Carlo p-value uses the add-one correction, `p = (1 + #{|T*| ≥ |T_obs|}) / (1 + B)`, so p is never
  reported as zero. The exchangeability assumption is over evaluation documents. **This is not a test of
  training randomness.**
- **Multiplicity.** BH is applied separately within each (configuration × metric) family of 14 comparisons;
  the SciQ difference-in-differences forms its own families, one per scoring metric, never mixed with the
  single-arm comparisons. Raw p and BH q are both reported and q < 0.05 is the threshold *inside that family*.
  The full metric list and the comparison families were fixed before any result was seen; no metric was chosen
  after seeing significance. The 95% bootstrap intervals are **not** multiplicity-adjusted and are never
  described as BH-corrected; whether an interval excludes zero and whether q < 0.05 are reported separately.
  No claim is made of controlling an overall FDR across tasks or across metrics.
- **Seeds.** Each condition has a single training seed (1337). Every interval and every p-value below
  therefore reflects **evaluation-sample uncertainty only** and says nothing about training-seed variability.

### 7.6 SciQ support difference-in-differences

Both arms use identical original documents and candidate order. The primary analysis uses `acc`; `acc_norm`
is analysed separately and the two scoring conventions are never mixed. Per document *i*:

```
d_i = (correct_method,support − correct_method,nosupport)
    − (correct_Exact,support  − correct_Exact,nosupport)
```

whose mean is `(A_method,support − A_method,nosupport) − (A_Exact,support − A_Exact,nosupport)`. The bootstrap
resamples original documents, with all four cells always using the same draw; the label-exchange test uses
each document's method/Exact *support gain* as the paired unit, preserving the within-arm relationship. A
negative value means the method gains less from the supporting passage than Exact does.

Wording is limited to **whether the approximate operator reduced the benefit obtained from supplied
material**. It is not attributed to any specific retrieval mechanism. Exact's own support gain shows only that
Exact benefits from the material; it is not evidence that an approximate operator damages that ability.

### 7.7 Completeness and integrity of the run

`code/layer3_stageC_verify.py` ran over the per-item files before any statistic was computed:
**768 of 768 checks passed** (`results/layer3/stageC/integrity_check.json`).

| Quantity | Value |
|---|---:|
| Condition × configuration units completed | 105 / 105 |
| Per-item rows written | 1,399,470 |
| Scored requests | 1,399,470 |
| Truncated requests | **0** |
| Failed requests | **0** |
| Retries | **0** |
| GPU time (sum of unit wall-times) | 17,140 s = 4.76 h on one RTX 5090 |

Nothing was silently dropped: no example failed, none needed a retry, and no example exceeded the 1,024-token
context, so the left-truncation path was never taken anywhere in the sweep.

The checks that matter for comparability all passed for every configuration: the `order_hash` — a hash over
the ordered scored inputs — is **identical across all 15 conditions** in each configuration, as are row
counts, document hashes, gold indices and (for LAMBADA) per-example target token counts. The 15 checkpoints
are 15 distinct files, all at step 19,074, and each carries the operator its manifest label claims. The
standard `acc` and `acc_norm` rebuilt from the saved per-candidate scores reproduce the harness's own values
to within 1e-12 for every condition and configuration, and for BLiMP every one of the 67 per-paradigm
accuracies matches the harness exactly (maximum difference 0.0). The two SciQ arms share documents, document
hashes, gold indices and candidate order, and differ in every single scored input.

Per-condition cost ranged from 592 s (Exact) to 1,639 s (MinMax-Prob K4); the Prob-STE and hard-Nearest
operators are 1.7–2.8× slower than exact softmax at batch 1.

### 7.8 Results

Full machine-readable tables are in `results/layer3/stageC/tables/`: `point_estimates.csv`,
`paired_comparisons.csv` (all 21 configuration × metric families × 14 methods), `blimp_subtasks.csv`
(67 paradigms × 15 conditions), `sciq_difference_in_differences.csv`, `stats_meta.json`.

#### Standard metrics, seven configurations × 15 conditions

All values are percent. BLiMP is the paradigm macro-average over 67 paradigms of the limit=200 subset.

| Condition | BLiMP (subset) acc (macro) | HellaSwag acc | HellaSwag acc_norm | PIQA acc | PIQA acc_norm | ARC-Easy acc | ARC-Easy acc_norm | LAMBADA acc | SciQ default acc | SciQ default acc_norm | SciQ no-support acc | SciQ no-support acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 79.48 | 28.25 | 29.99 | 60.83 | 59.74 | 49.79 | 45.20 | 21.64 | 73.10 | 66.10 | 51.20 | 47.60 |
| LERP K1 | 74.23 | 27.49 | 28.05 | 60.66 | 59.52 | 45.54 | 41.29 | 8.71 | 64.60 | 58.80 | 46.00 | 43.00 |
| LERP K2 | 78.50 | 28.22 | 29.97 | 62.30 | 59.96 | 49.41 | 42.85 | 20.22 | 75.10 | 65.60 | 52.70 | 45.50 |
| LERP K4 | 78.37 | 28.36 | 30.35 | 61.21 | 59.52 | 48.91 | 44.61 | 21.72 | 74.40 | 67.20 | 52.70 | 47.10 |
| LERP K16 | 78.22 | 28.43 | 30.17 | 61.43 | 60.55 | 49.79 | 45.50 | 21.54 | 73.50 | 63.80 | 51.70 | 46.10 |
| LERP K32 | 77.90 | 28.35 | 30.15 | 60.39 | 60.23 | 50.72 | 44.57 | 22.39 | 74.10 | 65.60 | 51.60 | 47.30 |
| MinMax-Weight K4 | 60.08 | 26.23 | 26.08 | 57.51 | 55.60 | 38.05 | 35.61 | 3.26 | 53.80 | 49.90 | 33.60 | 35.80 |
| MinMax-Weight K16 | 69.96 | 26.89 | 27.32 | 58.43 | 57.51 | 43.81 | 39.52 | 10.07 | 63.80 | 56.70 | 42.50 | 38.30 |
| MinMax-Weight K32 | 75.72 | 27.59 | 28.37 | 60.34 | 59.30 | 46.51 | 41.75 | 17.27 | 70.90 | 62.20 | 49.80 | 43.80 |
| MinMax-Weight K64 | 77.53 | 28.20 | 29.68 | 61.70 | 60.12 | 49.71 | 44.53 | 21.09 | 74.20 | 65.10 | 51.80 | 47.40 |
| MinMax-Prob K4 | 78.28 | 28.13 | 29.37 | 60.72 | 60.99 | 49.37 | 44.02 | 19.25 | 72.60 | 65.30 | 52.10 | 46.10 |
| QRM-Weight K2 | 74.04 | 27.81 | 28.89 | 60.45 | 58.76 | 48.11 | 42.59 | 17.99 | 71.90 | 62.90 | 49.80 | 44.90 |
| QRM-Weight K4 | 76.72 | 28.18 | 29.54 | 61.48 | 60.72 | 49.49 | 45.03 | 19.95 | 74.70 | 65.80 | 52.10 | 46.00 |
| QRM-Weight K16 | 78.96 | 28.43 | 29.87 | 61.43 | 61.43 | 50.17 | 44.91 | 22.47 | 77.00 | 66.10 | 54.50 | 47.60 |
| QRM-Prob K4 | 78.43 | 28.56 | 29.63 | 61.26 | 60.50 | 49.79 | 44.40 | 21.72 | 72.40 | 64.80 | 53.50 | 47.20 |

#### Supplementary continuous metrics (nats)

| Condition | BLiMP (subset) margin | HellaSwag margin | PIQA margin | ARC-Easy margin | SciQ default margin | SciQ no-support margin | LAMBADA NLL corpus/token | LAMBADA NLL per example | LAMBADA NLL token-norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 0.3373 | -0.3794 | 0.0821 | -0.4449 | 0.9615 | -0.1041 | 4.2043 | 5.3629 | 4.6604 |
| LERP K1 | 0.2542 | -0.4194 | 0.0814 | -0.5333 | 0.6308 | -0.2283 | 5.6107 | 7.1568 | 6.2193 |
| LERP K2 | 0.3064 | -0.3803 | 0.0861 | -0.4767 | 0.9992 | -0.1304 | 4.1936 | 5.3492 | 4.6492 |
| LERP K4 | 0.3166 | -0.3772 | 0.0738 | -0.4452 | 1.0535 | -0.0785 | 4.1106 | 5.2433 | 4.5526 |
| LERP K16 | 0.3240 | -0.3777 | 0.0879 | -0.4764 | 0.8856 | -0.1444 | 4.1722 | 5.3219 | 4.6175 |
| LERP K32 | 0.3086 | -0.3792 | 0.0900 | -0.4152 | 0.9913 | -0.0881 | 4.0824 | 5.2074 | 4.5314 |
| MinMax-Weight K4 | 0.0763 | -0.4807 | 0.0393 | -0.7130 | 0.0226 | -0.7489 | 7.2106 | 9.1976 | 7.9451 |
| MinMax-Weight K16 | 0.1883 | -0.4499 | 0.0556 | -0.5951 | 0.4911 | -0.4529 | 5.7081 | 7.2811 | 6.3107 |
| MinMax-Weight K32 | 0.2688 | -0.4165 | 0.0727 | -0.5093 | 0.7488 | -0.1949 | 4.6832 | 5.9738 | 5.1770 |
| MinMax-Weight K64 | 0.3054 | -0.3901 | 0.0892 | -0.4109 | 0.9832 | -0.0597 | 4.2615 | 5.4358 | 4.7176 |
| MinMax-Prob K4 | 0.3177 | -0.3901 | 0.0838 | -0.4467 | 1.0050 | -0.1013 | 4.3383 | 5.5338 | 4.8056 |
| QRM-Weight K2 | 0.2510 | -0.4019 | 0.0746 | -0.5047 | 0.7979 | -0.1835 | 4.6314 | 5.9077 | 5.1315 |
| QRM-Weight K4 | 0.2956 | -0.3878 | 0.0878 | -0.4499 | 0.9893 | -0.0519 | 4.3152 | 5.5044 | 4.7769 |
| QRM-Weight K16 | 0.3222 | -0.3770 | 0.0895 | -0.4474 | 1.0579 | -0.0319 | 4.1043 | 5.2353 | 4.5454 |
| QRM-Prob K4 | 0.3073 | -0.3829 | 0.0841 | -0.4443 | 0.8929 | -0.0872 | 4.2194 | 5.3821 | 4.6692 |

#### Paired comparisons, method − Exact

`*` the 95% paired bootstrap interval excludes zero (not multiplicity-adjusted); `†` BH q < 0.05 inside the (configuration × metric) family of 14 comparisons.


**BLiMP (subset) — acc_macro** (paradigm macro-average of standard accuracy; percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 74.2313 | -5.2463 | -5.9104 | -4.5896 | 0.0000 | 0.0001 | *† |
| LERP K2 | 78.5000 | -0.9776 | -1.6045 | -0.3433 | 0.0026 | 0.0028 | *† |
| LERP K4 | 78.3731 | -1.1045 | -1.6868 | -0.5149 | 0.0004 | 0.0006 | *† |
| LERP K16 | 78.2164 | -1.2612 | -1.8507 | -0.6642 | 0.0000 | 0.0001 | *† |
| LERP K32 | 77.8955 | -1.5821 | -2.1644 | -0.9701 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K4 | 60.0821 | -19.3955 | -20.2388 | -18.5448 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 69.9627 | -9.5149 | -10.2315 | -8.8060 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 75.7164 | -3.7612 | -4.4030 | -3.1119 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 77.5299 | -1.9478 | -2.5524 | -1.3358 | 0.0000 | 0.0001 | *† |
| MinMax-Prob K4 | 78.2761 | -1.2015 | -1.7910 | -0.5894 | 0.0002 | 0.0003 | *† |
| QRM-Weight K2 | 74.0373 | -5.4403 | -6.1045 | -4.7836 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 76.7164 | -2.7612 | -3.3808 | -2.1343 | 0.0000 | 0.0001 | *† |
| QRM-Weight K16 | 78.9627 | -0.5149 | -1.1194 | +0.0746 | 0.1034 | 0.1034 | — |
| QRM-Prob K4 | 78.4254 | -1.0522 | -1.6493 | -0.4328 | 0.0007 | 0.0008 | *† |

**BLiMP (subset) — margin_macro** (paradigm macro-average of logP(good)/N_good - logP(bad)/N_bad; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 0.2542 | -0.0831 | -0.0883 | -0.0778 | 0.0000 | 0.0000 | *† |
| LERP K2 | 0.3064 | -0.0309 | -0.0359 | -0.0259 | 0.0000 | 0.0000 | *† |
| LERP K4 | 0.3166 | -0.0207 | -0.0254 | -0.0159 | 0.0000 | 0.0000 | *† |
| LERP K16 | 0.3240 | -0.0133 | -0.0181 | -0.0087 | 0.0000 | 0.0000 | *† |
| LERP K32 | 0.3086 | -0.0287 | -0.0332 | -0.0238 | 0.0000 | 0.0000 | *† |
| MinMax-Weight K4 | 0.0763 | -0.2610 | -0.2694 | -0.2528 | 0.0000 | 0.0000 | *† |
| MinMax-Weight K16 | 0.1883 | -0.1490 | -0.1548 | -0.1432 | 0.0000 | 0.0000 | *† |
| MinMax-Weight K32 | 0.2688 | -0.0685 | -0.0736 | -0.0632 | 0.0000 | 0.0000 | *† |
| MinMax-Weight K64 | 0.3054 | -0.0319 | -0.0367 | -0.0272 | 0.0000 | 0.0000 | *† |
| MinMax-Prob K4 | 0.3177 | -0.0196 | -0.0241 | -0.0149 | 0.0000 | 0.0000 | *† |
| QRM-Weight K2 | 0.2510 | -0.0863 | -0.0917 | -0.0811 | 0.0000 | 0.0000 | *† |
| QRM-Weight K4 | 0.2956 | -0.0417 | -0.0465 | -0.0368 | 0.0000 | 0.0000 | *† |
| QRM-Weight K16 | 0.3222 | -0.0151 | -0.0196 | -0.0104 | 0.0000 | 0.0000 | *† |
| QRM-Prob K4 | 0.3073 | -0.0300 | -0.0349 | -0.0250 | 0.0000 | 0.0000 | *† |

**HellaSwag — acc_norm** (standard acc_norm (primary); percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 28.0522 | -1.9418 | -2.5794 | -1.2946 | 0.0000 | 0.0002 | *† |
| LERP K2 | 29.9741 | -0.0199 | -0.5676 | +0.5477 | 0.9717 | 0.9717 | — |
| LERP K4 | 30.3525 | +0.3585 | -0.1792 | +0.9162 | 0.2134 | 0.3319 | — |
| LERP K16 | 30.1733 | +0.1792 | -0.3585 | +0.7170 | 0.5422 | 0.6846 | — |
| LERP K32 | 30.1534 | +0.1593 | -0.3784 | +0.6971 | 0.5868 | 0.6846 | — |
| MinMax-Weight K4 | 26.0805 | -3.9136 | -4.7401 | -3.1368 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | 27.3153 | -2.6787 | -3.3360 | -2.0312 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | 28.3708 | -1.6232 | -2.2207 | -1.0655 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K64 | 29.6754 | -0.3187 | -0.8664 | +0.2589 | 0.2703 | 0.3785 | — |
| MinMax-Prob K4 | 29.3667 | -0.6274 | -1.1751 | -0.0896 | 0.0242 | 0.0566 | * |
| QRM-Weight K2 | 28.8887 | -1.1054 | -1.6630 | -0.5375 | 0.0001 | 0.0004 | *† |
| QRM-Weight K4 | 29.5359 | -0.4581 | -1.0257 | +0.0799 | 0.1127 | 0.2254 | — |
| QRM-Weight K16 | 29.8745 | -0.1195 | -0.6572 | +0.4083 | 0.6858 | 0.7385 | — |
| QRM-Prob K4 | 29.6256 | -0.3685 | -0.9162 | +0.1792 | 0.2060 | 0.3319 | — |

**HellaSwag — acc** (standard acc; percentage points; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 27.4945 | -0.7568 | -1.2251 | -0.2689 | 0.0020 | 0.0096 | *† |
| LERP K2 | 28.2215 | -0.0299 | -0.4382 | +0.3884 | 0.9249 | 0.9249 | — |
| LERP K4 | 28.3609 | +0.1095 | -0.3087 | +0.5278 | 0.6338 | 0.8289 | — |
| LERP K16 | 28.4306 | +0.1792 | -0.2091 | +0.5778 | 0.4094 | 0.7164 | — |
| LERP K32 | 28.3509 | +0.0996 | -0.2788 | +0.4979 | 0.6513 | 0.8289 | — |
| MinMax-Weight K4 | 26.2298 | -2.0215 | -2.6090 | -1.4240 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K16 | 26.8871 | -1.3643 | -1.8721 | -0.8464 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K32 | 27.5941 | -0.6572 | -1.0854 | -0.1992 | 0.0048 | 0.0168 | *† |
| MinMax-Weight K64 | 28.2016 | -0.0498 | -0.4581 | +0.3386 | 0.8415 | 0.9062 | — |
| MinMax-Prob K4 | 28.1318 | -0.1195 | -0.5377 | +0.3286 | 0.6110 | 0.8289 | — |
| QRM-Weight K2 | 27.8132 | -0.4382 | -0.8666 | +0.0100 | 0.0610 | 0.1709 | — |
| QRM-Weight K4 | 28.1816 | -0.0697 | -0.4680 | +0.3585 | 0.7840 | 0.9062 | — |
| QRM-Weight K16 | 28.4306 | +0.1792 | -0.2191 | +0.5875 | 0.4059 | 0.7164 | — |
| QRM-Prob K4 | 28.5600 | +0.3087 | -0.0899 | +0.7269 | 0.1560 | 0.3640 | — |

**HellaSwag — margin** (token-normalised gold minus best-wrong margin; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | -0.4194 | -0.0400 | -0.0459 | -0.0338 | 0.0000 | 0.0001 | *† |
| LERP K2 | -0.3803 | -0.0009 | -0.0053 | +0.0033 | 0.6843 | 0.7369 | — |
| LERP K4 | -0.3772 | +0.0022 | -0.0020 | +0.0062 | 0.3038 | 0.3866 | — |
| LERP K16 | -0.3777 | +0.0016 | -0.0024 | +0.0056 | 0.4141 | 0.4832 | — |
| LERP K32 | -0.3792 | +0.0002 | -0.0037 | +0.0041 | 0.9355 | 0.9355 | — |
| MinMax-Weight K4 | -0.4807 | -0.1014 | -0.1099 | -0.0924 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | -0.4499 | -0.0705 | -0.0765 | -0.0642 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | -0.4165 | -0.0371 | -0.0418 | -0.0323 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | -0.3901 | -0.0107 | -0.0146 | -0.0067 | 0.0000 | 0.0001 | *† |
| MinMax-Prob K4 | -0.3901 | -0.0107 | -0.0148 | -0.0064 | 0.0000 | 0.0001 | *† |
| QRM-Weight K2 | -0.4019 | -0.0225 | -0.0269 | -0.0178 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | -0.3878 | -0.0084 | -0.0125 | -0.0039 | 0.0002 | 0.0003 | *† |
| QRM-Weight K16 | -0.3770 | +0.0024 | -0.0016 | +0.0063 | 0.2287 | 0.3202 | — |
| QRM-Prob K4 | -0.3829 | -0.0035 | -0.0076 | +0.0007 | 0.0950 | 0.1478 | — |

**PIQA — acc_norm** (standard acc_norm (primary); percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 59.5212 | -0.2176 | -1.9042 | +1.4690 | 0.8464 | 0.8464 | — |
| LERP K2 | 59.9565 | +0.2176 | -1.3602 | +1.7954 | 0.8368 | 0.8464 | — |
| LERP K4 | 59.5212 | -0.2176 | -1.6866 | +1.3058 | 0.8338 | 0.8464 | — |
| LERP K16 | 60.5550 | +0.8161 | -0.8705 | +2.3939 | 0.3465 | 0.6581 | — |
| LERP K32 | 60.2285 | +0.4897 | -0.9793 | +1.9587 | 0.5542 | 0.8464 | — |
| MinMax-Weight K4 | 55.6039 | -4.1349 | -6.2568 | -1.9587 | 0.0002 | 0.0028 | *† |
| MinMax-Weight K16 | 57.5082 | -2.2307 | -4.0261 | -0.4353 | 0.0165 | 0.1158 | * |
| MinMax-Weight K32 | 59.3036 | -0.4353 | -2.0675 | +1.1425 | 0.6332 | 0.8464 | — |
| MinMax-Weight K64 | 60.1197 | +0.3808 | -1.1425 | +1.9042 | 0.6734 | 0.8464 | — |
| MinMax-Prob K4 | 60.9902 | +1.2514 | -0.3264 | +2.7761 | 0.1246 | 0.4363 | — |
| QRM-Weight K2 | 58.7595 | -0.9793 | -2.5571 | +0.6529 | 0.2490 | 0.5810 | — |
| QRM-Weight K4 | 60.7182 | +0.9793 | -0.5985 | +2.5571 | 0.2428 | 0.5810 | — |
| QRM-Weight K16 | 61.4255 | +1.6866 | +0.2176 | +3.1556 | 0.0295 | 0.1379 | * |
| QRM-Prob K4 | 60.5005 | +0.7617 | -0.8161 | +2.3395 | 0.3761 | 0.6581 | — |

**PIQA — acc** (standard acc; percentage points; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 60.6638 | -0.1632 | -1.7424 | +1.4146 | 0.8985 | 0.9442 | — |
| LERP K2 | 62.2960 | +1.4690 | +0.0544 | +2.8849 | 0.0543 | 0.2536 | * |
| LERP K4 | 61.2078 | +0.3808 | -1.0881 | +1.7954 | 0.6654 | 0.8088 | — |
| LERP K16 | 61.4255 | +0.5985 | -0.8705 | +2.0675 | 0.4802 | 0.8088 | — |
| LERP K32 | 60.3917 | -0.4353 | -1.9587 | +1.0337 | 0.6179 | 0.8088 | — |
| MinMax-Weight K4 | 57.5082 | -3.3188 | -5.3319 | -1.2514 | 0.0023 | 0.0329 | *† |
| MinMax-Weight K16 | 58.4331 | -2.3939 | -4.2437 | -0.5441 | 0.0122 | 0.0854 | * |
| MinMax-Weight K32 | 60.3373 | -0.4897 | -2.1219 | +1.0337 | 0.5866 | 0.8088 | — |
| MinMax-Weight K64 | 61.6975 | +0.8705 | -0.7073 | +2.3939 | 0.2972 | 0.8088 | — |
| MinMax-Prob K4 | 60.7182 | -0.1088 | -1.6880 | +1.4146 | 0.9442 | 0.9442 | — |
| QRM-Weight K2 | 60.4461 | -0.3808 | -2.0131 | +1.2514 | 0.6932 | 0.8088 | — |
| QRM-Weight K4 | 61.4799 | +0.6529 | -0.8161 | +2.1763 | 0.4425 | 0.8088 | — |
| QRM-Weight K16 | 61.4255 | +0.5985 | -0.8705 | +2.1219 | 0.4735 | 0.8088 | — |
| QRM-Prob K4 | 61.2622 | +0.4353 | -1.0337 | +1.8498 | 0.6124 | 0.8088 | — |

**PIQA — margin** (token-normalised gold minus best-wrong margin; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 0.0814 | -0.0007 | -0.0148 | +0.0133 | 0.9218 | 0.9218 | — |
| LERP K2 | 0.0861 | +0.0040 | -0.0086 | +0.0164 | 0.5227 | 0.6653 | — |
| LERP K4 | 0.0738 | -0.0083 | -0.0192 | +0.0033 | 0.1515 | 0.4496 | — |
| LERP K16 | 0.0879 | +0.0058 | -0.0060 | +0.0180 | 0.3470 | 0.5398 | — |
| LERP K32 | 0.0900 | +0.0079 | -0.0035 | +0.0193 | 0.1698 | 0.4496 | — |
| MinMax-Weight K4 | 0.0393 | -0.0428 | -0.0635 | -0.0211 | 0.0001 | 0.0021 | *† |
| MinMax-Weight K16 | 0.0556 | -0.0265 | -0.0422 | -0.0102 | 0.0013 | 0.0091 | *† |
| MinMax-Weight K32 | 0.0727 | -0.0094 | -0.0220 | +0.0028 | 0.1410 | 0.4496 | — |
| MinMax-Weight K64 | 0.0892 | +0.0071 | -0.0052 | +0.0192 | 0.2643 | 0.4625 | — |
| MinMax-Prob K4 | 0.0838 | +0.0017 | -0.0105 | +0.0132 | 0.7806 | 0.8406 | — |
| QRM-Weight K2 | 0.0746 | -0.0075 | -0.0202 | +0.0058 | 0.2572 | 0.4625 | — |
| QRM-Weight K4 | 0.0878 | +0.0057 | -0.0073 | +0.0191 | 0.3902 | 0.5463 | — |
| QRM-Weight K16 | 0.0895 | +0.0074 | -0.0037 | +0.0185 | 0.1927 | 0.4496 | — |
| QRM-Prob K4 | 0.0841 | +0.0020 | -0.0101 | +0.0145 | 0.7602 | 0.8406 | — |

**ARC-Easy — acc_norm** (standard acc_norm (primary); percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 41.2879 | -3.9141 | -5.5556 | -2.2306 | 0.0000 | 0.0002 | *† |
| LERP K2 | 42.8451 | -2.3569 | -3.7879 | -0.8838 | 0.0025 | 0.0058 | *† |
| LERP K4 | 44.6128 | -0.5892 | -2.0623 | +0.8849 | 0.4666 | 0.5938 | — |
| LERP K16 | 45.4966 | +0.2946 | -1.1785 | +1.7677 | 0.7369 | 0.7936 | — |
| LERP K32 | 44.5707 | -0.6313 | -2.0623 | +0.7997 | 0.4208 | 0.5892 | — |
| MinMax-Weight K4 | 35.6061 | -9.5960 | -11.4057 | -7.7441 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | 39.5202 | -5.6818 | -7.3232 | -4.0404 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | 41.7508 | -3.4512 | -4.9663 | -1.9360 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K64 | 44.5286 | -0.6734 | -2.1465 | +0.7997 | 0.4128 | 0.5892 | — |
| MinMax-Prob K4 | 44.0236 | -1.1785 | -2.6515 | +0.3367 | 0.1418 | 0.2837 | — |
| QRM-Weight K2 | 42.5926 | -2.6094 | -4.1246 | -1.0943 | 0.0011 | 0.0031 | *† |
| QRM-Weight K4 | 45.0337 | -0.1684 | -1.6414 | +1.3468 | 0.8655 | 0.8655 | — |
| QRM-Weight K16 | 44.9074 | -0.2946 | -1.7256 | +1.1785 | 0.7335 | 0.7936 | — |
| QRM-Prob K4 | 44.4024 | -0.7997 | -2.3148 | +0.6313 | 0.3076 | 0.5384 | — |

**ARC-Easy — acc** (standard acc; percentage points; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 45.5387 | -4.2508 | -5.9343 | -2.5253 | 0.0000 | 0.0002 | *† |
| LERP K2 | 49.4108 | -0.3788 | -1.9781 | +1.1785 | 0.6809 | 0.9611 | — |
| LERP K4 | 48.9057 | -0.8838 | -2.5673 | +0.7997 | 0.3122 | 0.6244 | — |
| LERP K16 | 49.7896 | -0.0000 | -1.6414 | +1.5993 | 1.0000 | 1.0000 | — |
| LERP K32 | 50.7155 | +0.9259 | -0.6734 | +2.5253 | 0.2679 | 0.6244 | — |
| MinMax-Weight K4 | 38.0471 | -11.7424 | -13.7626 | -9.7212 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | 43.8131 | -5.9764 | -7.7862 | -4.0825 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | 46.5067 | -3.2828 | -4.9663 | -1.5993 | 0.0003 | 0.0012 | *† |
| MinMax-Weight K64 | 49.7054 | -0.0842 | -1.6835 | +1.5572 | 0.9591 | 1.0000 | — |
| MinMax-Prob K4 | 49.3687 | -0.4209 | -2.1044 | +1.2205 | 0.6491 | 0.9611 | — |
| QRM-Weight K2 | 48.1061 | -1.6835 | -3.3249 | +0.0421 | 0.0561 | 0.1571 | — |
| QRM-Weight K4 | 49.4949 | -0.2946 | -1.9781 | +1.3899 | 0.7664 | 0.9754 | — |
| QRM-Weight K16 | 50.1684 | +0.3788 | -1.2205 | +2.0623 | 0.6865 | 0.9611 | — |
| QRM-Prob K4 | 49.7896 | +0.0000 | -1.5993 | +1.5993 | 1.0000 | 1.0000 | — |

**ARC-Easy — margin** (token-normalised gold minus best-wrong margin; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | -0.5333 | -0.0884 | -0.1235 | -0.0546 | 0.0000 | 0.0002 | *† |
| LERP K2 | -0.4767 | -0.0318 | -0.0610 | -0.0025 | 0.0311 | 0.0537 | * |
| LERP K4 | -0.4452 | -0.0003 | -0.0278 | +0.0284 | 0.9842 | 0.9842 | — |
| LERP K16 | -0.4764 | -0.0315 | -0.0603 | -0.0031 | 0.0342 | 0.0537 | * |
| LERP K32 | -0.4152 | +0.0297 | +0.0027 | +0.0571 | 0.0345 | 0.0537 | * |
| MinMax-Weight K4 | -0.7130 | -0.2681 | -0.3135 | -0.2234 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | -0.5951 | -0.1502 | -0.1879 | -0.1127 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | -0.5093 | -0.0644 | -0.0973 | -0.0322 | 0.0001 | 0.0003 | *† |
| MinMax-Weight K64 | -0.4109 | +0.0340 | +0.0039 | +0.0646 | 0.0275 | 0.0537 | * |
| MinMax-Prob K4 | -0.4467 | -0.0018 | -0.0314 | +0.0281 | 0.9068 | 0.9842 | — |
| QRM-Weight K2 | -0.5047 | -0.0598 | -0.0904 | -0.0293 | 0.0002 | 0.0006 | *† |
| QRM-Weight K4 | -0.4499 | -0.0050 | -0.0343 | +0.0235 | 0.7351 | 0.9842 | — |
| QRM-Weight K16 | -0.4474 | -0.0025 | -0.0314 | +0.0250 | 0.8600 | 0.9842 | — |
| QRM-Prob K4 | -0.4443 | +0.0006 | -0.0280 | +0.0280 | 0.9687 | 0.9842 | — |

**LAMBADA — acc** (standard harness acc; percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 8.7134 | -12.9245 | -13.9336 | -11.9154 | 0.0000 | 0.0001 | *† |
| LERP K2 | 20.2212 | -1.4167 | -2.2511 | -0.5628 | 0.0013 | 0.0023 | *† |
| LERP K4 | 21.7155 | +0.0776 | -0.7568 | +0.9509 | 0.8956 | 0.8956 | — |
| LERP K16 | 21.5408 | -0.0970 | -0.8927 | +0.7180 | 0.8487 | 0.8956 | — |
| LERP K32 | 22.3947 | +0.7568 | -0.0388 | +1.6301 | 0.0808 | 0.1131 | — |
| MinMax-Weight K4 | 3.2602 | -18.3776 | -19.4649 | -17.2909 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 10.0718 | -11.5661 | -12.5558 | -10.5958 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 17.2715 | -4.3664 | -5.2591 | -3.4543 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 21.0945 | -0.5434 | -1.3778 | +0.3105 | 0.2226 | 0.2834 | — |
| MinMax-Prob K4 | 19.2509 | -2.3870 | -3.2602 | -1.5331 | 0.0000 | 0.0001 | *† |
| QRM-Weight K2 | 17.9895 | -3.6484 | -4.5022 | -2.7557 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 19.9495 | -1.6883 | -2.5034 | -0.8733 | 0.0002 | 0.0004 | *† |
| QRM-Weight K16 | 22.4723 | +0.8345 | +0.0000 | +1.7077 | 0.0621 | 0.0966 | — |
| QRM-Prob K4 | 21.7155 | +0.0776 | -0.7568 | +0.9509 | 0.8918 | 0.8956 | — |

**LAMBADA — target_nll_corpus_token_weighted** (sum(NLL) / sum(target tokens) (primary); nats; positive = worse)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 5.6107 | +1.4063 | +1.3569 | +1.4538 | 0.0000 | 0.0001 | *† |
| LERP K2 | 4.1936 | -0.0107 | -0.0414 | +0.0188 | 0.4862 | 0.4862 | — |
| LERP K4 | 4.1106 | -0.0938 | -0.1242 | -0.0633 | 0.0000 | 0.0001 | *† |
| LERP K16 | 4.1722 | -0.0322 | -0.0607 | -0.0031 | 0.0261 | 0.0304 | *† |
| LERP K32 | 4.0824 | -0.1219 | -0.1511 | -0.0933 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K4 | 7.2106 | +3.0062 | +2.9370 | +3.0733 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 5.7081 | +1.5038 | +1.4546 | +1.5541 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 4.6832 | +0.4789 | +0.4441 | +0.5144 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 4.2615 | +0.0571 | +0.0270 | +0.0869 | 0.0001 | 0.0001 | *† |
| MinMax-Prob K4 | 4.3383 | +0.1339 | +0.1032 | +0.1657 | 0.0000 | 0.0001 | *† |
| QRM-Weight K2 | 4.6314 | +0.4271 | +0.3937 | +0.4605 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 4.3152 | +0.1109 | +0.0813 | +0.1412 | 0.0000 | 0.0001 | *† |
| QRM-Weight K16 | 4.1043 | -0.1001 | -0.1286 | -0.0724 | 0.0000 | 0.0001 | *† |
| QRM-Prob K4 | 4.2194 | +0.0151 | -0.0139 | +0.0442 | 0.3094 | 0.3332 | — |

**LAMBADA — target_nll_example_mean** (mean per-example total target NLL; nats; positive = worse)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 7.1568 | +1.7939 | +1.7348 | +1.8509 | 0.0000 | 0.0001 | *† |
| LERP K2 | 5.3492 | -0.0137 | -0.0530 | +0.0240 | 0.4862 | 0.4862 | — |
| LERP K4 | 5.2433 | -0.1196 | -0.1580 | -0.0808 | 0.0000 | 0.0001 | *† |
| LERP K16 | 5.3219 | -0.0410 | -0.0774 | -0.0040 | 0.0261 | 0.0304 | *† |
| LERP K32 | 5.2074 | -0.1555 | -0.1926 | -0.1192 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K4 | 9.1976 | +3.8346 | +3.7519 | +3.9129 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 7.2811 | +1.9182 | +1.8561 | +1.9777 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 5.9738 | +0.6109 | +0.5672 | +0.6559 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 5.4358 | +0.0729 | +0.0343 | +0.1107 | 0.0001 | 0.0001 | *† |
| MinMax-Prob K4 | 5.5338 | +0.1708 | +0.1313 | +0.2115 | 0.0000 | 0.0001 | *† |
| QRM-Weight K2 | 5.9077 | +0.5448 | +0.5015 | +0.5869 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 5.5044 | +0.1414 | +0.1036 | +0.1802 | 0.0000 | 0.0001 | *† |
| QRM-Weight K16 | 5.2353 | -0.1276 | -0.1639 | -0.0927 | 0.0000 | 0.0001 | *† |
| QRM-Prob K4 | 5.3821 | +0.0192 | -0.0179 | +0.0564 | 0.3094 | 0.3332 | — |

**LAMBADA — target_nll_token_norm_example_mean** (mean per-example NLL / target tokens; nats; positive = worse)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 6.2193 | +1.5589 | +1.5050 | +1.6101 | 0.0000 | 0.0001 | *† |
| LERP K2 | 4.6492 | -0.0112 | -0.0457 | +0.0219 | 0.5172 | 0.5570 | — |
| LERP K4 | 4.5526 | -0.1079 | -0.1417 | -0.0731 | 0.0000 | 0.0001 | *† |
| LERP K16 | 4.6175 | -0.0430 | -0.0745 | -0.0102 | 0.0093 | 0.0108 | *† |
| LERP K32 | 4.5314 | -0.1290 | -0.1612 | -0.0968 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K4 | 7.9451 | +3.2847 | +3.2124 | +3.3569 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 6.3107 | +1.6503 | +1.5946 | +1.7068 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 5.1770 | +0.5166 | +0.4769 | +0.5563 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 4.7176 | +0.0571 | +0.0225 | +0.0903 | 0.0009 | 0.0011 | *† |
| MinMax-Prob K4 | 4.8056 | +0.1452 | +0.1103 | +0.1804 | 0.0000 | 0.0001 | *† |
| QRM-Weight K2 | 5.1315 | +0.4711 | +0.4313 | +0.5088 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 4.7769 | +0.1165 | +0.0831 | +0.1514 | 0.0000 | 0.0001 | *† |
| QRM-Weight K16 | 4.5454 | -0.1150 | -0.1475 | -0.0829 | 0.0000 | 0.0001 | *† |
| QRM-Prob K4 | 4.6692 | +0.0087 | -0.0242 | +0.0413 | 0.6023 | 0.6023 | — |

**SciQ default — acc** (standard acc (primary); percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 64.6000 | -8.5000 | -11.2000 | -5.7000 | 0.0000 | 0.0002 | *† |
| LERP K2 | 75.1000 | +2.0000 | -0.3000 | +4.3000 | 0.1153 | 0.2691 | — |
| LERP K4 | 74.4000 | +1.3000 | -0.9000 | +3.5000 | 0.2941 | 0.5146 | — |
| LERP K16 | 73.5000 | +0.4000 | -1.9000 | +2.7000 | 0.8010 | 0.8010 | — |
| LERP K32 | 74.1000 | +1.0000 | -1.1025 | +3.2000 | 0.4213 | 0.5362 | — |
| MinMax-Weight K4 | 53.8000 | -19.3000 | -22.3000 | -16.1000 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | 63.8000 | -9.3000 | -12.2000 | -6.6000 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | 70.9000 | -2.2000 | -4.6000 | +0.2000 | 0.0884 | 0.2475 | — |
| MinMax-Weight K64 | 74.2000 | +1.1000 | -1.1000 | +3.4000 | 0.3863 | 0.5362 | — |
| MinMax-Prob K4 | 72.6000 | -0.5000 | -2.9000 | +1.9000 | 0.7431 | 0.8002 | — |
| QRM-Weight K2 | 71.9000 | -1.2000 | -3.5000 | +1.2000 | 0.3625 | 0.5362 | — |
| QRM-Weight K4 | 74.7000 | +1.6000 | -0.6000 | +3.9000 | 0.1881 | 0.3762 | — |
| QRM-Weight K16 | 77.0000 | +3.9000 | +1.8000 | +6.0000 | 0.0004 | 0.0014 | *† |
| QRM-Prob K4 | 72.4000 | -0.7000 | -2.9000 | +1.5000 | 0.5898 | 0.6881 | — |

**SciQ default — acc_norm** (standard acc_norm; percentage points; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 58.8000 | -7.3000 | -9.7000 | -5.0000 | 0.0000 | 0.0002 | *† |
| LERP K2 | 65.6000 | -0.5000 | -2.5000 | +1.5000 | 0.6909 | 0.8060 | — |
| LERP K4 | 67.2000 | +1.1000 | -0.9000 | +3.0000 | 0.3126 | 0.5343 | — |
| LERP K16 | 63.8000 | -2.3000 | -4.2000 | -0.2975 | 0.0341 | 0.0797 | * |
| LERP K32 | 65.6000 | -0.5000 | -2.3000 | +1.3000 | 0.6715 | 0.8060 | — |
| MinMax-Weight K4 | 49.9000 | -16.2000 | -19.1000 | -13.4000 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K16 | 56.7000 | -9.4000 | -11.8000 | -6.9000 | 0.0000 | 0.0002 | *† |
| MinMax-Weight K32 | 62.2000 | -3.9000 | -6.0000 | -1.8000 | 0.0004 | 0.0016 | *† |
| MinMax-Weight K64 | 65.1000 | -1.0000 | -2.8000 | +0.8000 | 0.3435 | 0.5343 | — |
| MinMax-Prob K4 | 65.3000 | -0.8000 | -2.9000 | +1.3000 | 0.5161 | 0.7225 | — |
| QRM-Weight K2 | 62.9000 | -3.2000 | -5.3000 | -1.2000 | 0.0032 | 0.0090 | *† |
| QRM-Weight K4 | 65.8000 | -0.3000 | -2.3000 | +1.7000 | 0.8413 | 0.9060 | — |
| QRM-Weight K16 | 66.1000 | +0.0000 | -1.8000 | +1.8000 | 1.0000 | 1.0000 | — |
| QRM-Prob K4 | 64.8000 | -1.3000 | -3.2000 | +0.6000 | 0.2346 | 0.4693 | — |

**SciQ default — margin** (token-normalised gold minus best-wrong margin; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 0.6308 | -0.3307 | -0.4119 | -0.2506 | 0.0000 | 0.0001 | *† |
| LERP K2 | 0.9992 | +0.0377 | -0.0271 | +0.1044 | 0.2574 | 0.3276 | — |
| LERP K4 | 1.0535 | +0.0920 | +0.0314 | +0.1552 | 0.0042 | 0.0084 | *† |
| LERP K16 | 0.8856 | -0.0759 | -0.1373 | -0.0134 | 0.0182 | 0.0318 | *† |
| LERP K32 | 0.9913 | +0.0298 | -0.0290 | +0.0892 | 0.3261 | 0.3804 | — |
| MinMax-Weight K4 | 0.0226 | -0.9389 | -1.0461 | -0.8335 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K16 | 0.4911 | -0.4704 | -0.5604 | -0.3854 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K32 | 0.7488 | -0.2127 | -0.2816 | -0.1436 | 0.0000 | 0.0001 | *† |
| MinMax-Weight K64 | 0.9832 | +0.0217 | -0.0410 | +0.0856 | 0.5179 | 0.5179 | — |
| MinMax-Prob K4 | 1.0050 | +0.0434 | -0.0224 | +0.1101 | 0.1990 | 0.2786 | — |
| QRM-Weight K2 | 0.7979 | -0.1636 | -0.2362 | -0.0921 | 0.0000 | 0.0001 | *† |
| QRM-Weight K4 | 0.9893 | +0.0278 | -0.0343 | +0.0927 | 0.4017 | 0.4326 | — |
| QRM-Weight K16 | 1.0579 | +0.0964 | +0.0367 | +0.1577 | 0.0029 | 0.0068 | *† |
| QRM-Prob K4 | 0.8929 | -0.0686 | -0.1311 | -0.0046 | 0.0357 | 0.0556 | * |

**SciQ no-support — acc** (standard acc (primary); percentage points; higher is better) — **primary display metric**

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 46.0000 | -5.2000 | -8.1000 | -2.5000 | 0.0003 | 0.0014 | *† |
| LERP K2 | 52.7000 | +1.5000 | -1.2000 | +4.1000 | 0.2889 | 0.5401 | — |
| LERP K4 | 52.7000 | +1.5000 | -1.0000 | +3.9000 | 0.2650 | 0.5401 | — |
| LERP K16 | 51.7000 | +0.5000 | -2.1000 | +3.1000 | 0.7649 | 0.8137 | — |
| LERP K32 | 51.6000 | +0.4000 | -2.1000 | +2.9000 | 0.8137 | 0.8137 | — |
| MinMax-Weight K4 | 33.6000 | -17.6000 | -20.8000 | -14.4000 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K16 | 42.5000 | -8.7000 | -11.6000 | -5.8000 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K32 | 49.8000 | -1.4000 | -4.2000 | +1.3000 | 0.3472 | 0.5401 | — |
| MinMax-Weight K64 | 51.8000 | +0.6000 | -2.0000 | +3.3000 | 0.7034 | 0.8137 | — |
| MinMax-Prob K4 | 52.1000 | +0.9000 | -1.8000 | +3.6000 | 0.5518 | 0.7194 | — |
| QRM-Weight K2 | 49.8000 | -1.4000 | -4.0000 | +1.2000 | 0.3242 | 0.5401 | — |
| QRM-Weight K4 | 52.1000 | +0.9000 | -1.9000 | +3.6000 | 0.5653 | 0.7194 | — |
| QRM-Weight K16 | 54.5000 | +3.3000 | +0.9000 | +5.7000 | 0.0088 | 0.0310 | *† |
| QRM-Prob K4 | 53.5000 | +2.3000 | -0.1000 | +4.9000 | 0.0951 | 0.2664 | — |

**SciQ no-support — acc_norm** (standard acc_norm; percentage points; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 43.0000 | -4.6000 | -7.0000 | -2.3000 | 0.0004 | 0.0019 | *† |
| LERP K2 | 45.5000 | -2.1000 | -4.2000 | +0.0000 | 0.0728 | 0.1699 | — |
| LERP K4 | 47.1000 | -0.5000 | -2.5000 | +1.6000 | 0.7144 | 0.9923 | — |
| LERP K16 | 46.1000 | -1.5000 | -3.6000 | +0.6000 | 0.1860 | 0.2894 | — |
| LERP K32 | 47.3000 | -0.3000 | -2.4000 | +1.8000 | 0.8506 | 0.9923 | — |
| MinMax-Weight K4 | 35.8000 | -11.8000 | -14.6000 | -9.0975 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K16 | 38.3000 | -9.3000 | -11.9000 | -6.8000 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K32 | 43.8000 | -3.8000 | -6.0000 | -1.7000 | 0.0007 | 0.0026 | *† |
| MinMax-Weight K64 | 47.4000 | -0.2000 | -2.3000 | +2.0000 | 0.9276 | 0.9989 | — |
| MinMax-Prob K4 | 46.1000 | -1.5000 | -3.6000 | +0.4000 | 0.1753 | 0.2894 | — |
| QRM-Weight K2 | 44.9000 | -2.7000 | -4.9000 | -0.6000 | 0.0200 | 0.0560 | * |
| QRM-Weight K4 | 46.0000 | -1.6000 | -3.7000 | +0.4000 | 0.1474 | 0.2894 | — |
| QRM-Weight K16 | 47.6000 | +0.0000 | -2.0000 | +1.9000 | 1.0000 | 1.0000 | — |
| QRM-Prob K4 | 47.2000 | -0.4000 | -2.5000 | +1.7000 | 0.7826 | 0.9923 | — |

**SciQ no-support — margin** (token-normalised gold minus best-wrong margin; nats; higher is better)

| Method | Value | Δ vs Exact | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | -0.2283 | -0.1242 | -0.1980 | -0.0525 | 0.0011 | 0.0051 | *† |
| LERP K2 | -0.1304 | -0.0263 | -0.0857 | +0.0363 | 0.3848 | 0.5208 | — |
| LERP K4 | -0.0785 | +0.0256 | -0.0348 | +0.0853 | 0.4092 | 0.5208 | — |
| LERP K16 | -0.1444 | -0.0403 | -0.0985 | +0.0212 | 0.1829 | 0.2846 | — |
| LERP K32 | -0.0881 | +0.0159 | -0.0391 | +0.0746 | 0.5924 | 0.6447 | — |
| MinMax-Weight K4 | -0.7489 | -0.6448 | -0.7475 | -0.5455 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K16 | -0.4529 | -0.3488 | -0.4307 | -0.2705 | 0.0000 | 0.0003 | *† |
| MinMax-Weight K32 | -0.1949 | -0.0908 | -0.1569 | -0.0248 | 0.0078 | 0.0275 | *† |
| MinMax-Weight K64 | -0.0597 | +0.0444 | -0.0143 | +0.1038 | 0.1556 | 0.2723 | — |
| MinMax-Prob K4 | -0.1013 | +0.0028 | -0.0605 | +0.0651 | 0.9324 | 0.9324 | — |
| QRM-Weight K2 | -0.1835 | -0.0794 | -0.1435 | -0.0143 | 0.0183 | 0.0427 | *† |
| QRM-Weight K4 | -0.0519 | +0.0522 | -0.0089 | +0.1147 | 0.0967 | 0.1934 | — |
| QRM-Weight K16 | -0.0319 | +0.0722 | +0.0120 | +0.1313 | 0.0155 | 0.0427 | *† |
| QRM-Prob K4 | -0.0872 | +0.0169 | -0.0441 | +0.0801 | 0.5986 | 0.6447 | — |

#### SciQ support difference-in-differences


**Scored with acc.** Exact: support 73.10%, no-support 51.20%, support gain +21.90 pp. Negative DiD means the method gains less from the supporting passage than Exact does.

| Method | support % | no-support % | gain pp | DiD pp | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 64.60 | 46.00 | +18.60 | -3.30 | -6.40 | -0.10 | 0.0516 | 0.3612 | * |
| LERP K2 | 75.10 | 52.70 | +22.40 | +0.50 | -2.50 | +3.50 | 0.7885 | 1.0000 | — |
| LERP K4 | 74.40 | 52.70 | +21.70 | -0.20 | -3.00 | +2.70 | 0.9453 | 1.0000 | — |
| LERP K16 | 73.50 | 51.70 | +21.80 | -0.10 | -3.00 | +2.80 | 1.0000 | 1.0000 | — |
| LERP K32 | 74.10 | 51.60 | +22.50 | +0.60 | -2.20 | +3.30 | 0.7261 | 1.0000 | — |
| MinMax-Weight K4 | 53.80 | 33.60 | +20.20 | -1.70 | -5.00 | +1.70 | 0.3678 | 1.0000 | — |
| MinMax-Weight K16 | 63.80 | 42.50 | +21.30 | -0.60 | -3.90 | +2.60 | 0.7641 | 1.0000 | — |
| MinMax-Weight K32 | 70.90 | 49.80 | +21.10 | -0.80 | -3.80 | +2.30 | 0.6484 | 1.0000 | — |
| MinMax-Weight K64 | 74.20 | 51.80 | +22.40 | +0.50 | -2.50 | +3.30 | 0.7863 | 1.0000 | — |
| MinMax-Prob K4 | 72.60 | 52.10 | +20.50 | -1.40 | -4.50 | +1.70 | 0.4018 | 1.0000 | — |
| QRM-Weight K2 | 71.90 | 49.80 | +22.10 | +0.20 | -2.60 | +3.10 | 0.9473 | 1.0000 | — |
| QRM-Weight K4 | 74.70 | 52.10 | +22.60 | +0.70 | -2.30 | +3.70 | 0.6906 | 1.0000 | — |
| QRM-Weight K16 | 77.00 | 54.50 | +22.50 | +0.60 | -2.10 | +3.20 | 0.7162 | 1.0000 | — |
| QRM-Prob K4 | 72.40 | 53.50 | +18.90 | -3.00 | -5.80 | -0.30 | 0.0420 | 0.3612 | * |

**Scored with acc_norm.** Exact: support 66.10%, no-support 47.60%, support gain +18.50 pp. Negative DiD means the method gains less from the supporting passage than Exact does.

| Method | support % | no-support % | gain pp | DiD pp | CI low | CI high | p | q (BH) |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LERP K1 | 58.80 | 43.00 | +15.80 | -2.70 | -5.50 | +0.10 | 0.0717 | 0.5022 | — |
| LERP K2 | 65.60 | 45.50 | +20.10 | +1.60 | -1.10 | +4.20 | 0.2691 | 0.9418 | — |
| LERP K4 | 67.20 | 47.10 | +20.10 | +1.60 | -0.80 | +4.10 | 0.2440 | 0.9418 | — |
| LERP K16 | 63.80 | 46.10 | +17.70 | -0.80 | -3.20 | +1.70 | 0.5966 | 1.0000 | — |
| LERP K32 | 65.60 | 47.30 | +18.30 | -0.20 | -2.70 | +2.30 | 0.9361 | 1.0000 | — |
| MinMax-Weight K4 | 49.90 | 35.80 | +14.10 | -4.40 | -7.70 | -1.10 | 0.0119 | 0.1673 | * |
| MinMax-Weight K16 | 56.70 | 38.30 | +18.40 | -0.10 | -3.10 | +3.00 | 1.0000 | 1.0000 | — |
| MinMax-Weight K32 | 62.20 | 43.80 | +18.40 | -0.10 | -2.60 | +2.50 | 1.0000 | 1.0000 | — |
| MinMax-Weight K64 | 65.10 | 47.40 | +17.70 | -0.80 | -3.40 | +1.70 | 0.5977 | 1.0000 | — |
| MinMax-Prob K4 | 65.30 | 46.10 | +19.20 | +0.70 | -1.70 | +3.20 | 0.6459 | 1.0000 | — |
| QRM-Weight K2 | 62.90 | 44.90 | +18.00 | -0.50 | -3.30 | +2.20 | 0.7769 | 1.0000 | — |
| QRM-Weight K4 | 65.80 | 46.00 | +19.80 | +1.30 | -1.30 | +3.90 | 0.3665 | 1.0000 | — |
| QRM-Weight K16 | 66.10 | 47.60 | +18.50 | +0.00 | -2.50 | +2.60 | 1.0000 | 1.0000 | — |
| QRM-Prob K4 | 64.80 | 47.20 | +17.60 | -0.90 | -3.30 | +1.50 | 0.5244 | 1.0000 | — |

#### BLiMP paradigms

67 paradigms, 200 pairs each. Full table: `results/layer3/stageC/tables/blimp_subtasks.csv`. Summary of each condition's paradigm distribution:

| Condition | macro % | min | median | max | paradigms below Exact |
|---|---:|---:|---:|---:|---:|
| Exact | 79.48 | 21.5 | 86.0 | 100.0 | 0 |
| LERP K1 | 74.23 | 21.5 | 80.0 | 100.0 | 46 |
| LERP K2 | 78.50 | 23.5 | 85.0 | 100.0 | 33 |
| LERP K4 | 78.37 | 24.0 | 85.5 | 100.0 | 41 |
| LERP K16 | 78.22 | 13.5 | 86.5 | 100.0 | 39 |
| LERP K32 | 77.90 | 9.5 | 82.0 | 100.0 | 38 |
| MinMax-Weight K4 | 60.08 | 0.5 | 65.0 | 100.0 | 59 |
| MinMax-Weight K16 | 69.96 | 8.5 | 76.5 | 100.0 | 56 |
| MinMax-Weight K32 | 75.72 | 21.5 | 80.5 | 100.0 | 42 |
| MinMax-Weight K64 | 77.53 | 26.0 | 83.5 | 100.0 | 40 |
| MinMax-Prob K4 | 78.28 | 21.5 | 86.0 | 100.0 | 37 |
| QRM-Weight K2 | 74.04 | 21.5 | 80.5 | 100.0 | 53 |
| QRM-Weight K4 | 76.72 | 16.5 | 81.5 | 100.0 | 44 |
| QRM-Weight K16 | 78.96 | 17.5 | 87.0 | 100.0 | 36 |
| QRM-Prob K4 | 78.43 | 18.0 | 86.0 | 100.0 | 38 |

### 7.9 What these results support, and what they do not

**Measured, and supported.**

1. **The pathological tier degrades on real benchmarks, consistently and by large margins.** Across the seven
   primary display accuracies, 27 of the 35 pathological-tier comparisons reach q < 0.05, and **every one of
   the 35 point estimates is negative**. MinMax-Weight K4 is significantly worse on all seven configurations,
   by −3.91 pp (HellaSwag `acc_norm`) to −19.40 pp (BLiMP macro); LERP K1 on six of seven, down to
   −12.92 pp on LAMBADA accuracy; MinMax-Weight K16 on six of seven. On the continuous side, the corpus
   token-weighted LAMBADA target NLL rises by +0.427 to +3.006 nats for all five pathological conditions,
   every one significant. This is the ordering the §1.1 tiers predicted from corpus ΔNLL alone.
2. **Degradation is monotone in K within each family**, on every configuration: MinMax-Weight K4 → K16 → K32
   → K64 recovers 60.08 → 69.96 → 75.72 → 77.53 on BLiMP macro and 3.26 → 10.07 → 17.27 → 21.09 on LAMBADA
   accuracy against Exact's 79.48 and 21.64; LERP K1 → K2 recovers most of the gap in one step.
3. **The good tier shows no consistent degradation.** Of its 35 primary-accuracy comparisons, 8 reach
   q < 0.05 and **they point both ways**: the tier's mean difference is +0.30 pp, its range −2.36 pp
   (LERP K2 on ARC-Easy `acc_norm`) to +3.90 pp (QRM-Weight K16 on SciQ `acc`). QRM-Weight K16 is
   significantly *better* than Exact on both SciQ arms (+3.9 and +3.3 pp) and numerically at or above Exact on
   six of seven configurations.
4. **The intermediate tier separates only where resolution is highest.** Six of its 28 primary-accuracy
   comparisons reach q < 0.05 and its mean difference is −0.23 pp. All six fall on BLiMP or LAMBADA:
   QRM-Weight K4 −2.76 pp and MinMax-Weight K64 −1.95 pp and QRM-Prob K4 −1.05 pp and MinMax-Prob K4
   −1.20 pp on BLiMP macro, MinMax-Prob K4 −2.39 pp and QRM-Weight K4 −1.69 pp on LAMBADA accuracy. On the
   other five configurations none of them separates from Exact. This is what §1.1 anticipated for this tier —
   "possibly detectable, task dependent" — and the task dependence turns out to track configuration
   resolution rather than task content.
5. **BLiMP is by far the most resolving configuration, and on it the good tier is not indistinguishable from
   Exact.** 13 of 14 methods separate from Exact on BLiMP macro accuracy and 14 of 14 on the BLiMP margin. The
   four good-tier LERP conditions all sit slightly but reliably below Exact: −0.98 (K2), −1.10 (K4),
   −1.26 (K16), −1.58 (K32) pp, all q ≤ 0.0028. Only QRM-Weight K16 does not separate (−0.51 pp, q = 0.103).
   **What this does not establish:** with one training seed per condition, a difference of one percentage
   point on BLiMP cannot be attributed to the operator rather than to the training run. The paired interval
   covers evaluation-sample variation only. The honest statement is that a ~1 pp BLiMP gap exists and is
   larger than evaluation noise, and that its cause is not identified by this design.
6. **On LAMBADA's continuous metric the good tier is slightly *better* than Exact, not merely equal.** The
   corpus token-weighted target NLL differences are LERP K2 −0.011, LERP K16 −0.032, LERP K4 −0.094,
   QRM-Weight K16 −0.100, LERP K32 −0.122 nats; four of the five exclude zero. Under §1.3's decision rule
   these are detectable differences in the direction of improvement, not equivalence. Only LERP K2's interval
   (−0.041, +0.019) lies within the declared ±0.01-nat band in the sense of covering it; the rest exceed it.
7. **Approximate operators did not reduce the benefit obtained from the supplied SciQ passage.** Exact gains
   +21.9 pp of accuracy from the support passage (73.1% with, 51.2% without). **No** difference-in-differences
   survives BH in either scoring family: the smallest q is 0.361 for `acc` and 0.167 for `acc_norm`. Three
   comparisons have bootstrap intervals excluding zero before correction — LERP K1 −3.3 pp, QRM-Prob K4
   −3.0 pp on `acc`, MinMax-Weight K4 −4.4 pp on `acc_norm` — and are reported as such, not as findings. The
   most damaged condition illustrates the point: MinMax-Weight K4 loses 19.3 pp with the passage and 17.6 pp
   without, a difference of only 1.7 pp, so its damage is broad rather than specific to supplied material.
   Exact's own +21.9 pp gain shows only that Exact benefits from the material; it is not evidence about the
   approximate operators, and none of this is attributed to any particular retrieval mechanism.

**Measured, and explicitly not concluded.**

8. **"No detectable difference" is not equivalence and not losslessness.** Where a comparison's interval covers
   zero, the statement is only that this evaluation did not detect a difference. Per §1.3 no equivalence margin
   is declared for accuracy or margin metrics at all, so no equivalence claim is available for them in
   principle, however small the point estimate.
9. **PIQA resolved almost nothing.** Only MinMax-Weight K4 (−4.13 pp) and, before correction, MinMax-Weight
   K16 survive; 1 of 14 comparisons reaches q < 0.05 on either accuracy metric. That is a property of this
   configuration at this model scale, not evidence that the operators are harmless.
10. **Margin and accuracy are not interchangeable.** The supplementary margin detects far more differences
   than accuracy does — 14/14 on BLiMP, 8/14 on HellaSwag and SciQ default, against 13, 5 and 4 for the
   corresponding accuracies. A significant margin shift is **not** reported as a change in answering accuracy;
   it is a shift in the scoring geometry that may or may not cross a decision boundary.
11. **Nothing is combined into a single score.** NLL, margin and accuracy are reported separately throughout,
    and no task or metric was dropped, added or re-weighted after seeing results.

**A stage-B assumption that the data contradicts.** §6.3 measured a paired shrinkage
`s = SD(method − Exact) / SD(Exact)` of 0.058–0.385 on natural-text corpus NLL, and §6.4 built its
"natural-text calibrated" banding on `s = 0.20`. The realised shrinkage on these downstream metrics is
**0.269 to 1.320, median 0.637**, with paired correlations of 0.259 to 0.964 — far weaker pairing than the
corpus-NLL analogy suggested. Had the stage-B banding been used as a gate it would have been materially
over-optimistic about every task's resolution. This is direct evidence for the instruction not to carry a
shrinkage measured on one data type into another, and it is why §7.5 builds every interval from realised
per-document differences instead.

**Scope limits that apply to all of the above.** One training seed per condition, so no interval or p-value
here covers training-seed variability. BLiMP is a 200-pairs-per-paradigm subset and its interval is
conditional on this fixed set of 67 paradigms, not an extrapolation to grammatical phenomena in general.
Only the native forward was run, so nothing here separates "the trained weights are worse" from "the trained
weights are matched to their own operator" — that is what the exact-forward arm, not run in this round, would
address. WinoGrande, ARC-Challenge and OpenBookQA were not extended to stage C (§7.1).

### 7.10 Reproduction

All paths are relative to `analysis_main_v1/downstream_v1/` on gpu-host-A
(`<REPO>/analysis_main_v1/downstream_v1`), mirrored locally.

```bash
python code/layer3_stageC.py --manifest     # manifest: checkpoints, hashes, operators, steps, tokens
python code/layer3_stageC.py --preflight    # 105 end-to-end checks on 2 conditions x 5 reduced configs
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python code/layer3_stageC.py --all   # the sweep (4.76 h, 1 GPU)
python code/layer3_stageC_verify.py         # 768 completeness and integrity checks
python code/layer3_stageC_stats.py          # bootstrap, permutation, BH, SciQ DiD -> tables/
python code/layer3_stageC_tables.py         # markdown rendering of the tables above
```

`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` is required, not cosmetic: with the hub reachable, the harness
revalidates all 67 BLiMP paradigm datasets against it on every task-dict build and the process stalls on
unauthenticated rate-limited requests. All datasets are already in the local cache.

**Resume and reuse.** Every configuration writes `<cfg>.csv.gz`, `<cfg>_summary.json` and `<cfg>.done`
atomically (temp file plus `os.replace`), so an interrupted run never leaves a partial file. Re-running
`--all` recomputes only what is missing or stale. A finished configuration is reused **only** when its
`.done` fingerprint matches on every component: checkpoint SHA-256, frozen training-code root, `lm-eval`
version, task id, `limit`, the SciQ no-support task YAML hash, the seed set, and the scorer version. Any
change to the checkpoint, the code, the task definition, the example list or the scoring implementation
changes the fingerprint and forces that part to be recomputed. `--force` recomputes regardless, and
`--only <cfg,…>` restricts the work to named configurations.

**Logs.** `logs/stageC.log` (sweep), `logs/stageC_stats.log` (statistics). Artefacts:
`results/layer3/stageC/manifest_stageC.json`, `results/layer3/stageC/integrity_check.json`,
`results/layer3/stageC_preflight/preflight.json`, `results/layer3/stageC/<condition>/<cfg>.csv.gz`
(per-item rows), `results/layer3/stageC/tables/*.csv`.

**Per-item schema.** Multiple-choice rows carry: `condition`, `cfg`, `task`, `subtask` (the BLiMP paradigm),
`split`, `doc_id`, `doc_hash` (hash of the original document), `n_choices`, `choice_idx`, `gold_idx`,
`is_gold`, `input_hash` (hash of the exact scored context+continuation), `cont_tok_hash` (hash of the scored
token id sequence), `cont_chars` (characters of the scored continuation, delimiter included),
`choice_chars` (characters of the raw choice, delimiter excluded — the harness `acc_norm` denominator),
`target_delimiter_chars`, `n_scored_tokens`, `raw_total_logprob`, `token_norm_logprob`, `truncated`,
`status`, `retries`, `pred_acc_idx`, `acc_correct`, `pred_accnorm_idx`, `accnorm_correct`, `harness_acc`,
`harness_acc_norm`, `margin_token_norm`. LAMBADA rows follow its own structure instead of being forced into
multiple-choice fields: `target_chars`, `n_scored_tokens`, `raw_total_logprob`, `target_nll`,
`token_norm_target_nll`, `greedy_correct`, `harness_acc`. Character counts are Python `len()` over Unicode
code points. The original document id and the scored input hash are kept separate, which is what lets the two
SciQ arms share documents while differing in every scored input.
