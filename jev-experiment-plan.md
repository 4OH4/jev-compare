# Jev experiments: plan for Claude Code

Three phases, run in order. Each produces its own artefact; Phases 1–2 build 
the infrastructure and test performance, Phase 3 digs into the unique aspects
of Jev.

1. **Notebook** exercising Jev's three question primitives (`Noul`, `Choice`,
   `Score`) on realistic state.
2. **Benchmark**: Jev (zero-shot / few-shot) vs three frontier LLMs
   (zero-shot / few-shot) vs a trained SVM, on three text-classification
   datasets — adapting the protocol from an existing independent
   Jev-vs-classical-ML benchmark (https://quicqdev.github.io/Jev-vs-ML/),
   swapping most of the classical ML side for frontier LLMs but keeping one
   classical reference point so the post still answers "and how does this
   compare to just training a small model on labelled data".
3. **Calibration deep-dive**: reliability diagrams and ECE for Jev vs one
   frontier LLM, on Banking77, as originally scoped.

**Language: pure Python throughout, no JavaScript.** This changes the Jev
access approach:

- **Jev via Vercel's AI Gateway, called directly over HTTP** — there's no
  Python SDK for the Gateway's evaluation-model path (`experimental_evaluate`
  is a JS-only AI SDK 7 feature), so this needs a plain `requests`/`httpx`
  POST rather than a client library. Before writing the client wrapper,
  have Claude Code check Vercel's current AI Gateway API docs for the exact
  REST endpoint and request/response schema for `typesafe-ai/jev` — do not
  assume the JS AI SDK's field names (`result.answers`,
  `result.providerMetadata.typesafe.confidence`, the `boolean`/`probability`
  vocabulary) carry over to a raw HTTP call, since those are the JS
  library's own wrapper shape, not necessarily the underlying JSON. Confirm
  the real response shape from one live call and print it in full before
  building anything that parses it.
- **Fallback**: if the Gateway doesn't document a stable plain-HTTP
  contract for evaluation models (only the AI SDK abstraction), the
  alternative is TypeSafe's own direct endpoint
  (`POST https://api.typesafe.ai/v1/systemone`), which does have a
  published, stable REST contract (`model`/`state`/`questions` in,
  `answers`/`usage` out) independent of any SDK. That needs a
  `TYPESAFE_API_KEY` from TypeSafe directly rather than the Gateway key, and
  is billed directly rather than through Vercel — a real change from
  "access via Vercel," so treat this as a fallback to flag and confirm
  before switching, not a silent substitution.
- Auth: `AI_GATEWAY_API_KEY` for the Gateway path. Pricing is the same
  $0.042/M input tokens either way.
- **LLM SDKs**: use the official Python SDKs — `anthropic`, `openai`,
  `google-genai` (Google's current Python SDK; confirm it's still the
  live one and not a renamed/successor package, since this space moves
  fast). All three support structured output / forced tool-calling
  (Anthropic tool use, OpenAI JSON schema response format, Gemini response
  schema) for the label-constrained classification calls in Phase 2.


Shared infrastructure (dataset loading, stratified sampling, label
descriptions, Jev client wrapper) should be written once and reused across
all three phases — don't rebuild it per notebook.

---

## Phase 1 — Jev mechanics notebook

Purpose: a worked example of each primitive, using one realistic piece of
state, that you can screenshot/quote in the post's opening section. Not a
benchmark — just exercising the API cleanly.

Structure:

1. **Setup**: build a small Python wrapper around the Jev HTTP call (see
   the Jev access note above — confirm the real request/response shape
   from the Gateway docs and one live call before relying on any assumed
   field names), set `AI_GATEWAY_API_KEY`, one call to confirm connectivity
   and print the model version string (`jev-1.13.0` or current) and the
   full raw response from the response body.
2. **One state, three questions** — reuse the support-ticket example shape
   from the docs (a plausible bug report / billing message), and run:
   - a `Noul` (e.g. "does this ask for a refund?")
   - a `Choice` (e.g. department routing across 4–5 options, including an
     `other` bucket)
   - a `Score` (e.g. bug severity, 3 ordered levels with concrete
     descriptions, not bare adjectives)
   Show the full response for each — `probabilities`, `confidence`,
   `usage`, latency — not just the headline field.
3. **Confidence in practice**: same state, deliberately construct one
   *ambiguous* input per primitive (an email that's neither clearly a
   refund request nor clearly not) and show confidence collapsing —
   `probabilities` spread across options rather than concentrated. This is
   the "confidence is a real signal" demo for the post.
4. **Fan-out**: one call asking 5 independent questions against the same
   state at once, vs the same 5 questions as 5 sequential calls — log
   latency and total input tokens for both, to reproduce (at small scale)
   the batching economics claim.
5. **Where it breaks**: 2–3 short demos of documented failure modes —
   a counting question ("how many times does X appear"), a hex-colour
   comparison, and a bare-adjective Score level ("moderately severe") vs the
   same Score with concrete situational criteria — to show the calibration
   collapsing on inputs it's not suited for. This makes the notebook honest
   rather than a vendor demo, and gives the post a "here's what it's not
   for" beat.

Output: `01_jev_mechanics.ipynb`, self-contained, runnable top to bottom,
each cell's output preserved so the notebook is readable without rerunning.

---

## Phase 2 — Jev vs frontier LLMs, adapted benchmark

### What's being adapted, and what's changing

The reference benchmark (QuicqDev, "Jev vs. classical ML", Sept 2026)
compared Jev zero-shot/few-shot against 11 classical ML pipelines across 8
datasets, using balanced accuracy, a train/validation/policy/test split, and
separate reporting of raw vs threshold-adjusted decisions. Read the
published methodology and limitations sections before starting —
particularly the caveats about cached zero-shot predictions not being
independent replications, and API failures being counted as incorrect
predictions — because this run should not repeat them.

**Changes for this version:**

- Classical ML pipelines (11 in the reference) → three frontier LLMs, each
  run zero-shot and few-shot, plus **one trained SVM** kept as a classical
  reference point:
  - Claude Haiku 4.5 (Anthropic)
  - the equivalent small/fast tier from OpenAI at time of running — check
    OpenAI's current model list rather than assuming a name, since this
    tier gets renamed every few months (candidates as of writing: a GPT-5.x
    Mini-class model)
  - the equivalent small/fast tier from Google at time of running (Gemini
    Flash-class)

  Record the exact model ID string used for each, in the results file —
  not just "GPT" or "Gemini" — since these are moving targets and the post
  needs to be precise about what was actually tested.

  **SVM**: TF-IDF vectoriser + linear-kernel SVM, one per dataset, trained
  on the train split. Unlike the LLMs there's no zero-shot/few-shot
  distinction — it's a single trained condition per dataset, so it gets one
  column, not two, in the results table. Pick `C` via k-fold cross-validation
  *within* the train split (no separate validation split needed just for
  this — a 5-fold CV grid search over a handful of `C` values is enough;
  don't reintroduce the reference's full validation-split machinery for one
  model). This is a deliberately unglamorous, small classical baseline —
  the point isn't to find the best possible classical pipeline (the
  reference already did that with 11 candidates and a proper validation
  split), it's to have one honest "what does a couple of lines of
  scikit-learn get you" data point sitting next to Jev and the LLMs.

- Datasets: **Banking77** (77-class intent, fine-grained), **AG News**
  (4-class topic, coarse), **IMDb** (binary sentiment) — one dataset per
  distinct kind of classification difficulty, kept from the original 8 so
  results are at least loosely comparable to the published Jev numbers on
  the same three tasks.

- **Splits**: none of the three datasets ships an official three-way
  train/validation/test split — checked directly: Banking77
  (`PolyAI/banking77`) has official train (10,003) / test (3,080) only; AG
  News and IMDb are the same shape, train/test only. There is no published
  validation split to reuse for any of them. So:
  - Use the **official published test split** (or a stratified sample of
    it — see sample size below) as the one and only evaluation set, for
    every model and every condition. Nothing in this project touches it
    until final scoring.
  - Use the **official published train split** as the source for
    everything that needs labelled examples: the few-shot pool, the SVM's
    training data, and (for IMDb) the threshold-fitting data. Any
    subdivision of train (e.g. carving out a chunk for threshold-fitting so
    it's disjoint from the few-shot examples) is *our own* stratified
    carve-out, not a published split — say so explicitly in the post rather
    than imply a validation split exists where it doesn't.
  - **Zero-shot conditions (Jev and all three LLMs) see only the test row's
    text and the shared label descriptions — never anything from train.**
    That's what zero-shot means, but it's worth stating explicitly since
    few-shot and the SVM are the only conditions in this benchmark that
    touch train at all.
- No separate validation split for hyperparameter selection — that step
  existed in the reference to select among classical-model hyperparameter
  candidates across 11 families; with a single SVM the equivalent tuning
  happens via internal cross-validation on the train split (see above), not
  a held-out split of its own. Two splits, not four:
  - **Train** (official): pool that few-shot examples are drawn from (one
    example per class, matching the reference's few-shot design), and the
    data the SVM is fit on (including its internal CV for `C`). For IMDb,
    also the source of a disjoint stratified carve-out used to fit the
    decision threshold — drawn from train, never from test, and excluding
    whichever rows get used as few-shot examples.
  - **Test** (official): the fixed, untouched holdout, shared across every
    model and shot-condition for a given dataset.

- **Sample sizes**: cost and rate limits are real constraints once you're
  paying for 3 LLMs × 2 shot-conditions, on top of Jev's 2 shot-conditions
  and the SVM, across 3 datasets. Use a stratified **1,000-row sample of the
  official test split per dataset** (at the low end of the reference's
  1,000–1,500, and smaller than Banking77's full 3,080-row test set) —
  about 13 rows per Banking77 class, enough for a stable balanced accuracy
  at this scale, and it keeps total spend manageable. (An earlier draft of
  this plan said 300. It was raised to 1,000 on 2026-09-20 because 300 gives
  only 3–4 rows per Banking77 class.) Rate limits, mostly on Jev, decide
  the run time, which is hours. The sample is drawn *from* the
  official test split, not a remix of train and test, so it stays a
  faithful (if smaller) subset of the published evaluation data. State the
  sample size plainly in the post; it's a legitimate reason results may
  not match the reference numbers exactly even on the same datasets, on top
  of the LLMs actually being a different comparison arm. Phase 3 reuses
  this same sample.

- **Independent replication, not caching**: the reference notes that
  identical cached requests were reused across their three training seeds,
  so their zero-shot rows weren't independent replications. Don't do that
  here — if you want a repeatability signal, make three genuinely
  independent API calls for zero-shot conditions (temperature > 0, no
  response caching) rather than reusing one cached response three times.
  Given cost, decide up front whether repeatability is worth the 3x spend
  or whether a single run per condition, clearly labelled as such, is
  enough for a blog post (it usually is — just don't imply repeated-seed
  rigour you didn't do).

### Metric

**Balanced accuracy** (mean per-class recall), matching the reference, so
the numbers are at least conceptually comparable to the published Jev
zero-shot/few-shot figures on Banking77, AG News and IMDb, even though your
sample sizes and run differ.

### Task definition

- Zero-shot: task instructions + class descriptions, no examples — same
  descriptions used for Jev's `Choice` criteria and for the LLM prompts, from
  one shared source file, so no model gets a better-written label set than
  another. **Zero-shot conditions read only the official test sample —
  train is never touched for these calls,** for Jev or any of the three
  LLMs.
- Few-shot: add one labelled example per class, drawn from the train split,
  identically formatted across all four models (Jev + 3 LLMs). For
  Banking77 few-shot means the prompt/state carries 77 examples — check this
  fits comfortably inside each provider's context window and inside Jev's
  ~32k-token single-question limit before running the full sample.
- LLMs: forced structured output (JSON schema / tool use) constraining the
  response to the valid label set, single-shot, no chain-of-thought — for
  the same reason as Phase 3: comparing decision quality, not reasoning
  depth, and keeping this arm methodologically consistent with the
  calibration experiment that follows. The output is a label plus a
  `confidence` (a number from 0 to 1: the model's own estimate that its
  label is right), in every condition and for all three LLMs, because
  Phase 3 needs the confidence and the table stays like for like. Whether
  the extra field changes accuracy compared with label-only output was not
  tested.
- Jev: one `Choice` question per row, criteria = the shared label
  descriptions, run at `jev-latest` (or pin the version and record it).
- SVM: fit once per dataset on the train split (TF-IDF fit on train only,
  not on test, to avoid leakage); no per-row prompting since it's not a
  language model — this is the one row in the results table that isn't
  zero-shot or few-shot, just "trained".

### Execution steps

1. Reuse or rebuild the dataset-loading and stratified-sampling utility
   from Phase 1's setup work; load the **official** Banking77, AG News,
   IMDb train and test splits (no validation split exists for any of
   these); sample 1,000 stratified rows from each official test split with a
   fixed seed; save `sample_<dataset>.csv` before running any model. The
   official train splits stay untouched at this point — they're only
   read from in steps 3 and 5.
2. Write the shared label-description file per dataset (one file per
   dataset, used by every model — see the fairness note above).
3. Build few-shot pools (one example per class) drawn from each dataset's
   official train split. For IMDb, also carve out a disjoint stratified
   chunk of train (excluding the few-shot examples) to fit the decision
   threshold in step 6 — this carve-out is ours, not a published split, so
   label it as such wherever it's referenced.
4. Run the four API-based models (Jev, Haiku, GPT-small, Gemini-small) ×
   both shot-conditions × three datasets. Save every raw response
   (`results_<model>_<shot>_<dataset>.jsonl`) including full probability
   distributions where available, tokens, and latency — not just the final
   label, since Phase 3 will want Jev's and one LLM's full distributions
   again and you don't want to rerun API calls to get them.
5. Fit the SVM once per dataset on the train split (TF-IDF + linear SVM,
   `C` chosen by 5-fold CV on train), predict on the same test split as
   everything else, and save `results_svm_<dataset>.csv` — no tokens or
   latency-per-call in the API sense, but do record wall-clock fit time and
   inference time for the cost/latency comparison, since "runs on your own
   CPU in milliseconds once trained" is itself a data point worth showing
   next to per-call API pricing.
6. Fit the IMDb decision threshold on the train-derived carve-out from
   step 3 for the binary-adjusted panel (LLMs and Jev only — the SVM's own
   decision boundary from training stands as its raw result); leave
   Banking77 and AG News as raw-only.
7. Compute balanced accuracy per model × shot-condition × dataset
   (SVM contributes one number per dataset, not two); assemble into one
   summary table, formatted similarly to the reference's panel (Jev columns
   first, SVM alongside the LLMs, one row per dataset) so a reader can
   visually compare the two posts if they want to.
8. Cost and latency table alongside the accuracy table — this is the
   number the post's audience actually cares about once accuracy is roughly
   comparable across models. The SVM's "cost" here is training compute plus
   near-zero marginal inference cost, which is worth stating explicitly
   rather than leaving a blank cell next to the API pricing columns.

Output: `02_jev_vs_llms_benchmark.ipynb` plus the saved `results_*.jsonl` /
`results_svm_*.csv` and `summary.csv` files, so numbers in the post trace
back to files, not just a chart.

---

## Phase 3 — Calibration deep-dive (Jev vs one frontier LLM, Banking77)

This is the headline analysis. Full detail was scoped previously — carried
over here with one adjustment: **reuse Phase 2's Jev and LLM zero-shot
Banking77 results** rather than resampling and rerunning from scratch.
Phase 2 records Jev's full probability distributions and each LLM's
self-reported confidence in every condition, so Phase 3 needs no
supplementary run, draws no new sample and makes no API calls. Everything
traces to Phase 2's one sample (1,000 rows, whatever `N_TEST` is set to
there).

Pick **one** LLM from Phase 2's three for this deep-dive (Claude Haiku 4.5
is the natural choice given it's the one you have most direct visibility
into) rather than doing full calibration analysis on all three — that's a
good follow-up post, not this one.

### Metrics

1. **Reliability diagram**: bucket predictions by confidence
   ([0,0.5), [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9), [0.9,1.0]); plot mean
   predicted confidence against actual accuracy per bucket, both models on
   one chart, diagonal reference line, bucket counts shown (point size or a
   table) since sparse buckets are noisy. The [0,0.5) bucket is there
   because a model on a 77-class task can report a confidence below 0.5,
   and those rows would otherwise be dropped. If many rows land in it,
   split it further (for example [0,0.3), [0.3,0.5)).
2. **ECE** (expected calibration error): weighted mean of
   |accuracy − confidence| across buckets. Headline number for the post.
3. **Brier score** as a secondary/corroborating metric.
4. **Confidence-gated accuracy curve**: sweep threshold X from 0 to 0.95
   (from 0 so that the first point is the whole sample);
   plot accuracy on the subset with confidence ≥ X, and the fraction of the
   sample that clears each threshold. This is the "what does this mean for
   a pipeline" payoff chart.
5. Cost/latency panel, reused directly from Phase 2's zero-shot Banking77
   run rather than recomputed.

### Caveats to state in the post

- The reused 1,000-row sample gives a stable top-line ECE but sparse
  low-confidence buckets for both models if both are generally accurate on
  this dataset — say so, and use the bootstrap intervals on ECE rather than
  the point estimate alone.
- Banking77 is a clean, well-specified classification task — close to
  Jev's home turf. Don't generalise the calibration result beyond this kind
  of task from one dataset.
- The LLM's self-reported confidence is a fundamentally different
  mechanism to Jev's trained calibration target (RLCD vs next-token
  introspection) — that's the whole point of the test, but say it plainly
  rather than let readers assume it's apples-to-apples by default.
- Record exact model version strings for both models (Jev's response
  includes this; capture the LLM's as well) so the post is dated against a
  specific pair of models, not "Jev vs an LLM" in the abstract.

Output: `03_calibration_analysis.ipynb`, reusing Phase 2's saved results
where possible, producing the two figures (reliability diagram,
threshold-sweep) and the summary metrics table for the post.

---

## Post structure this supports

1. Open with a couple of Phase 1 mechanics examples — what a `Choice` /
   `Score` call actually looks like, including a case where confidence
   collapsed on an ambiguous input.
2. Middle: Phase 2's headline accuracy/cost/latency table across Jev,
   three frontier LLMs (zero-shot and few-shot), and a trained SVM, on
   Banking77/AG News/IMDb — the "does the cheap fast thing hold up, and how
   does it compare to just training something small" section.
3. Close with Phase 3's calibration result as the actual takeaway — cheaper
   and faster is expected; whether the confidence number can be trusted is
   the finding worth someone's attention.
