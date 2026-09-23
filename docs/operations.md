# Operating model

## Release contract

A release is a tuple:

`(input_manifest_digest, policy_version, code_commit, model_registry_snapshot, decision_manifest_digest)`.

Do not publish a mutable directory as a release. A release manifest names only accepted records and points to immutable objects. Review/reject rows remain queryable in a restricted audit table with reason codes.

## Metrics to dashboard

Track by generator version, language, voice, ingest batch and duration bucket:

- decode failure, reject, review and accept rates;
- top reason-code rates and week-over-week changes;
- exact duplicate and confirmed near-duplicate rates;
- L2 queue rate, latency, error rate and compute cost per accepted hour;
- MOS/ASR/speaker score distributions;
- audited false-accept/false-reject rates with confidence intervals; and
- human-label disagreement rate.

Alert on distribution shifts, missing partitions, new reason-code spikes or L2 completion drops—not just aggregate acceptance-rate change.

## CPU/GPU worker boundary

The L0/L1 CLI is CPU-only and never imports optional ML dependencies. It writes
`metric_jobs` only when the caller explicitly selects `--metrics`. The metric
CLI reads those jobs and executes only the requested metrics, creating one
runner per metric/device pair. For production scale, partition the jobs by
metric and device before invoking the metric CLI: use bounded CPU process pools
and duration-bucketed GPU batches. Metric failures are emitted as error rows;
the downstream decision service must route them to review/retry and never
implicitly accept them.

## Failure handling

| Condition | Handling |
| --- | --- |
| unreadable object | reject with `hard:decode_failed`; retain path and diagnostic |
| transient model/storage error | review/retry queue with bounded attempts; never accept by timeout |
| global dedup unavailable | fail partition closed or mark unreleasable; do not make local decisions |
| model revision changes | shadow-score stratified sample and recalibrate before promotion |
| high audit disagreement | freeze L2 hard rejection for affected stratum and revise rubric |

## Suggested warehouse tables

- `audio_manifest`: immutable input identity and provenance
- `qc_signal_features`: one row per audio/policy computation
- `qc_model_scores`: one row per audio/model invocation
- `qc_decisions`: versioned decision, reasons and release id
- `qc_human_labels`: blind labels, rubric version and adjudication
- `qc_releases`: immutable release tuple and approval evidence
