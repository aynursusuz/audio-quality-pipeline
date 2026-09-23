# Audio Quality Pipeline

An auditable, cost-aware quality-control baseline for speech and audio datasets.
It supports generated and recorded audio at millions-of-clips scale: deterministic
checks run first, while neural inference is restricted to ambiguous records and a
stable sample. Every decision is reproducible from an immutable manifest,
versioned policy and model registry.

## Pipeline

```text
immutable JSONL manifest
          |
          v
L0 contract + decode ──reject──> quarantine / reason code
          |
          v
L1 streaming DSP + exact hash ──reject──> quarantine / reason code
          |
          +──accept/review JSONL (partitioned by generator, language, date)
                         |
                         v
L2 conditional GPU/CPU inference: VAD, MOS, ASR↔text, speaker/near-dup
                         |
                         v
L3 calibrated decision + stratified human audit ──> release manifest
```

`L0` and `L1` are the implemented, dependency-free baseline. `L2` is an adapter contract: run it only for all `review` items plus a deterministic sample of clean items. `L3` is where human labels calibrate model-specific thresholds; it must not be replaced by a fixed global MOS cutoff.

## Quick start

```bash
git clone https://github.com/aynursusuz/audio-quality-pipeline.git
cd audio-quality-pipeline
python -m venv .venv && source .venv/bin/activate
pip install '.[dev]'

audio-quality-pipeline \
  --manifest examples/manifest.jsonl \
  --policy config/policy.example.json \
  --dedup-db artifacts/release-2026-09.sqlite \
  --output artifacts/decisions.jsonl
```

The command above runs only L0/L1 CPU checks. Optional models are disabled unless
named explicitly. This command schedules only MOS and speaker-similarity jobs for
eligible records; it does not load either model in the L0/L1 process:

```bash
audio-quality-pipeline ... --metrics mos,speaker_similarity
```

To explicitly schedule every current optional metric, use `--metrics all`.

The decision JSONL contains compact `metric_jobs` records with the selected
metric, device and scope. Submit those records to separate CPU/GPU workers. An
empty selection is the safe default. The repository includes a private-worker
CLI for the two release-gate metrics, MOS and speaker similarity:

```bash
audio-quality-metrics \
  --decisions artifacts/decisions.jsonl \
  --output artifacts/metric-results.jsonl \
  --metrics mos,speaker_similarity \
  --utmos-checkpoint /private/model-cache/fold0_s42_best_model.pth \
  --enrollment-manifest /private/enrollments.jsonl \
  --model-cache /private/model-cache
```

The enrollment manifest is private NDJSON with `audio_id` and
`enrollment_path`. It is intentionally excluded from metric output and must
contain only consented references. Results contain model name/revision, device,
latency and scalar values—never embeddings or enrollment paths. MOS results also
record the SHA-256 of the checkpoint actually loaded.

Provision model dependencies only in the private worker image. Pin the UTMOSv2
code and checkpoint revision before a run; the reference revisions are embedded
in the result records. Do not put model caches or enrollment manifests in this
repository.

```bash
pip install 'git+https://github.com/sarulab-speech/UTMOSv2.git@cc2700db57bb83ee13dc31ebe1b868c254e15d09'
pip install '.[speaker]'
hf download sarulab-speech/UTMOSv2 fold0_s42_best_model.pth \
  --revision 506474f2b33dc77c234d668cc419be1861899cad \
  --local-dir /private/model-cache/utmosv2
```

The manifest is NDJSON. Keep it append-only and locate audio in object storage or a mounted read-only path; do not rely on directory listing order.

For an ad-hoc folder without a manifest, use stable recursive discovery:

```bash
audio-quality-pipeline \
  --input-dir /data/audio \
  --policy config/policy.example.json \
  --dedup-db artifacts/release.sqlite \
  --output artifacts/decisions.jsonl \
  --workers 8 \
  --max-in-flight 32
```

`--workers` parallelizes only CPU decoding/DSP. `--max-in-flight` caps pending
work, so memory stays bounded. Start with 4–8 workers per local SSD or storage
throughput partition; increase only after measuring I/O saturation. Exact-dedup
and JSONL writing remain single-writer to preserve deterministic output.

```json
{"audio_id":"tts-0000001","path":"/data/tts-0000001.wav","expected_text":"Merhaba dünya.","language":"tr","generator":"acme-tts","generator_version":"2026.09","voice_id":"tr_f_01","metadata":{"batch_id":"b-42"}}
```

Each decision includes status, reason codes, signal features, routing flags,
policy version and duplicate lineage. Persist decisions with the release manifest.

## L0/L1 checks implemented now

| Check | Cost | Action |
| --- | --- | --- |
| WAV decode, duration, channels, sample rate | O(bytes) | reject malformed/unsupported input |
| Peak/RMS dBFS, DC offset, clipping ratio | O(samples) | reject physical signal faults |
| 20 ms energy silence ratio | O(samples) | reject mostly empty audio |
| SHA-256 PCM payload | O(bytes) | globally reject exact duplicate |
| Deterministic routing | O(1) | send stable sample to L2/human audit |

The dependency-free decoder deliberately accepts PCM WAV only. Normalise input once during ingestion and preserve the source checksum; adding FFmpeg/FLAC adapters is a container concern, not a reason to weaken policy semantics.

## Optional metric execution

| Metric | Suggested runtime | Default device | Default scope | Output |
| --- | --- | --- | --- |
| `vad` | Silero VAD / ONNX Runtime | CPU | routed | speech coverage and silence boundaries |
| `asr` | Faster-Whisper | GPU | routed | normalized WER and language agreement |
| `dnsmos` | DNSMOS P.835 | CPU | routed | noise/artifact proxy |
| `mos` | UTMOSv2 | GPU or CPU | routed | calibrated naturalness ranking |
| `speaker_similarity` | ECAPA-TDNN | GPU or CPU | routed | consented voice match / near-dup confirmation |

`routed` means all review items plus the deterministic L2 sample. Change a
metric's policy to `review` or `non_rejected` only when its added compute is
intentional. The default device and scope live in
`config/policy.example.json`; select metrics with `--metrics`, never by default.

UTMOSv2 selects a speech segment internally. The worker seeds that selection
from `audio_id`, so retries are reproducible. Its MOS is a ranking and review
signal, not a global release threshold; calibrate it by language, generator and
duration against human labels.

Keep worker output uniform: `audio_id, model_name, model_revision, device, score,
threshold_set, decision, latency_ms, error`. Store raw scores separately from
final decisions so thresholds can be replayed. `audio-quality-metrics` executes
only `mos` and `speaker_similarity`; VAD, ASR and DNSMOS remain explicit job
contracts so they can be deployed in independently scaled workers without
coupling their dependencies to the release-gate GPU worker.

## Operating at scale

1. **Shard by immutable manifest partition** (`ingest_date/generator_version/language`), not runtime file discovery. A worker writes one decision shard.
2. **Read once, compute together.** Decode, DSP and hash are fused in one pass; avoid separate hash/silence/clipping jobs.
3. **Centralize exact dedup.** In production, write `(sha256, first_audio_id)` to a transactional key-value table or a single-writer stream—not a SQLite DB per worker. The included SQLite registry is single-worker development only.
4. **Use deterministic hash sampling** for L2 and audits. Retries then do not bias samples and spend stays predictable.
5. **Batch by duration and model.** Bucket clips, run asynchronous inference on GPUs, and cap retries.
6. **Near dedup in two steps.** Build candidates in a dedicated embedding index,
   then confirm similarity with a versioned model. Do not use a heuristic hash as evidence.
7. **Release after stratified audit.** Sample by generator, language, voice, duration, status and score decile; track false accepts/rejects against human labels.

## Calibration

Default L2 routing is `100% review + 2% stable accept sample`; human audit is `0.5% stable sample`. Replace it after measuring clip-duration distribution and desired escape rate. Release on the *estimated bad-audio escape rate by stratum* (with confidence interval), not merely the global acceptance rate.

Start in shadow mode, label a stratified sample, then set per-stratum thresholds
for the acceptable false-accept budget. A `review` state is safer than a
premature binary gate.

## Data governance

- Treat speaker embeddings and voice-match outcomes as sensitive biometric-adjacent data. Restrict access, encrypt at rest, set retention limits and use consented enrollments.
- Keep source prompt/license/consent and generator/model version on each manifest record. QC approval is not license or consent approval.
- Never overwrite a decision release. Corrections are a new policy/model version and superseding manifest.

## Repository layout

```text
config/                  versioned policy example
docs/                    research and operations
src/audio_quality_pipeline/  pipeline implementation
tests/                   deterministic unit tests
.github/workflows/       CI
```

## Research basis

Direct sources and implications are in [docs/research.md](docs/research.md). The short version: compact ONNX VAD, quantized/batched ASR, non-intrusive quality proxies and quality-prediction research all support a staged pipeline—but also argue for local human calibration.

## Publication gate

Do not create a remote or push until an owner has reviewed the policy, dependencies,
license and privacy posture. The repository must contain no audio, prompts,
embeddings, API keys, internal paths or internal metrics. Pin model revisions and
obtain a code-review approval before publication.
