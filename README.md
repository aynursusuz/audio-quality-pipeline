# Audio Quality Pipeline

A fast, auditable filter for generated and recorded speech/audio datasets. It is
designed for millions of files: low-cost CPU checks run first, and model-based
metrics run only when explicitly selected.

## What runs by default

The default command uses **CPU only**. It reads PCM WAV files once and checks:

- invalid files, duration, sample rate, and channel count;
- silence, clipping, loudness, DC offset, and exact duplicates;
- deterministic routing to `accept`, `review`, or `reject`.

No model is downloaded or loaded by default.

## Quick start

```bash
git clone https://github.com/aynursusuz/audio-quality-pipeline.git
cd audio-quality-pipeline
python -m venv .venv && source .venv/bin/activate
pip install '.[dev]'

audio-quality-pipeline \
  --manifest examples/manifest.jsonl \
  --policy config/policy.example.json \
  --dedup-db artifacts/release.sqlite \
  --output artifacts/decisions.jsonl
```

For a folder of WAV files:

```bash
audio-quality-pipeline \
  --input-dir /data/audio \
  --policy config/policy.example.json \
  --dedup-db artifacts/release.sqlite \
  --output artifacts/decisions.jsonl \
  --workers 8 \
  --max-in-flight 32
```

`--workers` parallelizes CPU decoding and signal checks. `--max-in-flight` caps
memory use. Start with 4–8 workers and increase only after measuring storage
throughput.

## Metrics and hardware

Choose metrics with `--metrics`; an empty selection is the default. `routed`
means review files plus a stable sample of accepted files.

<table>
  <thead>
    <tr><th>Metric</th><th>Availability</th><th>Recommended device</th><th>Purpose</th></tr>
  </thead>
  <tbody>
    <tr><td>VAD</td><td>Scheduled job</td><td>CPU</td><td>Speech coverage and silence boundaries</td></tr>
    <tr><td>ASR</td><td>Scheduled job</td><td>GPU recommended; CPU supported</td><td>Transcript and language validation</td></tr>
    <tr><td>DNSMOS</td><td>Scheduled job</td><td>CPU</td><td>Noise and artifact proxy</td></tr>
    <tr><td>MOS (UTMOSv2)</td><td>Worker included</td><td>GPU recommended; CPU supported</td><td>Naturalness ranking</td></tr>
    <tr><td>Speaker similarity (ECAPA-TDNN)</td><td>Worker included</td><td>GPU recommended; CPU supported</td><td>Consented voice matching and duplicate confirmation</td></tr>
  </tbody>
</table>

Use only the metrics needed for a run:

```bash
# Schedule MOS and speaker-similarity jobs only.
audio-quality-pipeline ... --metrics mos,speaker_similarity
```

`--metrics all` schedules every listed metric but does not load any model in the
main process. MOS and speaker similarity run with the included metric worker;
VAD, ASR, and DNSMOS are intentionally separate worker contracts so they can
scale independently.

```bash
audio-quality-metrics \
  --decisions artifacts/decisions.jsonl \
  --output artifacts/metric-results.jsonl \
  --metrics mos,speaker_similarity \
  --utmos-checkpoint /private/model-cache/fold0_s42_best_model.pth \
  --enrollment-manifest /private/enrollments.jsonl \
  --model-cache /private/model-cache
```

MOS is a ranking/review signal, not a fixed global rejection threshold. Calibrate
thresholds per language, source, and duration with human labels. Speaker
similarity requires consented enrollment audio; embeddings and enrollment paths
are excluded from output.

## Scale and privacy

- Use an append-only JSONL manifest and partition work by date, source, and language.
- Keep raw audio, prompts, model caches, enrollment files, credentials, and internal paths out of this repository.
- Use a production key-value store or stream for global exact deduplication; the included SQLite registry is for single-worker development.

## Details

- [Metric research and sources](docs/research.md)
- [Operating guidance](docs/operations.md)
- [Example policy](config/policy.example.json)
