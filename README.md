# Audio Quality Pipeline

A fast, auditable filter for generated and recorded speech/audio datasets. It is
designed for millions of files: low-cost CPU checks run first, and model-based
metrics run only when explicitly selected.

## Two ways to use this repository

1. **Fast filter — ready now.** The default command uses **CPU only**. It reads PCM WAV files once and checks:

- invalid files, duration, sample rate, and channel count;
- silence, clipping, loudness, DC offset, and exact duplicates;
- deterministic routing to `accept`, `review`, or `reject`.

2. **Optional quality metrics.** Select only the metrics needed for a run. They use a separate command, so the fast filter stays lightweight.

No model is downloaded or loaded by the fast filter.

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

The output is `decisions.jsonl`: one decision per audio file with a status,
reason codes, signal values, and duplicate information.

## Metrics and hardware

Metrics are always opt-in. The fast filter does not load a model. The metric
command runs only the metrics explicitly named with `--metrics`.

<table>
  <thead>
    <tr><th>Metric</th><th>Device</th></tr>
  </thead>
  <tbody>
    <tr><td>Core checks</td><td>CPU</td></tr>
    <tr><td>VAD</td><td>CPU</td></tr>
    <tr><td>ASR</td><td>GPU recommended; CPU supported</td></tr>
    <tr><td>DNSMOS</td><td>CPU</td></tr>
    <tr><td>MOS (UTMOSv2)</td><td>GPU recommended; CPU supported</td></tr>
    <tr><td>Speaker similarity (ECAPA-TDNN)</td><td>GPU recommended; CPU supported</td></tr>
  </tbody>
</table>

Install and run only the metrics needed for a run:

```bash
pip install '.[vad,asr,dnsmos]'

audio-quality-pipeline ... --metrics vad,asr,dnsmos

audio-quality-metrics \
  --decisions artifacts/decisions.jsonl \
  --output artifacts/metric-results.jsonl \
  --metrics vad,asr,dnsmos \
  --model-cache /private/model-cache
```

For MOS and speaker similarity, install their dependency and provide the
required private input files:

```bash
pip install '.[speaker]'

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

ASR stores detected language and transcript length. Supply a private
`--reference-manifest` only when normalized WER is required; transcripts are not
written to the result file.

## Scale and privacy

- Use an append-only JSONL manifest and partition work by date, source, and language.
- Keep raw audio, prompts, model caches, enrollment files, credentials, and internal paths out of this repository.
- Use a production key-value store or stream for global exact deduplication; the included SQLite registry is for single-worker development.

## Details

- [Metric research and sources](docs/research.md)
- [Operating guidance](docs/operations.md)
- [Example policy](config/policy.example.json)
