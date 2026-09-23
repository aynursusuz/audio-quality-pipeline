# Research notes and design decisions

Research date: 2026-09-23. These sources informed an implementation strategy, not a claim that any one quality model is a production ground truth.

## Evidence used

| Source | Finding used here | Engineering consequence |
| --- | --- | --- |
| [DNSMOS implementation](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | DNSMOS is a non-intrusive perceptual metric for noise-suppression evaluation. | Use it as an artifact/noise proxy; not as the sole naturalness gate for TTS. |
| [UTMOS, VoiceMOS 2022](https://arxiv.org/abs/2204.02152) | MOS prediction is assessed in in-domain and OOD settings; the submitted system uses an ensemble. | Calibrate against in-house human ratings by language/model. |
| [UTMOSv2 model card](https://huggingface.co/sarulab-speech/UTMOSv2) | A model exists for naturalness MOS prediction of high-quality synthetic speech. | Viable L2 ranking model after license, coverage and calibration checks. |
| [SHEET / MOS-Bench model card](https://huggingface.co/unilight/sheet-models) | Authors state models should not replace scientific human tests, though useful for heterogeneous comparisons. | Retain human audit and use scores as signals, not labels. |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Compact VAD with ONNX support and data-cleaning use case. | Run after cheap energy checks on CPU. |
| [Silero offline sequence inference](https://github.com/snakers4/silero-vad/tree/master/examples/onnx_sequence) | Consecutive frames can be evaluated in one ONNX call, reducing long-recording overhead. | Candidate optimization for long clips; not enabled by the included runner. |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 implementation supports 8-bit quantization and batched transcription. | Run ASR only on text-bearing/review/sample clips and bucket by duration. |
| [WavLM](https://arxiv.org/abs/2110.13900) | SSL representations retain speaker, paralinguistic and content information. | Version embeddings used for candidate confirmation; never use an unversioned heuristic as final match. |
| [Common Voice on Kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice) | Includes majority-voted valid/invalid audio-text metadata. | May seed prompt-audio validation tests subject to terms; it is not synthetic-speech quality ground truth. |
| [NISQA Corpus on Kaggle](https://www.kaggle.com/datasets/pratt3000/nisqa-corpus) | Includes subjective ratings for live/simulated impairments. | Use under stated license to test artifact/noise adapters, not to certify TTS naturalness. |

## Rejected shortcuts

1. **One global MOS cutoff.** MOS predictors are domain-sensitive; a global cutoff conceals differences by language, voice and generator.
2. **ASR on every clip.** Usually the largest compute cost. Use expected text where available, all borderline clips and a stable sample; estimate escape rate through audit.
3. **Exact hashes as near-dup detection.** Hashes prove only byte/PCM identity. Use candidate blocks plus embedding/ANN confirmation for acoustic similarity.
4. **Uncalibrated speaker-similarity rejection.** Similarity changes with encoder, enrollment, channel, language and generated voice. It needs consent, validation and a review path.

## Threshold benchmark before production

Create a blinded, stratified gold set covering every deployed generator version, language, voice, duration bucket and anticipated failure mode. Collect at least two independent labels for prompt correctness, intelligibility, artifacts, naturalness and policy acceptability. Adjudicate conflicts, then report precision, recall and false-accept rate with confidence intervals per stratum. Store exact policy/model revisions with the report. Promote an L2 signal from review to reject only after it meets the agreed error budget.

## Deployment reproducibility

Pin a Git commit or model revision, model artifact checksum, runtime version, CPU/GPU image digest and policy version. “Latest” is not a quality-control dependency.
