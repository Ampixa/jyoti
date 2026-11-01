# jyoti: a 2.3 kbps streaming speech codec (development candidate)

Weights, code and evaluation files: **https://huggingface.co/ampixa/jyoti**. This README mirrors the model card there.

> **Status: development candidate, not qualified.** The main model (`codec.pt`, "D3") passes 4 of our 5
> development gates. It **fails** the "difficult" gate and degrades Nepali recognition substantially
> (SLR54 CER 28.3 % vs 14.7 % on the original audio). The sealed final evaluation cohort has never been opened.
> Please don't treat these weights as a finished codec.

`jyoti` is a small neural speech codec for 16 kHz mono speech. It turns audio into a 2,300 bit/s stream and back,
in 20 ms frames, in a single CPU thread.

| | |
|---|---|
| Parameters | **828,547** (encoder 192,164 / quantizer 3,356 / decoder 633,027) |
| Payload bitrate | **2.3 kbps** (46 bits per 20 ms frame, 50 frames/s) |
| Wire format | 56 bytes of header + footer (CRC), plus 6 bytes per frame; 1,670 bytes for a 5.38 s clip (2,483 bit/s including framing) |
| Algorithmic latency | **40 ms**: the first 320 output samples arrive after 640 input samples (one 20 ms frame plus one 20 ms decoder lookahead hop). Compute and network time are additional. |
| Audio | 16 kHz, mono, speech only |
| Speed (measured) | Apple M3 Pro CPU, 1 thread: encode real-time factor 0.035, decode 0.089 (model load excluded) |
| Streaming state | 5 KB encoder / 10 KB decoder of persistent float state |

Architecture: a causal residual-convolution encoder (total stride 320) produces 64-dimensional features at 50 Hz,
quantized by a 4-stage residual finite scalar quantizer. The decoder predicts a magnitude spectrum plus an explicit
phase residual (`unit_scale_phase_residual_v1`, phase policy `after_tanh`; FFT 640, hop 320), and synthesizes audio by
inverse FFT with overlap-add. All three checkpoints in this repository share this architecture and code.

The design is inspired by the TinyCall paper ([arXiv:2609.25379](https://arxiv.org/abs/2609.25379)). This is an
**independent implementation with independently trained weights**, not the authors' code or model, and it is not
affiliated with them. Module names containing `tinycall` refer to that inspiration only.

## Models in this repository

| path | name | what |
|---|---|---|
| `codec.pt` | **D3** (main) | C plus 100,000 more updates on a wider mix: the original 100 h plus web-sourced Nepali and noisy English speech (see Training) |
| `variants/C-noise-p050/` | C | the previous main model: 100 h (LibriSpeech + SLR54) only, input-noise probability 0.50 |
| `variants/B-noise-p025/` | B | the same as C but with input-noise probability 0.25 |

## Measured quality

All numbers below were measured through the packed bitstream (encode → bytes → decode) on 128 utterances per pool.
The tables report how much worse an ASR system does on the decoded audio than on the original audio.

- **English ASR:** `facebook/wav2vec2-base-960h` (revision `22aad52d`, greedy CTC, no LM).
- **English pools:** clean = LibriSpeech dev-clean; difficult = LibriSpeech dev-other; pink 10 / 5 dB = the clean pool
  with synthetic pink noise added at the codec input.
- **Nepali, read speech (supplemental, not a gate):** [`ampixa/nepali-conformer-offline`](https://huggingface.co/ampixa/nepali-conformer-offline)
  on 128 SLR54 utterances from held-out speakers who are not in training.

The development gates are our own engineering targets, registered before training and never changed. They are not
values from the TinyCall paper.

| gate | threshold | **D3 (`codec.pt`)** | C | B |
|---|---|---:|---:|---:|
| clean WER increase | ≤ 5 pp | **+3.86** ✅ | +4.08 ✅ | +3.59 ✅ |
| difficult WER increase | ≤ 10 pp | **+13.06** ❌ | +12.70 ❌ | +11.79 ❌ |
| pink 10 dB WER increase | ≤ 10 pp | **+7.22** ✅ | +6.73 ✅ | +8.84 ✅ |
| pink 5 dB WER increase | ≤ 10 pp | **−10.54** ✅ | −8.03 ✅ | −5.29 ✅ |
| clean mean STOI | ≥ 0.85 | **0.900** ✅ | 0.902 ✅ | 0.905 ✅ |
| **gates passed** | | **4 / 5** | 4 / 5 | 4 / 5 |
| mean WER increase, 4 recognition gates | | **3.40** | 3.87 | 4.73 |

Absolute numbers for D3, codec vs original audio:

| pool | WER codec | WER original | STOI | PESQ-WB |
|---|---:|---:|---:|---:|
| clean | 6.68 % | 2.83 % | 0.900 | 2.11 |
| difficult | 21.00 % | 7.94 % | 0.885 | 1.97 |
| pink 10 dB | 24.14 % | 16.91 % | 0.728 | 1.25 |
| pink 5 dB | 40.65 % | 51.19 % | 0.639 | 1.14 |

| Nepali, read speech (SLR54) | WER | CER |
|---|---:|---:|
| original audio | 40.7 % | **14.7 %** |
| D3 | 64.0 % | **28.3 %** |
| C | 61.1 % | 26.9 % |
| B | 63.5 % | 31.1 % |

**Paired D3 vs C** (same 128 utterances per pool; 95 % intervals from 10,000 speaker-level bootstrap resamples):

| pool | change in WER, pp |
|---|---|
| clean | −0.22 [−1.05, +0.58] |
| difficult | +0.36 [−1.48, +2.21] |
| pink 10 dB | +0.49 [−1.08, +2.05] |
| pink 5 dB | **−2.51 [−4.60, −0.48]** |
| SLR54 Nepali | WER +2.91 [−2.65, +8.84]; CER +1.38 [−1.21, +3.89] |

D3 replaced C as the main model under a rule we fixed before D3 was scored. It had to pass at least as many gates as C,
have a lower mean WER increase over the four recognition gates, and have a Nepali WER change whose interval is not
entirely above zero. D3 meets all three conditions, but its English advantage is small. Only the pink 5 dB difference
is outside noise, and D3 is slightly (not significantly) worse than C on read Nepali.

### Conversational Nepali (information only)

We also scored the codecs on 128 clips (3–12 s) of conversational Nepali, each clip from a different YouTube video.
None of these videos is in any training set of these models (checked against every clip id in D3's training pool).
The references are **machine transcripts** from a separate Nepali ASR model, not human transcripts. Compare the rows
with each other, not the absolute levels.

| condition | WER | CER | WER increase vs original | CER increase vs original |
|---|---:|---:|---:|---:|
| original audio | 16.99 % | 8.93 % | | |
| **D3** | 33.03 % | 19.89 % | +16.04 | +10.96 |
| C | 37.13 % | 23.40 % | +20.13 | +14.47 |

Paired D3 vs C (10,000 bootstrap resamples over the 128 videos): WER **−4.09 pp [−6.28, −1.95]**, CER **−3.51 pp
[−5.12, −2.01]**.

**Same-domain caveat.** D3 was trained on many other YouTube Nepali videos, and C was not. D3's advantage here
mostly reflects that domain match, not general robustness. The pool was never used to choose a model. Details:
`evaluation-conversational-nepali.json`.

### Open question: is it the data or the extra training?

D3 differs from C in two ways: it has more data and it has 100,000 more updates. A matched control run, "D3C", has
the same parent, recipe, seed and number of updates, but uses only the original 100 h. It is registered and has not
yet been run. Until then, we can't say whether D3's changes come from the added data or from the longer training.

How to read these numbers:

- **Pink-noise pools.** The codec was trained with white/pink input noise, so the pink pools partly measure
  in-distribution robustness. The negative pink 5 dB value means the codec denoises somewhat. It does not mean it
  reproduces the original better.
- **Nepali.** Decoding roughly doubles the Nepali character error rate on read speech. Speech decoded by this codec is
  noticeably harder to recognise in Nepali than in English.
- Full per-pool records are in `evaluation.json` and `variants/*/evaluation.json`.

## Usage

```bash
pip install -r requirements.txt   # torch 2.14.1, numpy 2.5.4, soundfile 0.14.0 (versions used for the measurements)

# in-process round trip (see example.py)
python example.py speech_16k_mono.wav decoded.wav
python example.py speech_16k_mono.wav decoded_c.wav --checkpoint variants/C-noise-p050/codec.pt

# streaming CLI: file or raw stdin/stdout
python scripts/tinycall_live_codec_v2.py encode --checkpoint codec.pt --input speech.wav --output speech.tcl
python scripts/tinycall_live_codec_v2.py decode --checkpoint codec.pt --input speech.tcl --output decoded.wav
arecord -f S16_LE -r 16000 -c 1 -t raw | python scripts/tinycall_live_codec_v2.py encode --checkpoint codec.pt \
    --input - --input-format s16le --output - | python scripts/tinycall_live_codec_v2.py decode \
    --checkpoint codec.pt --input - --output - --output-format f32le > decoded.f32
```

In Python:

```python
from tinyasr.codec_factory_v3 import load_codec_checkpoint
from tinyasr.codec_live_v1 import LiveEncoder
from tinyasr.codec_phase_streaming_v1 import PhaseLiveDecoder

codec = load_codec_checkpoint("codec.pt")
enc = LiveEncoder(codec, model_sha256)        # model_sha256 = sha256 of codec.pt bytes
data = enc.start() + b"".join(enc.push(block) for block in blocks) + enc.finish()   # blocks: float32 tensors
dec = PhaseLiveDecoder(codec, model_sha256)
audio = dec.push(data); tail, end = dec.finish()   # finish() validates footer + CRC
```

Behaviour to know about:

- Each stream records the checkpoint's SHA-256. If you decode with a different checkpoint, decoding stops with an
  error: `ValueError: Live stream requires a different model checkpoint`. Streams made with C are not decodable with
  D3, and the reverse is also true.
- Truncated streams, trailing data, and CRC mismatches also raise errors.
- In file mode the CLI writes to `<output>.partial` and renames it only after the footer checks pass, so a failed
  decode leaves a `.partial` file behind.
- Input must already be 16 kHz mono; the code does not resample.
- `docs/tinycall-live-format-v1.md` specifies the byte format.

## Training

- **From scratch.** Training started from a random initialization. No pretrained model's weights were used. The ASR
  models above were used for evaluation only. The codec objective is reconstruction only (waveform L1, multi-resolution
  spectral and MFCC losses), so transcripts were not used for training. We used them only to exclude evaluation
  material.
- **C and B: two public corpora.**
  - [LibriSpeech](https://www.openslr.org/12/) train-clean-100 English, 55.0 h (CC BY 4.0).
  - [SLR54](https://www.openslr.org/54/) Large Nepali ASR training data, 45.0 h (CC BY-SA 4.0).
  - C and B went through a chain of from-scratch stages on growing prefixes of the same speaker-ranked selection
    (1 h → 10 h → 100 h). The final stage is a GPU continuation of 100,000 updates (batch 64 × 1.28 s crops) with a
    warmup–stable–decay LR schedule (peak 4e-4) and input-noise probability 0.50 for C (0.25 for B).
- **D3: C plus 100,000 updates on a wider mix** (same recipe as C, noise probability 0.50, its own seed). Share of
  training crops by source:
  - 50 %: the same 100 h as above (LibriSpeech 25 %, SLR54 25 %).
  - 33 %: about 12,300 h of web-sourced Nepali speech, almost all from public YouTube videos. 14 % of all crops come
    from audio that had been stored Opus-compressed.
  - 17 %: 310 h of noisy and conversational English from
    - [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (headset microphones, CC BY 4.0);
    - [The People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) (CC BY / CC BY-SA 4.0);
    - [Vystadial 2013 English](https://lindat.mff.cuni.cz/) (CC BY-SA 3.0). Vystadial contains **telephone calls
      to a research dialogue system**, recorded and published for speech research.
  - The 100,000 updates cover about 2,280 h of crops, so most of the added hours were sampled at most once.
  - Material that could overlap our evaluation data was removed before training:
    - any Nepali clip whose transcript matches an SLR54 transcript;
    - a whole sub-collection of short read prompts;
    - all English development speakers and transcripts.
  - The YouTube clips used for the conversational test come from videos that are not in this training data.
- **No Ampixa product, customer or user recordings** were used for any model here.
- **Noise augmentation** is synthetic (white/pink noise at 5–20 dB SNR, added at the encoder input only; the target
  stays clean).
- **Hardware.** Training ran with deterministic CUDA on an NVIDIA L40S.
- **Exported weights.** Each `codec.pt` holds the inference tensors only (no optimizer state). Its
  `origin_checkpoint_sha256` field names the full training checkpoint:
  - D3: `c4fe5a6c…`
  - C: `04408921…`
  - B: `392fb4ee…`

  The decoder output of each exported file was checked to be exactly equal to that of its training checkpoint.

## Limitations

- **Not qualified.**
  - It fails the difficult (dev-other) gate.
  - Quality on the sealed final cohort is unknown.
  - Streaming quality was not separately qualified.
- **Large Nepali loss.** Nepali intelligibility loss is large (see above). The conversational result relies on machine
  references and is not an independent test for D3 (see the same-domain caveat).
- **The Nepali ASR used for evaluation may have been trained on some of the same YouTube sources.**
- **Speech only.** Expect poor results on music, multiple overlapping talkers, or non-16 kHz input.
- **Not a waveform match.** The codec does not preserve the waveform. Waveform SNR against the original is low
  (about 8 dB on a dev-clean clip) because phase is synthesized. Judge it by intelligibility and listening.
- **No embedded claim.** No microcontroller or embedded performance claim is made.

## Files

| path | what |
|---|---|
| `codec.pt` | D3, step 100,000 (inference tensors, 3.3 MB) |
| `codec-config.json` | architecture, sample rate, bitrate, hop, FFT size, parameter count |
| `evaluation.json` | D3 gate table, per-pool WER/CER/STOI/PESQ, Nepali scores, evaluator identities |
| `evaluation-conversational-nepali.json` | conversational Nepali results for D3 and C (information only) |
| `variants/C-noise-p050/` | C (`codec.pt`, `codec-config.json`, `evaluation.json`); unchanged from the previous release |
| `variants/B-noise-p025/` | B (same three files); unchanged |
| `tinyasr/` | the 13 Python modules needed to load, encode, and decode (unchanged from the evaluated sources) |
| `scripts/tinycall_live_codec_v2.py` | streaming encode/decode CLI |
| `example.py` | minimal in-process round trip |
| `docs/tinycall-live-format-v1.md` | live byte-stream format: framing, latency, CRC |
| `requirements.txt` | pinned runtime versions |
| `SHA256SUMS` | SHA-256 of every file above (`sha256sum -c SHA256SUMS`) |

Not included: training data, evaluation audio, training checkpoints, or training code.

## Licence and attribution

This repository (weights and code) is released under **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**,
because the weights are trained on share-alike corpora (SLR54, Vystadial, parts of The People's Speech). If you
redistribute it or derivatives, keep the attributions below and use the same licence. The web-sourced Nepali audio
used for D3 was not published by us and is not included here.

- **LibriSpeech:** V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain
  audio books", ICASSP 2015. https://www.openslr.org/12/ (CC BY 4.0).
- **SLR54, Large Nepali ASR training data set:** O. Kjartansson, S. Sarin, K. Pipatsrisawat, M. Jansche, L. Ha,
  "Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali", SLTU 2018.
  https://www.openslr.org/54/ (CC BY-SA 4.0).
- **AMI Meeting Corpus:** AMI Consortium / University of Edinburgh. https://groups.inf.ed.ac.uk/ami/corpus/ (CC BY 4.0).
- **The People's Speech:** D. Galvez et al., "The People's Speech: A Large-Scale Diverse English Speech Recognition
  Dataset for Commercial Usage", 2021. https://huggingface.co/datasets/MLCommons/peoples_speech (CC BY / CC BY-SA 4.0).
- **Vystadial 2013:** M. Korvas et al., "Free English and Czech telephone speech corpus shared under the CC-BY-SA 3.0
  license", LREC 2014. https://lindat.mff.cuni.cz/ (CC BY-SA 3.0).

Released by [Ampixa](https://ampixa.com).
