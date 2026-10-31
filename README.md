# jyoti

A small streaming speech codec for Nepali and English, built by Ampixa.

## Current model (2026-10-31, development candidate, not yet qualified)

- 828,547 parameters; 50 frames/s x 46 bits = 2,300 bps; first audio after 40 ms (20 ms frame + 20 ms look-ahead).
- Encoder: causal 1-D convolutions on the 16 kHz waveform (strides 2, 4, 5, 8; ~0.19 M params, ~43 MMAC/s).
- Quantizer: 4-stage residual FSQ (15 + 15 + 8 + 8 bits per frame).
- Decoder: Vocos-style blocks predicting log-magnitude + phase, inverse STFT (n_fft 640, hop 320; ~0.63 M params, ~32 MMAC/s).

## Measured (frozen evaluators, 128 utterances per pool; WER gap to the original audio, percentage points)

| model | clean | difficult | pink 10 dB | pink 5 dB | STOI |
|---|---|---|---|---|---|
| CPU-trained 32k updates | +9.7 | +25.8 | +45.9 | +37.6 | 0.86 |
| GPU continuation, noisy-input p = 0.50, 100k | +4.1 | +12.7 | +6.7 | -8.0 | 0.90 |

Gates: clean <= +5, other pools <= +10. The current best passes 4 of 5; "difficult" still fails. Nepali (SLR54 read
speech) character error on reconstruction is 26.9 % against 14.7 % for the original; that is the main open gap.

## Status

Code, training recipe and bundles will be added from the internal project once a candidate qualifies. No training
data, recordings or call audio are stored in this repository.
