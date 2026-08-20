# Local voice assistant roadmap

This roadmap records the agreed priorities for the Windows WSL2 / RTX 3080
deployment. Each functional increment is committed, pushed, deployed, and
tested on that machine before the next increment begins.

## Stable baseline

- ASR: Qwen3-ASR-0.6B-hf
- LLM: Qwen3.5-4B, NF4 4-bit, non-thinking
- TTS: CosyVoice-300M-SFT, `中文女`
- Audio: half duplex, strict WebRTC VAD, low-latency WSLg playback
- CosyVoice3 remains an experimental zero-shot profile.

The stable voice was selected by same-machine listening. Playback echo must
remain absent in every later latency experiment.

## 1. Response latency

Current warm-turn time to first audio is roughly 2.6–3.6 seconds. The first
target is a median below 2.0 seconds without reducing recognition quality or
reintroducing playback echo.

Planned increments:

1. ✅ Add a repeatable end-to-end latency report over the structured session
   log.
2. ✅ Separate cold-start, first-turn, and warmed-turn measurements.
3. ⏸️ Keep the existing TTS startup warmup. Additional ASR/LLM warmup is
   deferred because it only improves the first turn and does not reduce
   steady-state latency.
4. ❌ Reject short-segment LLM-to-TTS streaming for the stable path. A
   Windows test reduced warmed median time to first audio to about 1.55
   seconds, but independent TTS segments degraded Chinese prosody and
   retriggered the microphone.
5. ❌ Reject CosyVoice SFT JIT for the stable path. On the Windows machine,
   median TTS time to first audio improved from 1.424 to 1.229 seconds, but
   model loading increased from 13.7 to 23.2 seconds and the first two
   measured syntheses took 6.68 and 8.18 seconds.

Further latency work is paused until a backend can preserve full-sentence
prosody and improve the end-to-end warmed median without startup regressions.

## 2. Tool use and web search

Tool use should be a general capability rather than a web-search special case.

Planned increments:

1. Define a tool registry with JSON schemas, timeouts, result-size limits, and
   a bounded LLM tool loop.
2. Validate the loop first with deterministic local tools such as time and
   calculator.
3. Add a configurable web-search adapter, preferring a self-hosted SearXNG
   endpoint or an explicitly configured search API.
4. Add safe page retrieval, source metadata, spoken summaries, and a clear
   offline/error fallback.
5. Keep network tools disabled unless enabled in configuration.

Acceptance requires answering a current-information question from retrieved
results, retaining source URLs in logs/output, and never looping indefinitely.

This is the next active milestone.

## 3. Optional authorized voice matching

The current SFT voice remains the fallback. Voice matching must use a clean
reference recording that the user owns or is authorized to use; reference
audio stays out of Git.

CosyVoice3 and Qwen3-TTS experiments will be compared on naturalness, Chinese
prosody, time to first audio, and stability. A Doubao-like result is optional
and must not block latency or tool-use work if an acceptable authorized
reference cannot be obtained.

## Later backlog

- Bounded conversation memory and summarization
- True barge-in with acoustic echo cancellation
- Wake word / explicit listening states
- Regression evaluation for ASR accuracy, reply quality, and audio glitches
