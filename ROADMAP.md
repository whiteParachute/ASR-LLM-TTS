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
6. ✅ Record live TTS chunk gaps, minimum buffered audio, and estimated
   underflow so intermittent playback noise can be separated from artifacts
   already present in saved WAV output.

Further latency work is paused until a backend can preserve full-sentence
prosody and improve the end-to-end warmed median without startup regressions.

## 2. Tool use and web search

Tool use should be a general capability rather than a web-search special case.

Planned increments:

1. ✅ Define a tool registry with JSON schemas, timeouts, result-size limits,
   and a bounded native Qwen3.5 tool loop.
2. ✅ Validate the loop with deterministic local time and calculator tools
   on the Windows deployment.
3. ✅ Gate built-in tools by explicit intent and return trusted deterministic
   results without a second LLM round. Ordinary chat no longer carries tool
   schemas.
4. Add a configurable web-search adapter, preferring a self-hosted SearXNG
   endpoint or an explicitly configured search API.
5. Add safe page retrieval, source metadata, spoken summaries, and a clear
   offline/error fallback.
6. Keep network tools disabled unless enabled in configuration.

Acceptance requires answering a current-information question from retrieved
results, retaining source URLs in logs/output, and never looping indefinitely.

This is the next active milestone.

## 3. Agent permissions and execution safety

Action-capable tools must share one permission model before shell, browser, or
desktop control is enabled.

Planned increments:

1. Classify every tool as pure computation, read-only, reversible write, or
   destructive/external action.
2. Add per-tool capability scopes, argument validation, timeouts, output-size
   limits, cancellation, and privacy-safe audit events.
3. Add a confirmation boundary for state-changing actions. A model-generated
   tool call is never itself user authorization.
4. Add path, host, and network allowlists plus secret redaction. Destructive
   actions, privilege escalation, and credential entry stay disabled by
   default.
5. Add an emergency stop that cancels the active tool and prevents further
   actions in the current turn.

Acceptance requires proving that read-only tools can run unattended, writes
cannot escape their configured scope, and rejected or cancelled calls produce
a clear spoken response without retry loops.

## 4. Local knowledge and persistent memory

Planned increments:

1. Add `search_workspace` and `read_workspace_file` with explicit root
   allowlists, text-only limits, and no arbitrary path traversal.
2. Add bounded conversation history and summarization before exposing memory
   as tools.
3. Add `search_memory`, `remember_fact`, and `forget_memory` using local
   storage. Writes and deletion require explicit user intent and remain
   inspectable.
4. Keep credentials, raw audio, and sensitive files out of long-term memory by
   default.

## 5. Sandboxed compute and Bash

This milestone covers both structured code computation and shell access. It
must depend on the permission layer in section 3.

Planned increments:

1. Add `python_compute` for bounded data transformation and calculations in an
   isolated temporary working directory, with CPU, memory, wall-time, output,
   and file-size limits. Network access is off by default.
2. Add a read-only `run_bash` profile restricted to approved commands and
   workspace roots. It cannot use `sudo`, access credentials, spawn background
   daemons, or run destructive commands.
3. Add an explicitly confirmed write profile for narrowly scoped build,
   formatting, and file-generation tasks.
4. Capture exit status and bounded stdout/stderr as tool results, while
   redacting secrets and preventing interactive processes.

Acceptance requires terminating runaway commands, rejecting path escapes and
privilege escalation, and showing the exact command and scope before any
state-changing execution.

## 6. Browser Use

Browser control comes after read-only web search because it can interact with
logged-in sessions and mutate external state.

Planned increments:

1. Add read-only browser tools for opening a URL, inspecting page structure,
   extracting visible text, and taking screenshots.
2. Restrict navigation with domain policies, private-network protection,
   download limits, timeouts, and a dedicated download directory.
3. Add click, type, upload, and download actions behind the shared confirmation
   policy. Submitting forms, purchases, messages, or account changes always
   requires explicit confirmation at the final action.
4. Treat page content as untrusted input and defend against prompt injection;
   webpages cannot grant new tool permissions.

Acceptance requires completing a read-only browsing task with citations and a
confirmed form interaction without exposing cookies, credentials, or unrelated
tabs.

## 7. Computer Use

Full desktop control is the highest-risk tool family and is scheduled after
Browser Use is stable.

Planned increments:

1. Add screen capture, active-window inspection, and application/window
   allowlists before enabling pointer or keyboard actions.
2. Add mouse and keyboard actions with visible step logging, bounded action
   counts, confirmation checkpoints, and a user-controlled kill switch.
3. Block password entry, security-setting changes, privilege prompts,
   destructive file operations, and financial/account actions by default.
4. Add recovery for unexpected windows, focus changes, and stale screenshots;
   the agent must stop instead of guessing.

Acceptance requires a harmless local application task under an action limit,
with every interaction auditable and immediately interruptible.

## 8. Optional authorized voice matching

The current SFT voice remains the fallback. Voice matching must use a clean
reference recording that the user owns or is authorized to use; reference
audio stays out of Git.

CosyVoice3 and Qwen3-TTS experiments will be compared on naturalness, Chinese
prosody, time to first audio, and stability. A Doubao-like result is optional
and must not block latency or tool-use work if an acceptable authorized
reference cannot be obtained.

## Later backlog

- True barge-in with acoustic echo cancellation
- Wake word / explicit listening states
- Regression evaluation for ASR accuracy, reply quality, and audio glitches
