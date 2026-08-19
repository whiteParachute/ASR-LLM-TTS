# Qwen3-TTS reference voice

Put an authorized, clean reference clip at `voices/reference.wav` and set
its exact transcript in the `reference_text` field of the active YAML
configuration. A 3–10 second clip with one speaker and little background
noise is a good starting point.

The baseline configuration contains the transcript of Qwen's public
voice-clone example. The stable WSL profile uses a fixed SFT voice and
needs no reference recording. The experimental
`configs/wsl_cuda_cosyvoice3.yaml` profile uses the official Chinese
sample shipped with the pinned CosyVoice runtime. To try the baseline
profile, download its matching reference clip without committing the WAV
file:

```bash
mkdir -p voices
curl --fail --location \
  --output voices/reference.wav \
  https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav
```

To use another voice, replace both the WAV and `reference_text`. Keep
`x_vector_only_mode: false` for the best cloning quality. Only use voices
you own or have permission to clone.
