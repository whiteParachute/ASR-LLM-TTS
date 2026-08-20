# Modular local voice assistant (WSL2)

The current modular runtime is a fully local, half-duplex voice loop:

```text
Windows microphone -> WebRTC VAD -> Qwen3-ASR-0.6B-hf
                   -> Qwen3.5-4B (NF4 4-bit, non-thinking)
                   -> CosyVoice-300M-SFT (中文女) -> Windows speakers
```

The tested WSL2 stack uses Python 3.11, PyTorch 2.11 with CUDA 12.8,
and an NVIDIA RTX 3080 12 GB. The installers create project-local
`.venv`, `.venv-tts`, and `.venv-cosyvoice` environments; they do not
replace Ubuntu's system Python. Speech models stay isolated because
their Transformers 4.x dependencies conflict with the Qwen3.5
runtime's Transformers 5.x.

## WSL2 setup

Keep the repository on the Linux filesystem, for example:

```bash
cd ~/projects/ASR-LLM-TTS
./scripts/doctor_wsl.sh
./scripts/install_wsl_runtime.sh
./scripts/install_wsl_cosyvoice_runtime.sh
```

Download only the files required by the local model stack from
ModelScope:

```bash
source .venv/bin/activate
python scripts/download_wsl_models.py
```

The WSL configuration expects these local directories:

```text
models/Qwen3-ASR-0.6B-hf
models/Qwen3.5-4B
models/CosyVoice-300M-SFT
```

The stable WSL profile uses CosyVoice's fixed `中文女` SFT speaker, selected
after same-machine listening and latency tests. The newer CosyVoice3
zero-shot model remains available in
`configs/wsl_cuda_cosyvoice3.yaml` for authorized reference-voice
experiments. Download it separately with:

```bash
source .venv/bin/activate
python scripts/download_wsl_models.py cosyvoice3
```

The TTS model and selected speaker state run in one persistent worker;
neither is reloaded for each conversation turn. Zero-shot profiles also
cache their reference-voice prompt during worker startup.
The official CosyVoice runtime is pinned under `.runtime/CosyVoice` by
the installer instead of using the older CosyVoice 1 source retained
from the upstream project.

Start continuous microphone conversation:

```bash
./scripts/run_wsl_realtime.sh
```

Speak after the startup message. The recorder starts on speech, stops
after 500 ms of silence, processes the turn, plays the reply through
WSLg PulseAudio, and then listens for the next turn. Press `Ctrl+C` to
stop.

For an A/B comparison with the newer zero-shot CosyVoice3 model, launch
the separate profile after downloading it:

```bash
./scripts/run_wsl_realtime.sh \
  --config configs/wsl_cuda_cosyvoice3.yaml
```

This profile keeps the same ASR, LLM, VAD, and low-latency WSLg playback
settings. It writes recordings to `output/wsl-cosyvoice3/` and performance
logs to `logs/wsl-cosyvoice3/`, leaving the stable SFT profile unchanged.

The CosyVoice worker returns PCM audio chunks as they are generated. One
persistent `paplay --raw` process receives those chunks and plays them
without waiting for a complete WAV. The same chunks are also written to
one reply WAV for debugging. Non-streaming TTS providers retain the
punctuation-based fallback controlled by
`runtime.reply_chunk_max_chars`.

An experimental path can feed Qwen3.5 text into CosyVoice while the LLM
is still generating. Set `runtime.stream_llm_to_tts: true` to start the
first TTS request after roughly `runtime.first_reply_chunk_chars`
characters (or an earlier sentence boundary); later segments use
`runtime.reply_chunk_max_chars`. The WSL profiles keep this disabled by
default because short, independently synthesized segments degraded Chinese
prosody and retriggered the microphone during same-machine testing.

Run the test suite with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

The WSL-specific runtime settings are in `configs/wsl_cuda.yaml`.
Generated recordings and replies are written under `output/wsl/` and
model weights under `models/`; both are ignored by Git.

## Tool use v1

The WSL profiles enable Qwen3.5's native tool-call chat template. The first
built-in tools are `get_current_time` and `calculate`; they provide a local,
deterministic validation path before network search is added. A conservative
intent gate keeps tool schemas out of ordinary chat prompts and exposes only
the matching built-in tool. Tool execution uses a strict registry with
argument schemas, per-call timeouts, bounded result text, and a maximum number
of model/tool rounds.

Tool use can be configured without changing model code:

```yaml
tools:
  enabled: true
  max_rounds: 3
  timeout_seconds: 2.0
  max_result_chars: 2000
```

The deterministic time and calculator tools format their trusted result
directly after a successful call, avoiding a second LLM generation. Tools that
need model summarization still use the bounded follow-up loop. Performance logs
record the selected route, tool names, timings, round counts, direct-result
status, and success status, but never tool arguments or returned content. Set
`tools.enabled: false` to disable the tool layer completely.

The planned expansion is tracked in `ROADMAP.md`: web search and safe page
retrieval first, then a shared permission/confirmation layer, local knowledge
and memory, sandboxed Python/Bash, Browser Use, and finally Computer Use. Shell,
browser actions, and desktop control remain disabled until their required
scope, confirmation, audit, and emergency-stop boundaries are implemented.

## Performance observability v1

Each conversation turn now prints privacy-safe stage timings for
recording, ASR, LLM generation, streaming TTS, streaming playback,
reply preparation, time to first audio, and the full post-recording turn.
Model loading is measured separately so cold-start time is not mixed
into steady-state conversation latency.

The WSL profile appends structured events to:

```text
logs/wsl/performance.jsonl
```

Every event includes a `session_id`, `turn_id`, stage, status, and
duration. ASR and TTS events also include audio duration and real-time
factor when the audio is a readable WAV file. Transcript and reply text
are deliberately excluded from performance logs.

Streaming playback events also record `max_chunk_gap_ms`,
`min_buffer_ahead_ms`, and `estimated_underflow_ms`. These fields distinguish
audio already present in the generated WAV from glitches caused while live TTS
generation and playback are running concurrently.

Inspect the latest events after a test conversation with:

```bash
tail -n 30 logs/wsl/performance.jsonl
```

Summarize the latest successful session by stage, or exclude its first
successful turn to focus on warmed latency:

```bash
./scripts/report_wsl_performance.sh
./scripts/report_wsl_performance.sh --warmed-only
```

The derived `tts_first_chunk` row isolates the wait for the first playable
TTS audio. On the complete-reply path it is
`time_to_first_audio - response_prepare`; on the streaming-LLM path it is
`time_to_first_audio - asr - llm_first_segment`.

Measure the warmed TTS worker without microphone, ASR, LLM, playback, or
WAV-writing overhead:

```bash
./scripts/benchmark_wsl_tts.sh --runs 5
```

The benchmark uses one fixed Chinese sentence for every run and reports
model startup, time to first audio, total synthesis time, generated audio
duration, first-chunk duration, chunk count, and real-time factor. Its JSON
report is written under `logs/wsl/` so different TTS models or reference
voices can be compared on the same hardware.

For a reference-voice A/B test, override the WAV and its exact transcript
without changing the realtime configuration:

```bash
./scripts/benchmark_wsl_tts.sh \
  --config configs/wsl_cuda_cosyvoice3.yaml \
  --runs 5 \
  --reference-audio .runtime/CosyVoice/asset/zero_shot_prompt.wav \
  --reference-text '希望你以后能够做的比我还好呦。'
```

Benchmark the stable fixed Chinese SFT voice with the same sentence and
measurement code:

```bash
./scripts/benchmark_wsl_tts.sh --runs 5
```

Observability settings can be changed under `observability` in the YAML
configuration. The generated `logs/` directory is ignored by Git.

## Original project notes

# 环境配置详细教程 [B站](https://www.bilibili.com/video/BV1HucueQEJo/)

0. anaconda\ffmpeg安装
```
    网上很多教程，自行搜索
```

```
    SenseVoiceSmall模型下载：
        自动下载：设置215行 model_dir = "iic/SenseVoiceSmall"
        手动下载：https://www.modelscope.cn/models/iic/SenseVoiceSmall/files
    
    QWen模型下载：
        自动下载：设置220行 model_name = "Qwen/Qwen2.5-1.5B-Instruct"，开启科学上网，可从huggingface自动下载
        手动下载：https://www.modelscope.cn/models/ 搜索QWen，结果中下载显存可支持模型
```

1. 创建虚拟环境
```
    conda create -n chatAudio python=3.10
    conda activate chatAudio
```
2. 安装pytorch+cuda版本，本地测试2.0以上版本均可，这里安装torch=2.3.1+cuda11.8
```
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

    其它适合自己电脑的torch+cuda版本可在torch官网查找
    https://pytorch.org/get-started/previous-versions/
```

3. 简易版本安装，不使用cosyvoice时依赖项较少
```
    pip install edge-tts==6.1.17 funasr==1.1.12 ffmpeg==1.4 opencv-python==4.10.0.84 transformers==4.45.2 webrtcvad==2.0.10 qwen-vl-utils==0.0.8 pygame==2.6.1 langid==1.1.6 langdetect==1.0.9 accelerate==0.33.0 PyAudio==0.2.14

    可执行验证：
    python 13_SenceVoice_QWen2.5_edgeTTS_realTime.py
```

至此，不调用cosyvoice作为合成的交互可成功调用了。

4. cosyvoice依赖库
```
    大家反馈较多pynini、wetext安装方法：
    conda install -c conda-forge pynini=2.1.6
    pip install WeTextProcessing --no-deps
```

5. cosyvoice其它依赖项安装（如遇到权限问题导致安装失败，以管理员形式打开终端）
```
   pip install HyperPyYAML==1.2.2 modelscope==1.15.0 onnxruntime==1.19.2 openai-whisper==20231117 importlib_resources==6.4.5 sounddevice==0.5.1 matcha-tts==0.0.7.0

   可执行验证：
    python 10_SenceVoice_QWen2.5_cosyVoice.py
```

# :sparkles: 241130-updata

## 新增声纹识别功能

设置固定声纹注册语音存储目录，如目录为空则自动进入声纹注册模式。默认注册语音时长大于3秒，可自定义，一般而言时长越长，声纹效果越稳定。
声纹模型采用阿里开源的CAM++，其采用3D-Speaker中文数据训练，符合中文对话需求

## 新增自由定义唤醒词功能

使用SenceVoice的语音识别能力实现，将语音识别的汉字转为拼音进行匹配。将唤醒词/指令词设置为中文对应拼音，可自由定制。15.0_SenceVoice_kws_CAM++.py中默认为'ni hao xiao qian'，15.1_SenceVoice_kws_CAM++.py中默认为'zhan qi lai'[暗影君王实在太cool辣]

## 新增对话历史内容记忆功能

通过建立user、system历史队列实现。开启新一轮对话时，首先获取历史记忆，而后拼接新的输入指令。可自由定义最大历史长度，默认为512。

对应脚本：

无历史记忆：15.0_SenceVoice_kws_CAM++.py

有历史记忆：15.1_SenceVoice_kws_CAM++.py

[演示demo，B站] (https://www.bilibili.com/video/BV1Q6zpYpEgv)

Have fun! 😊

# :sparkles: 241123-updata

## 更新单模态自由打断语音交互

使用webrtcvad进行实时vad检测，设置一个检测时间段=0.5s，有效语音激活率=40%，每个检测chunk=20ms。也就是说500ms/20ms=25个检测段，如果25*0.4=10个片段激活，则该0.5秒为有效音，加入缓存。

可改进点：使用模型VAD，去除噪声干扰

13_SenceVoice_QWen2.5_edgeTTS_realTime.py

## 音视频多模态语音交互

基于以上逻辑，替换QWen2.5-1.5B模型为QWen2-VL-2B，可实现音视频多模态交互。模型具有两种输入格式，图片/视频

14_SenceVoice_QWen2VL_edgeTTS_realTime.py

[演示demo，B站] (https://www.bilibili.com/video/BV1uQBCYrEYL)

# :sparkles: 241027-语音交互大模型/SenceVoice-QWen2.5-TTS

## 框架

SenceVoice-QWen2.5-CosyVoice搭建

此工程主代码来于[CosyVoice] (https://github.com/FunAudioLLM/CosyVoice)

在CosyVoice基础上添加[SenceVoice] (https://github.com/modelscope/FunASR) 作为语音识别模型

添加[QWwn2.5] (https://github.com/QwenLM/Qwen2.5) 作为大语言模型进行对话理解

## 3种语音合成方法

CoosyVoice推理速度慢，严重影响对话实时性，额外添加pyttsx3和edgeTTS

EdgeTTS实验过程出现链接错误问题，升级版本至6.1.17解决，无需科学上网

All dependencies are listed in requirements.txt, the interactive inference scripts are 10/11/12_SenceVoice_QWen2.5_xxx.py. 

Have fun! 😊
