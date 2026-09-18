# H6-03B — Raw waveform input correction

## H6-03A 已经定位到根因

历史 P5 frame cache 与以下路径几乎完全一致：

```text
raw waveform -> WavLM hidden_states[15]
mean frame cosine = 0.99999994
```

而下面三条都只有约：

```text
0.9268
```

```text
manual_znorm
HF feature extractor default
HF feature extractor + attention mask
```

所以根因不是：

```text
WavLM checkpoint 错
layer index 错
DTW 错
256D projection 错
```

而是：

```text
H6-03 Raw WAV Adapter 使用了 AutoFeatureExtractor
→ waveform 被 zero-mean / unit-variance normalization
→ 与历史 P5 pipeline 不一致
```

## 修复

Raw-WAV Adapter 改成与 P5 完全一致：

```text
WAV
→ soundfile float32
→ mono / 16 kHz
→ raw waveform tensor [1, N]
→ WavLM-large
→ hidden_states[15]
```

明确不再使用：

```text
AutoFeatureExtractor
waveform z-normalization
attention_mask（单条无 padding 推理）
```

## 下一步

覆盖：

```text
src/papr_ssl/inference/h6_raw_wav_adapter.py
```

删除旧 parity 输出：

```powershell
Remove-Item -Recurse -Force artifacts\h6_03_raw_wav_parity -ErrorAction SilentlyContinue
```

重新运行：

```powershell
python scripts/run_h6_03_raw_wav_parity.py `
  --audio-root "D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl\datasets\public\mdsc"
```

然后：

```powershell
python scripts/audit_h6_03_raw_wav_parity.py
```

期望：

```text
global min cosine      -> 接近 1
temporal frame cosine -> 接近 1
raw WAV adapter promoted: YES
```

不要放宽 parity Gate。
