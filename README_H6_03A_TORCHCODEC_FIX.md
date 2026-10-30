# H6-03A TorchCodec 修复

此次失败并非 WavLM 或 feature 不匹配导致。

环境中的 torchaudio 版本为 2.11，此版本的 `torchaudio.load()` 可能需要可选 package `torchcodec`。H6-03A 仅使用 `torchaudio.load()` 与 `soundfile` 做辅助比较。

诊断真实 P5-cache 不匹配并不需要此项比较。

此补丁将诊断流程调整为：

```text
soundfile loader -> always used
torchaudio loader -> optional
TorchCodec missing -> print SKIPPED and continue
```

不要仅为此诊断安装 TorchCodec。

运行：

```powershell
Remove-Item -Recurse -Force artifacts\h6_03a_mismatch_localization -ErrorAction SilentlyContinue

python scripts/diagnose_h6_03_raw_wav_mismatch.py `
  --audio-root "D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl\datasets\public\mdsc"
```

请提供最终的 `H6-03A AGGREGATE TOP MATCHES` 和 `diagnosis`。
