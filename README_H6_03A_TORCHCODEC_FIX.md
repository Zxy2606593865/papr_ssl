# H6-03A TorchCodec fix

The failure was not a WavLM or feature mismatch result.

Your environment has torchaudio 2.11, where `torchaudio.load()` may require the
optional `torchcodec` package. H6-03A used `torchaudio.load()` only as an
auxiliary comparison against `soundfile`.

That comparison is not required to diagnose the real P5-cache mismatch.

This patch changes the diagnostic to:

```text
soundfile loader -> always used
torchaudio loader -> optional
TorchCodec missing -> print SKIPPED and continue
```

Do **not** install TorchCodec just for this diagnostic.

Run:

```powershell
Remove-Item -Recurse -Force artifacts\h6_03a_mismatch_localization -ErrorAction SilentlyContinue

python scripts/diagnose_h6_03_raw_wav_mismatch.py `
  --audio-root "D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl\datasets\public\mdsc"
```

Send the final `H6-03A AGGREGATE TOP MATCHES` and `diagnosis`.
