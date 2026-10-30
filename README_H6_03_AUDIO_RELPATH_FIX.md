# H6-03 audio_relpath 修复

MDSC manifest 将 WAV 路径存储在：

```json
"audio_relpath": "Control/dev/wav/CF0010/CF0010_0001.wav"
```

之前的 parity script 未将 `audio_relpath` 纳入可接受的路径键，因此报告：

```text
manifest_rows_missing_path_field: 15390
```

此补丁仅修复路径解析。

将修复后的 script 复制到 `scripts/` 后，运行：

```powershell
python scripts/run_h6_03_raw_wav_parity.py --audio-root "D:\PATH\TO\MDSC_ROOT"
```

`MDSC_ROOT` 必须是包含以下文件的目录：

```text
D:\PATH\TO\MDSC_ROOT\Control\dev\wav\CF0010\CF0010_0001.wav
```

然后运行：

```powershell
python scripts/audit_h6_03_raw_wav_parity.py
```
