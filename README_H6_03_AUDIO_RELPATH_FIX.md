# H6-03 audio_relpath fix

Your MDSC manifest stores WAV paths in:

```json
"audio_relpath": "Control/dev/wav/CF0010/CF0010_0001.wav"
```

The previous parity script did not include `audio_relpath` in its accepted path keys, so it reported:

```text
manifest_rows_missing_path_field: 15390
```

This patch fixes only path parsing.

After copying the patched script into `scripts/`, run:

```powershell
python scripts/run_h6_03_raw_wav_parity.py --audio-root "D:\PATH\TO\MDSC_ROOT"
```

`MDSC_ROOT` must be the directory for which this exists:

```text
D:\PATH\TO\MDSC_ROOT\Control\dev\wav\CF0010\CF0010_0001.wav
```

Then:

```powershell
python scripts/audit_h6_03_raw_wav_parity.py
```
