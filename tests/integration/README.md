# P1-11 Real SSL Integration Tests

These tests load the three pinned real SSL checkpoints and are intentionally
excluded from the ordinary offline test path.

Default offline regression remains:

```powershell
python -m unittest discover -s tests -v
```

The real-model suite is opt-in.

Cache-only mode (recommended after P1-10):

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cpu `
  -v
```

CUDA cache-only mode:

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cuda `
  -v
```

If a checkpoint is not present in the Hugging Face cache, explicitly permit
network access:

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode network `
  --real-model-device cuda `
  -v
```

`--hf-mode cache` sets both `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, so a missing model fails instead of silently
downloading it.

Scope: P1 integration only. No training, datasets, Mean DR/SCAF optimization,
Attention DR, KD, Student, or sealed-test access.
