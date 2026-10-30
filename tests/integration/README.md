# P1-11 真实 SSL 集成测试

这些测试会加载三个已固定版本的真实 SSL checkpoint，并有意排除在常规离线测试流程之外。

默认离线回归测试仍使用：

```powershell
python -m unittest discover -s tests -v
```

真实模型测试套件需显式启用。

仅使用缓存的模式（P1-10 后建议采用）：

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cpu `
  -v
```

CUDA 仅缓存模式：

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cuda `
  -v
```

如果 Hugging Face cache 中没有相应 checkpoint，请显式允许网络访问：

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode network `
  --real-model-device cuda `
  -v
```

`--hf-mode cache` 会同时设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；如果模型缺失，测试会直接失败，而不会静默下载。

范围：仅限 P1 集成。不涉及训练、数据集、Mean DR/SCAF 优化、Attention DR、KD、Student，也不访问已封存的 test。
