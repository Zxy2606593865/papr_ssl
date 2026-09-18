# H6-03 — Raw WAV Adapter

## 当前在干什么？

H6-02 已经证明：

```text
Enrollment Memory
+ frozen features
→ Global Prototype
→ DTW
→ C/W/U Head
→ ACCEPT / CONFIRM / REJECT
```

整个 Runtime 软件链路已经能工作。

H6-03 只增加最后一块：

```text
真实 WAV
→ frozen feature
```

也就是：

```text
WAV
→ 16 kHz mono
→ WavLM-large hidden_states[15]
→ Attention DR 256D
├─ Global 256D
└─ Temporal 256D
→ H6-02 Runtime
```

## 为什么不能直接拿新 WAV Demo？

因为我们以前的研究结果全部建立在 P5/H4 保存的 feature 上。

如果 Raw WAV 新路径的预处理、WavLM layer、归一化或 Projection 有任何不同，
即使 H6 Runtime 完全正确，最终结果也会漂移。

所以 H6-03 第一件事不是“展示识别”，而是做 **parity check（特征一致性验证）**：

```text
同一条 DEV WAV

旧路径：
历史 cached feature

新路径：
原始 WAV → WavLM → frozen feature

二者必须几乎一致。
```

只有 parity PASS，Raw WAV Adapter 才能 PROMOTE。

## 为什么 H6-02 Demo 是 100% 不能当新结果？

H6-02 Audit 的：

```text
Known CA = 1.0
Unknown Reject = 1.0
```

只来自：

```text
一个 DEV speaker（CF0010）
一个固定 2-shot demo episode
cached features
```

它证明 Runtime 实现没有把逻辑串错。

它不是新的总体 benchmark，也不能替代 H6-01 的 4-speaker / 20-repeat 结果。

## 文件

```text
src/papr_ssl/inference/h6_raw_wav_adapter.py
scripts/run_h6_03_raw_wav_parity.py
scripts/audit_h6_03_raw_wav_parity.py
scripts/predict_h6_raw_wav.py
```

## 第一步：Parity

```powershell
python scripts/run_h6_03_raw_wav_parity.py
python scripts/audit_h6_03_raw_wav_parity.py
```

默认比较 8 条 DEV WAV。

要求：

```text
Global cosine >= 0.99999
Temporal per-frame cosine >= 0.9999
Temporal shape 完全一致
```

因为 H4 temporal reference 是 float16，所以 temporal 主要看 cosine，而不是要求逐元素 0 误差。

## 如果找不到 WAV

如果脚本提示：

```text
Only N DEV WAVs could be resolved
```

会生成：

```text
artifacts/h6_03_raw_wav_parity/path_diagnostic.json
```

如果 manifest 里保存的是相对路径，重新运行：

```powershell
python scripts/run_h6_03_raw_wav_parity.py --audio-root "你的MDSC数据根目录"
```

不要修改模型。

## Parity PASS 后

才可以：

```powershell
python scripts/predict_h6_raw_wav.py "path\to\query.wav"
```

它会使用 H6-02 保存的 demo user memory，输出：

```json
{
  "status": "ACCEPT / CONFIRM / REJECT",
  "intent_id": "...",
  "canonical_text": "...",
  "decision_scores": {
    "C": 0.0,
    "W": 0.0,
    "U": 0.0
  }
}
```

注意 decision scores 仍不是 calibrated probability。

## 还差什么才是最终真实用户 Demo？

H6-03 先验证“Raw WAV Query”。

再下一步才是：

```text
真实用户 WAV Enrollment
→ memory
→ 真实 Query WAV
→ ACCEPT / CONFIRM / REJECT
```

即完整的 Raw-WAV enrollment + query CLI/API。
