# H6-03A — Raw-WAV Parity Mismatch Localization

## 为什么 H6-03 失败？

当前结果：

```text
Global min cosine        = 0.9549
Temporal min frame cosine= 0.8821
Temporal shape match     = True
```

这不是浮点误差。

`shape match=True` 说明：

```text
同一条音频
同样的时间长度
同样数量的 WavLM frame
```

但 cosine 明显低于 1，说明：

> 新的 Raw-WAV 路径和最初生成 P5 cache 的路径，在数值处理上不是同一条流水线。

所以现在绝对不能通过放宽 parity threshold 来“让它 PASS”。

## H6-03A 做什么？

直接比较：

```text
Raw WAV
→ WavLM hidden_states[12..18]
```

和历史：

```text
P5 frozen frame cache [T,1024]
```

这样把 Attention / 256D Projection / H6 Head 全部绕开。

测试四种输入方式：

```text
raw
manual_znorm
hf_default
hf_explicit_mask
```

以及：

```text
hidden_states[12] ... hidden_states[18]
```

目的就是回答：

1. 是不是 waveform normalization 不一致？
2. 是不是 layer index 差一层？
3. 是不是 attention mask 行为不一致？
4. 如果这些都不是，才进一步查旧 P5 cache 生成脚本 / Transformers 实现版本。

## 运行

```powershell
Remove-Item -Recurse -Force artifacts\h6_03a_mismatch_localization -ErrorAction SilentlyContinue

python scripts/diagnose_h6_03_raw_wav_mismatch.py `
  --audio-root "D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl\datasets\public\mdsc"
```

默认检查 3 条 DEV WAV。

重点把终端最后：

```text
H6-03A AGGREGATE TOP MATCHES
diagnosis:
```

发回来。

## 当前 Gate

```text
H6-03 Raw WAV Adapter: FAIL / BLOCKED
```

在 H6-03A 找到原因之前：

```text
不要改阈值
不要打开 generic_test
不要用 Raw WAV Demo 宣称系统性能
```
