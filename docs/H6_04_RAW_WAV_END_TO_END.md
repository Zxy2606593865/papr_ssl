# H6-04 — Full Raw-WAV Enrollment + Query

## 当前在干什么？

H6-03 已经证明：

```text
Raw WAV
→ WavLM[15]
→ Global 256D / Temporal 256D
```

和历史科研特征几乎完全一致。

H6-04 再往前一步：

```text
Raw WAV Enrollment
→ UserMemory
→ Raw WAV Query
→ ACCEPT / CONFIRM / REJECT
```

也就是第一次验证：

> 注册和查询两边都不用 cached feature，完整走真实 WAV。

## 为什么还要和 cached runtime 比？

因为 H6-02 已经验证过 Runtime 逻辑。

所以 H6-04 用同一个 DEV demo episode，同时跑：

```text
Raw WAV Runtime
vs
Cached Feature Runtime
```

要求每一条 query：

```text
status 完全一致
intent_id 完全一致
```

这不是新 benchmark，只是部署一致性测试。

## 运行端到端 Smoke Test

```powershell
python scripts/run_h6_04_raw_wav_end_to_end.py `
  --audio-root "D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl\datasets\public\mdsc"
```

然后：

```powershell
python scripts/audit_h6_04_raw_wav_end_to_end.py
```

目标：

```text
status match rate = 1.0
intent match rate = 1.0
AUDIT STATUS = PASS
```

## 真正注册一个用户

准备 JSON：

```text
configs/demo_enrollment.json
```

参考：

```text
configs/h6_04_enrollment.example.json
```

然后：

```powershell
python scripts/enroll_h6_user_from_wavs.py `
  configs/demo_enrollment.json `
  artifacts/demo_users/student_001.pt
```

查询一条 WAV：

```powershell
python scripts/predict_h6_user_wav.py `
  artifacts/demo_users/student_001.pt `
  "path\to\query.wav"
```

输出就是：

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

注意：

```text
decision_scores 仍不是 calibrated probability
```

## H6-04 PASS 后

模型侧闭环就完成了。

后面主要是软件工程：

```text
CLI / FastAPI
→ Agent
→ 前端
```

以及最后冻结协议后才打开 generic_test。
