# Demo-05B/C v2 — 15 秒展示流程 + 渐进时频图 + 最终流式结果

这版专门解决三个问题：

1. 原版流程太快；
2. 波形/分析图一开始就完整出现，容易让人觉得答案提前已知；
3. 最终文字缺少“最后才生成”的视觉过程。

## 现在的行为

点击“开始个性化识别”后：

```text
0s
音频读取

≈1.6s
开始生成真实音频时频图

≈4.3s
语音特征提取

≈7.0s
个性化语音库匹配

≈9.8s
时序校验 / 拒识判断

≈12.3s
进入规范中文映射

≈12.8s
才真正 POST /api/recognize/{preset_id}

≈15s
完整流程结束，等待真实 Runtime 返回

最后
规范中文逐字显示
→ TTS 播放
```

因此页面开始阶段并没有拿到识别答案。

## 重要说明

这 15 秒是“展示流程动画”，不是模型真实计算耗时。
页面顶部明确标注这一点，最终仍单独展示：

```text
模型实际耗时：xxx ms
```

这样既能把项目处理链路展示清楚，也不会把人为展示时间冒充成真实推理时间。

## 时频图

本版不再在选择 WAV 后立即把完整分析图显示出来。

选择音频后只在浏览器内预解码；
真正点击识别之后才：

```text
0% → 100%
```

从左至右逐列显示真实音频计算出的简化 STFT 时频图，同时扫描线同步移动。

## 最终结果

最终 Runtime 返回后，`canonical_text` 不会瞬间整句弹出，而会逐字符显示。

注意：这是“结果呈现动画”，不是声称模型本身是自回归 ASR。

## 安装

备份：

```powershell
Copy-Item demo\demo05_web\static demo\demo05_web\static_before_05bc_v2 -Recurse
```

将 ZIP 解压到 `papr_ssl` 根目录覆盖：

```text
demo/demo05_web/static/index.html
demo/demo05_web/static/styles.css
demo/demo05_web/static/app.js
```

启动原后端：

```powershell
python scripts\run_demo05_web.py
```

浏览器：

```text
http://127.0.0.1:7860
```

强制刷新：

```text
Ctrl + F5
```

当前仍不包含：
- 连续麦克风；
- VAD / Endpoint Detection；
- Streaming ASR；
- Agent / Coze。
