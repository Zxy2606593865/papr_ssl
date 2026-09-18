# Demo-05B/C v6 — 双图分析 + 大型处理流水 + 交互控制

按当前反馈完成以下调整。

## 已删除

从页面中删除：

```text
已注册规范表达 模块
试点说明 模块
```

注意：顶部仍保留“已注册规范表达数量”这个业务指标，因为它属于项目运行数据，而不是独立模块。

## 双图同时展示

中央 AI 语音分析中心默认同时显示：

### 时频音谱图

```text
时间 × 频率 × 能量
```

来自真实 WAV 的简化 STFT。

### 实时频谱图

```text
频率 × 当前幅度
```

随当前分析帧动态变化，并显示：

```text
主峰频率
当前能量
分析帧
```

默认是“双图”模式，也可以手动切换：

```text
双图
音谱图
频谱图
```

## 运行处理流水加大

`运行处理流水` 高度显著增加，并采用：

```text
时间 | 处理事件 | 状态
```

三列结构。

现在保留更多历史流水，不再只显示最后几行。

新增：

```text
暂停滚动 / 继续滚动
清空流水
导出流水
```

导出为本地 TXT。

## 新增可交互按钮

左侧：

```text
播放原始语音
重新识别
启动智能识别
```

演示控制：

```text
清空流水
导出流水
复制结果
全屏展示
```

分析视图：

```text
双图
音谱图
频谱图
```

右侧：

```text
播放标准语音
停止
```

## 动态逻辑

仍保持：

- 约 15 秒展示流程；
- 非匀速阶段；
- 流水逐条出现；
- 真实 Runtime 请求在末段发出；
- Runtime 返回后才显示最终中文；
- TTS 仍使用浏览器本地中文语音。

## 安装

这是完整静态替换包。

建议备份：

```powershell
Copy-Item demo\demo05_web\static `
  demo\demo05_web\static_before_v6 `
  -Recurse
```

将 ZIP 解压到 `papr_ssl` 根目录，覆盖：

```text
demo/demo05_web/static/index.html
demo/demo05_web/static/styles.css
demo/demo05_web/static/app.js
```

启动：

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
