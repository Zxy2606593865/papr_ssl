# Demo-05B/C v4 — 流式处理流水 + 动态增强补丁

这一版专门解决你截图里的问题：

```text
处理流水
```

不再一次性把多行日志摆出来，而是：

```text
一条进入
→ 文字逐字出现
→ 稍停
→ 下一条进入
```

并且最终：

```text
最终识别结果已生成
```

只有在真实 Runtime 已经返回以后才入队显示。

## 新增动态效果

### 1. 处理流水真正“流式”
- 每条日志单独滑入；
- 每条日志内部逐字输出；
- 每条日志之间有自然停顿；
- 最新日志更亮，较早日志逐渐变淡；
- 底部一直保留 `LIVE 系统正在处理 ...` 动态指示；
- 最终结果日志不会与“等待 Runtime”同时出现。

### 2. 中央驾驶舱
- 当前阶段切换时轻微上浮；
- 当前阶段节点脉冲；
- RUNNING 状态灯呼吸；
- 综合进度数字平滑变化；
- 时频扫描线增加拖尾。

### 3. 智能研判
- 等待时雷达环动态旋转 / 呼吸；
- 真实结果返回后结果卡淡入；
- `canonical_text` 仍然逐字出现；
- 每个字符出现时有轻微高亮反馈。

### 4. TTS
播放标准普通话时，TTS 卡片下方出现一个简洁的动态声条。

## 安装方式

这是 **v3 驾驶舱版** 的升级补丁。

### 1. app.js
直接覆盖：

```text
demo/demo05_web/static/app.js
```

### 2. CSS
把：

```text
demo/demo05_web/static/v4-effects.css
```

里的内容 **追加到当前 v3 `styles.css` 最末尾**。

为了避免覆盖掉你已经验证过的 v3 驾驶舱完整 CSS，本补丁不重新提供整个旧 CSS。

PowerShell 可以直接：

```powershell
Get-Content demo\demo05_web\static\v4-effects.css |
  Add-Content demo\demo05_web\static\styles.css
```

然后启动：

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

## 不变的真实性边界

- 15 秒仍然是“演示流程”，不是模型真实计算时间；
- 最终 Runtime 请求仍然在流程末段才发出；
- `最终识别结果已生成` 只有真实 Runtime 返回后才显示；
- 流式文字属于结果呈现方式，不声称当前模型是自回归流式 ASR。
