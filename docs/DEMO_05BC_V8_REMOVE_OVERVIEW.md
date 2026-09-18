# Demo-05B/C v8 — 删除顶部概况条

按最新反馈，直接去掉这一整行：

```text
试点对象 王灏
已注册规范表达 5项
个性化样本 10条
系统运行状态 运行正常
演示流程 约15秒
```

不是缩小，而是完全不显示。

这样释放出来的垂直空间会全部交给主驾驶舱：

```text
本次任务 / 原始语音 / 演示控制
双图分析 / 运行处理流水
智能研判 / 应用闭环
```

## 安装

这是 v7 上的 CSS 小补丁。

将：

```text
demo/demo05_web/static/v8-remove-overview.css
```

追加到当前：

```text
demo/demo05_web/static/styles.css
```

PowerShell：

```powershell
Get-Content demo\demo05_web\static\v8-remove-overview.css |
  Add-Content demo\demo05_web\static\styles.css
```

然后：

```text
Ctrl + F5
```

即可。

不需要修改 app.js。
顶部概况节点仍然保留在 DOM 中，只是不显示，因此现有 JS 初始化逻辑不会报错。
