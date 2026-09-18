# Demo-05B/C v9 — 浅色政务业务系统版

v9 保留 v6~v8 已确认的功能和一屏结构，但彻底调整视觉语言。

## 保留
- 一屏左 / 中 / 右布局
- 双图：时频音谱图 + 实时频谱图
- 大型运行处理流水
- 约 15 秒非匀速展示流程
- 真实 Runtime 在末段请求
- 最终规范中文逐字显示
- TTS
- 应用闭环
- 播放 / 重跑 / 导出 / 复制 / 全屏 / 双图切换等交互

## 改变
### 从“智慧城市大屏”改为“政务业务系统”
删除或弱化：
- 黑色全局背景
- 青色发光边框
- HUD 四角装饰
- 雷达扫描
- 大量英文副标题
- 终端风状态码
- 科技网格背景

改为：
- 浅灰页面背景
- 白色业务卡片
- 政务蓝主色
- 扁平按钮
- 表单式任务信息
- 浅色处理流水表格
- 中文业务状态标签

### 运行处理流水
现在使用：
```text
时间 | 处理事件 | 状态
```
状态中文化：
```text
已接收 / 读取中 / 已检查 / 分析中 / 提取中 / 加载中 /
匹配中 / 校验中 / 判断中 / 等待 / 已完成
```

## 安装
这是完整静态替换包。

先备份：
```powershell
Copy-Item demo\demo05_web\static `
  demo\demo05_web\static_before_v9 `
  -Recurse
```

解压 ZIP 到 papr_ssl 根目录并覆盖：
```text
demo/demo05_web/static/index.html
demo/demo05_web/static/styles.css
demo/demo05_web/static/app.js
```

启动：
```powershell
python scripts\run_demo05_web.py
```

然后浏览器：
```text
Ctrl + F5
F11
```

建议现场使用 1920×1080、浏览器缩放 100%。
