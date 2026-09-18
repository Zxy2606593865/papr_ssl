# Demo-05B + Demo-05C 前端升级补丁

基于已经能运行的 Demo-05A，只覆盖前端三个文件，不修改模型、UserMemory、阈值或后端 API。

## Demo-05B
增加：
- 真实 WAV 波形；
- 播放指针；
- 识别扫描线；
- “音频读取→特征提取→个性化匹配→时序校验→拒识判断→规范中文映射”的过程动画；
- 单独显示真实后端 `latency_ms`。

过程动画约 2.3 秒，只负责展示真实处理链路，不伪装成逐阶段实时遥测。

## Demo-05C
增加浏览器本地中文 TTS：
- ACCEPT 后可自动播报 `canonical_text`；
- 可手动“播放标准语音 / 停止”；
- REJECT 不生成错误语音；
- 不需要云端 TTS API Key。

推荐 Edge / Chrome。中文音色取决于 Windows 已安装语音。

## 覆盖
先备份：

```powershell
Copy-Item demo\demo05_web\static demo\demo05_web\static_demo05a_backup -Recurse
```

把 ZIP 解压到 `papr_ssl` 根目录并覆盖，然后启动原来的后端：

```powershell
python scripts\run_demo05_web.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

如果仍显示旧页面：

```text
Ctrl + F5
```

当前不做连续麦克风、VAD、Endpoint Detection、Streaming、Agent/Coze。
