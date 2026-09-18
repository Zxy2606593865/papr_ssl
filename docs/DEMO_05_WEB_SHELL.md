# Demo-05 — 体制内展示风格前端壳子

## 产品呈现逻辑

前端不展示内部分类标签。

```text
王灏真实原始语音
        ↓
PAPR-SSL Personalized Runtime
        ↓
内部 intent_id
        ↓
canonical_text 映射
        ↓
前端展示规范中文
        ↓
未来发送至智能学伴 / Agent
```

例如内部：

```text
intent_id = Self-introduction
```

前端只显示：

```text
大家好，我叫王灏
```

## 视觉方向

第一版采用：
- 白底；
- 政务蓝主色；
- 顶部少量红色强调；
- 标准卡片和表格式信息；
- 大字号中文结果；
- 不使用霓虹、粒子、黑色科技风；
- 不使用政府徽标或虚构官方身份。

## Step 1：安装轻量 Demo 依赖

在现有 `papr_ssl` conda 环境：

```powershell
pip install -r demo\demo05_web\requirements-demo.txt
```

不会替换现有 torch/transformers。

## Step 2：准备演示 WAV

从 `papr_ssl` 根目录：

```powershell
python scripts\prepare_demo05_presets.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2"
```

这个脚本使用已经冻结的 Demo-04A-v2 预测结果：
- 5 个 registered intent 各选择 1 条 frozen correct-ACCEPT query；
- 选择 1 条 frozen REJECT unknown 作为拒识演示；
- 复制到 `demo/demo05_web/static/audio/presets/`。

这是演示样本选择，不是新的 benchmark。

## Step 3：启动

```powershell
python scripts\run_demo05_web.py
```

浏览器：

```text
http://127.0.0.1:7860
```

## 当前功能

- 王灏学生档案摘要；
- 已注册规范中文表达；
- 预置真实 WAV 选择；
- 浏览器播放原始语音；
- 点击“开始识别”执行真实 H6 Runtime；
- `ACCEPT` 时只显示 `canonical_text`；
- `REJECT` 时显示“未识别到已注册表达”；
- `CONFIRM` 已保留 UI；
- 模型分数默认不展示；
- 智能学伴区域先作为接口占位。

## 下一步

前端壳子确认后，再做 Demo-05C：
- 接 Coze/其他 Agent；
- ACCEPT → canonical_text 发送 Agent；
- REJECT → 不发送；
- CONFIRM → 用户确认后发送。

不要在 Agent 接入前重新调整 frozen Demo-04 baseline。
