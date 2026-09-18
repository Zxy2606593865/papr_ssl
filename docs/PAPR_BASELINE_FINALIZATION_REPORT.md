# PAPR 双项目工程基线收尾报告

报告日期：2026-09-18（Asia/Shanghai）

## 1. 收尾结论

本次仅处理以下两个独立仓库：

- `papr_ssl`：`https://github.com/Zxy2606593865/papr_ssl.git`
- `papr_audio_toolkit`：`https://github.com/Zxy2606593865/papr_audio_toolkit.git`

`whisper/papr` 明确排除，未执行检查、修改、提交或推送。

两个仓库均已在 clean working tree 上 fetch，并确认当时的 `origin/main` 各自只有一个初始化提交、tree 中只有 `LICENSE`。随后分别使用 `git merge origin/main --allow-unrelated-histories --no-commit` 进入待提交状态，审计 staged diff 后生成中文 merge commit。未发生冲突，未使用 force push、force-with-lease、rebase 或任何历史重写。

## 2. Git 合并与许可证记录

| 仓库 | merge 前本地 HEAD | 已审计的 origin/main HEAD | merge commit | 冲突 | LICENSE 最终来源 |
| --- | --- | --- | --- | --- | --- |
| `papr_ssl` | `70dde1ccbb563fe63ac8a3f7ebb131c3cc6c1383` | `2140516322729d334740b1ebc2cd64f904bc1e63` | `fc95fe8cb73c0032cbb59f470807ff3001f0fee5` | 无 | 保留 `origin/main` 的 Apache License 2.0；blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| `papr_audio_toolkit` | `ded07b3754d2d0e1fdd34df8d902e46e3a791569` | `50f3ba160be3150c3e38e165b5c6b4fe11f4c961` | `592de6e4a6ef65bc888664c7d0e2934d9a858789` | 无 | 保留 `origin/main` 的 Apache License 2.0；blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |

两个 merge commit 的提交说明均为：`合并远端初始化历史并保留许可证`。

待提交状态审计结果：

- staged change 仅为新增远端 `LICENSE`；
- 本地源码、文档和 `.gitignore` 未被覆盖；
- 无已有文件删除；
- 未加入 dataset、artifact、checkpoint、raw WAV、`.env`、token 或 secret；
- `LICENSE` 内容和 blob 均与各自已审计的 `origin/main` 一致。

由于没有冲突，不存在冲突取舍或手工合并项。

## 3. Teacher Target 与工程状态修正

后续文档修正 commit：

- `b93c5d1647790f55d4b51ceff5954c7e9ee5c89b` — `修正 Teacher Target 维度并明确 Head V2 状态`

当前冻结 Teacher 表征契约：

```text
microsoft/wavlm-large @ c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
→ outputs.hidden_states[15]
→ Attention DR
├─ Global: 256D float32 normalized embedding
└─ Temporal: T' × 256D float32 features
   shared projection, downsample = 3
```

P7 Teacher Target Export 尚未实现。其 manifest 应至少记录：

- model name、model revision、hidden layer；
- head/projection version；
- dtype、normalization；
- source audio/utt id、target type；
- hash、manifest version。

64D 是未来 Student embedding 目标，不是当前冻结 Teacher 的原始 target 维度。`Teacher 256D → KD alignment/projection → Student 64D` 属于 P8+ 设计，本次未实现。

工程状态区分如下：

- Teacher Backbone / Representation：基本冻结；
- H6 Runtime：已完成，是当前 Project 1 工程基线，包含 Enrollment、Prototype、Top-3 DTW、C/W/U、ACCEPT/CONFIRM/REJECT、`intent_id` 与 `canonical_text`；
- Head V2：设计完成，但尚未实现，且尚未完成新用户/跨会话验证。

历史结论保持不变：P6 absolute Gate 为 `FAIL / BLOCKED`；H7 的 freeze/audit PASS 仅表示最终配置封存通过，不代表 P6 PASS。P/H 阶段编号未改动。

## 4. 测试结果

| 仓库 | 命令 | 结果 |
| --- | --- | --- |
| `papr_ssl` | `D:\anaconda3\envs\papr_ssl\python.exe -m unittest discover -s tests -v` | 342 tests passed，5 skipped |
| `papr_audio_toolkit` | `D:\anaconda3\envs\papr_ssl\python.exe -m unittest discover -s tests -v` | 34 tests passed |

本次只修正文档与 Git 基线，未修改模型、Runtime、科研结果、训练配置或阶段编号。

## 5. Push 与最终状态

两仓均使用普通命令 `git push -u origin main`：

- `papr_ssl`：首次推送成功，`2140516..b93c5d1  main -> main`，已设置跟踪 `origin/main`；本报告随后作为独立中文提交再次普通推送。
- `papr_audio_toolkit`：推送成功，`50f3ba1..592de6e  main -> main`，已设置跟踪 `origin/main`。

终检结果：两个仓库 `git status --porcelain=v1` 均无输出，工作区 clean；`git status --short --branch` 均显示 `## main...origin/main` 且无 ahead/behind；本地 `HEAD` 与 `origin/main` 一致。本报告自身的提交 hash 记录在最终交付回执中。

## 6. 下一阶段入口

- Project 1：从 `AGENT-A0` 开始；本次未启动 Agent 开发。
- Project 2：从 P7 Teacher Target Export 开始；本次未启动 P7、Student、KD 或 Edge 开发。
