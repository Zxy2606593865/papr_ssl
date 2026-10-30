# SSL Teacher 路线图

| 阶段 | 范围 | 门槛 |
|---|---|---|
| T0 | Wav2Vec2 + Mean DR | 离线契约和单元测试通过。 |
| T1 | WavLM + Mean DR | 离线 wrapper/factory 测试通过；真实集成暂缓。 |
| T2 | W2v-BERT 2.0 + Mean DR | Feature-extractor 到 `input_features` 的流程通过离线测试。 |
| T3 | 各 backbone 的 Layer sweep | 候选展开后无重复；真实索引通过验证且不使用 fallback。 |
| T4 | 在 `generic_dev_score` 上选择 backbone 和 layer | 使用相同的 split、budget、head、SCAF 和 metrics。 |
| T5 | Mean DR 与 Attention DR 对比 | 仅在 T4 冻结后开始。 |
| T6 | 最佳单层与 Layer Fusion 对比 | 仅在 T5 冻结后开始。 |
| T7 | E0 Teacher 资格确认 | 所有开发 gate 和报告均通过。 |
| T8 | 冻结 Teacher | 固定代码、model revision、dependency lock、checkpoint 和 protocol。 |
| T9 | 开放 `generic_test` | 首次允许访问最终 generic test。 |
| T10 | 跨域评估 | 使用已冻结的 Teacher 和 threshold policy；不重新训练 Teacher。 |
| T11 | 导出 Teacher target | 建立带版本和 hash 校验的跨框架 target 契约。 |
| T12 | KD Student | 仅在 Teacher target 和资格确认冻结后开始。 |

T9 之前，`generic_test` 必须保持封存。它不得影响 T0–T8 阶段的任何选择，包括 backbone、hidden-layer、checkpoint、threshold、Attention DR 或 Layer Fusion。

下一阶段是 **T-Integration**：运行真实 Wav2Vec2、WavLM 和 W2v-BERT 2.0 smoke test；验证 processor/frontend、实际 `[T,D]`、mask、hidden-state 精确索引、内存和吞吐量；随后固定 model revision 和 dependency 版本。
