# PAPR：Head 设计评审与优化方案 V2

日期：2026-09-15

状态：设计建议，尚未实现或验证新方案性能。

范围：用户已确认“先把已注册短语识别做好”；服务于项目一的儿童语音 → 标准中文 → Agent。
依据：读取“开发准备分析”的相关历史、需求截图、实验日志；只读核查 papr_ssl 源码和已有 DEV 汇总；检索一手论文。未运行训练、未访问 generic_test 数据。

## 1. 结论

**保留 WavLM[15] + Attention Mean 256D，以及 Prototype + DTW；把 Head 改成“用户注册记忆 + 稳健证据融合 + 正确性/未知联合决策”。**

原方案的系统方向合理，但有三个必须先解决的问题：

1. 当前 15-shot 实验按类别构建全局注册库，没有用户维度，不能据此证明新孩子的个性化效果。
2. 用包含自身的原型估计类内均值方差存在乐观偏差；15-shot 的类内方差不能直接变成可靠概率或独立阈值。
3. P(Known) 不是最终短语正确的概率。ACCEPT 应以“当前输出可正确接受”为监督目标，同时约束未知误接受。

优先级应是：**用户级评估协议 → 稳健注册统计 → 决策校准 → 模板和融合优化**。暂时冻结现有表征是控制实验变量的工程选择，不代表它在所有儿童上已经最优。

## 2. 历史结果究竟证明了什么

### 2.1 现有 15-shot 缺少用户维度

[export_teacher256_15shot_embeddings.py](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/scripts/export_teacher256_15shot_embeddings.py:147) 的 build_15shot 只按 label_text 分组，从全 train 中按稳定 hash 取每类 15 条。

同文件第 208 行创建 [30,15,256]；第 219 行起把全部 DEV 作为 query。未按 user_id/speaker_id 构建个人支持集，也未按对应用户路由 query。

因此这些实验应描述为 **Core30 的类别级 few-shot 原型实验**。它们能证明组件在该协议下有效，但不能证明“一个未见孩子注册 15 条后”的效果。

本次没有统计这 450 条实际来自多少说话人，也没有证明 train/DEV 存在说话人交集，因此不把缺乏分组保证直接称为已发生的数据泄漏。

### 2.2 DTW 收益真实，但开放集问题仍存在

已有 DEV 汇总：

| 指标 | Global 256D + consistency | 加 DTW | 变化 |
|---|---:|---:|---:|
| 开放集流程中的 known Macro-F1 | 87.61% | 91.45% | +3.83 个百分点 |
| Correct Accept / known | 82.01% | 88.42% | +6.40 个百分点 |
| Wrong Intent / known | 1.18% | 1.31% | +0.14 个百分点 |
| Known Reject / known | 16.81% | 10.27% | −6.54 个百分点 |
| Unknown Reject / unknown | 85.37% | 85.57% | +0.20 个百分点 |
| 原协议 Gate 通过 | 11/20 | 13/20 | 尚不稳定 |

来源：[temporal_dtw_ablation.json](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/artifacts/p6_temporal_dtw/evaluation/temporal_dtw_ablation.json:9)。

未知误接受率约为 1−0.855688 = **14.43%**。它的分母是 unknown，不是所有接受样本；不能与 Wrong Intent 相加后当总错误率。

历史 closed-set 日志为 Macro-F1 90.1490% → 90.8766%，支持 DTW 对已知类排序有收益。这一结果与开放集流程中的 91.45% 不是相同指标口径，不能横向当作精度高低。

20 次重复划分使用同一批 DEV，不是 20 个独立外部测试；且表征选型此前已使用 DEV，这些结果仍属于开发证据。

### 2.3 现有实现与历史描述的差异

| 项目 | 源码事实 | 影响 |
|---|---|---|
| temporal projection | 复用包含 bias 的全局投影，逐帧 L2 → 每 3 帧均值 → L2 | 零新增网络参数，但没有帧级损失保证该投影最适合 DTW |
| 两个模板 | 按全局 embedding 到质心距离取最近两条 | 可能重复覆盖同一种发音，不是时序双 medoid |
| DTW | 局部成本 1−cos；最小累计成本路径后除路径长度 | 可保留为基线；不应写成“直接优化最小平均成本路径” |
| DTW 分数 | 最佳模板距离 d → 1−0.5d | 数值方向和范围合理，但不是概率 |
| rejector | 所有 known 均标 1，包括被分错的 known | 学习 knownness，不能把输出命名为识别正确概率 |
| cal/score 切分 | known 按类别随机切半，unknown 按录音随机切半 | 不保证说话人、会话或 unknown 真实短语分离 |

源码：[build_temporal_dtw_features.py](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/scripts/build_temporal_dtw_features.py:99)、[precompute_dtw_rerank.py](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/scripts/precompute_dtw_rerank.py:92)、[run_temporal_dtw_ablation.py](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/scripts/run_temporal_dtw_ablation.py:137)。

## 3. 论文依据及适用边界

| 一手研究 | 可以支持的设计 | 不能据此声称 |
|---|---|---|
| [LPM，Interspeech 2023](https://arxiv.org/abs/2306.05446) | 构音障碍语音的少量录音注册、帧级 latent sequence + DTW；包含跨天数据 | WavLM 第15层、256D、Top3和2模板已被证明最优；DTW 本身是新方法 |
| [D-ProtoNet，Interspeech 2022](https://www.isca-archive.org/interspeech_2022/kim22h_interspeech.html) | 用当前 episode 已知原型生成 dummy，训练显式考虑未知查询 | 在普通 GSC 上的结果可直接外推到儿童长短语 |
| [Few-Shot Open-Set On-Device KWS，Interspeech 2023](https://arxiv.org/abs/2306.02161) | 编码器 + 简单开放集原型分类仍是强基线；其研究中 normalized triplet 优于所比较的 dummy 方法 | 加 dummy 必然比简单方法好 |
| [PB-DSR，Interspeech 2024](https://www.isca-archive.org/interspeech_2024/wang24x_interspeech.html) | 用 HuBERT 表征和未见障碍说话人的逐词原型做个性化，无需每人再次微调 | 已解决开放集短语及可靠 Agent 决策 |
| [Adapt-KWS，ICASSP 2025，作者实现](https://github.com/Raynaming/CD-FSOS-KWS) | 冻结 DSCNN，支持集训练小 adapter，原型重投影；包含 GSC/UA-Speech/MDSC 任务 | 完全无需支持集优化、动态加类后无需处理旧类 |
| [Calibration，ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | 模型分数需要校准，独立验证上的简单校准值得作为基线 | sigmoid/softmax 输出天然可信；旧校准对任意新用户都成立 |
| [Selective Classification，NeurIPS 2017](https://arxiv.org/abs/1705.08500) | 用接受覆盖率与接受后的错误率选择工作点 | 直接设置 0.9 就得到 90% 正确率保证 |
| [Outlier Exposure，ICLR 2019](https://arxiv.org/abs/1812.04606) | 独立异常数据可用于学习拒识证据 | 曝光数据审计通过等于已经改善儿童语音拒识 |

补充检查到 2026 年的相关研究：[Transductive Prototype Refinement and Class Logit Enhancement](https://arxiv.org/abs/2607.26607) 明确讨论未知 query 污染原型，并区分分类与拒识。它联合观察整批未标注 query，不是当前逐条音频推理的直接替换方案。此处只借鉴“避免未知样本污染记忆、分类和拒识分开建模”的思路。

当前任务已限定为注册短语，因而 [MetaICL dysarthric ASR，2025](https://arxiv.org/abs/2509.15516) 等自由转写路线不进入 V1；其结果也不能替代本系统的短语识别评估。

**下述完整 V2 是结合论文、源码和统计分析提出的项目方案，不是某篇论文已经验证的架构，更不承诺性能提升幅度。**

## 4. V2 总体结构

~~~mermaid
flowchart TD
    A["单条音频 + user_id"] --> B["WavLM-Large hidden_states[15]"]
    B --> C["Attention Mean + Projection"]
    B --> D["复用 Projection + 帧掩码 + 下采样"]
    C --> E["Global z: 256D L2"]
    D --> F["Temporal H: T' × 256"]
    M["用户注册库 E(u,c)"] --> G["全类别余弦召回"]
    E --> G
    G --> H["候选集合 Top-K"]
    H --> I["候选类别 DTW"]
    F --> I
    M --> I
    M --> J["LOO统计 + 方差收缩 + 支持集一致性"]
    G --> K["共享低容量证据融合器"]
    I --> K
    J --> K
    K --> L["最终候选及竞争证据"]
    L --> N["正确 / 错分 / 未知 联合决策头"]
    N --> O["ACCEPT / CONFIRM / REJECT"]
    O --> P["intent_id → 标准文本 → Agent"]
~~~

职责：

- Backbone/Neck：先冻结现有 checkpoint；继续输出全局和时序表征。
- SCAF：属于训练表征的目标/分类层，不作为动态业务类别库。
- PCEN、Student、KD：保留在项目二的后续路线，不增加到当前 WavLM 原始波形输入链。
- 用户身份：由已选择的用户档案或会话提供；该 head 不自动承担声纹认证。
- 输入质量：无有效语音、空录音或处理失败返回明确 reason；不把这种错误解释成某个意图。

“Global”应称全局声学/短语表征相似度；不能未经验证称为语言模型意义上的语义理解。

## 5. 模块规格

### 5.1 Enrollment Memory：以 (用户, 短语) 为主键

对用户 u 的类别 c：

\[
E_{u,c}=\{(z_i,H_i,\text{session}_i,\text{quality}_i)\}_{i=1}^{n_{u,c}}.
\]

保存 n 条独立全局 embedding、n 条时序 embedding 与运行时模板索引、归一化平均原型、LOO/收缩统计、样本数/缺失标记、session/录音 ID/质量信息，以及 intent_id/canonical_text 和各版本号。

原型先采用等权均值：

\[
p_{u,c}=\operatorname{Norm}\left(\sum_i z_i\right).
\]

每类支持不同 n，不把 15 条写成接口硬限制。建议比较 1/2/5/10/15-shot；采集尽量跨会话，15 次连续复读不能代表跨天变化。

只剔除明确无效或标错录音；不能仅因离质心较远便删掉严重或少见的真实发音。标注有效但多样的发音应保留。

全部 30×15×256 的全局 float32 embedding 约 0.44 MiB。时序存储随有效帧数增长，应单独测量。

### 5.2 稳健注册统计：LOO + 收缩

原式 cos(z_i,p) 使用了包含 z_i 的 p；新 query 不享有这种自包含效应。

使用留一原型：

\[
p_{u,c}^{(-i)}=\operatorname{Norm}\left(\sum_{j\ne i}z_j\right),\quad
a_i=z_i^\top p_{u,c}^{(-i)}.
\]

计算均值 μ 和样本方差 v，向开发用户的共享先验收缩：

\[
\widetilde\mu=\alpha_n\mu+(1-\alpha_n)\mu_0(n),
\quad \alpha_n=\frac{n}{n+\kappa_\mu}
\]

\[
\widetilde v=\max\left(
\beta_n v+(1-\beta_n)v_0(n),v_{\min}
\right),
\quad \beta_n=\frac{n-1}{n-1+\kappa_v}.
\]

query 的统计特征：

\[
r_{u,c}(q)=
\operatorname{clip}\left(
\frac{z_q^\top p_{u,c}-\widetilde\mu}{\sqrt{\widetilde v}},
-B,B\right).
\]

这是建议公式。先验、收缩强度、方差下限、截断范围只在开发训练/内部验证确定。

实施约束：

1. r 只是“类内相容性特征”，不是标准正态变量、p-value 或正确概率。
2. n=1 无 LOO；n=2 两个 LOO cosine 相同，方差没有信息。n<3 或独立样本不足时回退共享先验并提供 missing 标记。
3. LOO 原型有 n−1 条、部署原型有 n 条，仍须用独立 query 做最终校准。
4. 有多会话时比较 leave-one-session-out；高度相关复读不当成独立证据。
5. 保留 raw cosine 和类间竞争信息，不能让高方差类别单靠“比较宽松”赢得所有 query。
6. 可加入支持样本与其他注册类原型的距离差，反映“稳定但易与另一类混淆”的情况。

对小样本统计做共享/正则化与 [Simple CNAPS](https://arxiv.org/abs/1912.03432) 思路相近，但这里是标量统计简化。不要用 15 条录音直接求逆 256×256 的无正则协方差。

### 5.3 Global 检索：先测召回上限

\[
g_c=z_q^\top p_{u,c},\qquad
\mathcal C_K=\operatorname{TopK}_c(g_c).
\]

独立报告 Recall@3、@5、@10、@全部类别。最终分类正确率不可能超过候选 Recall@K。

- 固定 Top3×2 保留为现有对照。
- 在目标用户 DEV 上选择满足召回/延迟要求的 K；候选数为 min(K,C_u)。
- V1 先用固定 K，便于公平比较和校准。
- 后续可试 Top3 → Top5/10 的不确定样本扩展；整个规则要进入训练和校准流程。
- 验证可靠前不设置“global 分数一低就提前 REJECT”，避免 DTW 尚未运行便失去救回机会。

### 5.4 Temporal Branch：优化模板覆盖

保持现有 1−cos 局部成本、路径长度归一化和下采样作为基线。query 和模板使用一致的 mask、归一化及版本。

新增对照为“时序双 medoid”：从有效支持集中选 1～2 条实际录音，使所有支持录音到最近模板的距离之和较小：

\[
\{m_1,m_2\}
=\arg\min_{\{j,k\}\subset E_{u,c}}
\sum_i\min(D(H_i,H_j),D(H_i,H_k)).
\]

n=15 时可在注册阶段枚举模板对；注册开销与在线推理开销分开报告。

- n=1 只保留一个模板；数据呈单一模式时，第二模板未必有收益。
- 两模板可能代表不同的真实发音模式，因此匹配分歧只作为特征，不强制要求两者同时高匹配。
- 记录最佳距离 d_min、另一模板距离或差值、时长比，作为融合证据。
- 最佳匹配 score 不是概率；模板数改变会改变偶然高匹配的机会。
- 避免把所有序列强行拉伸到相同长度；任何 DTW 窗口、斜率或时长限制都需用慢速/严重障碍语音验证。

复用 256D projection 暂时保留；如目标数据证明它损害时序区分，再独立对照 1024D 帧特征或小型时序投影，不在 V1 同时重训表征。

### 5.5 融合：共享证据打分，参数不随类别数增长

保留现有固定 λ 融合作为对照，再比较有 L2 正则的共享线性打分器：

\[
s_c=w^\top \operatorname{Standardize}(\phi_c)+b.
\]

候选级特征可从下列小集合开始：

\[
\phi_c=[
g_c,\,
g_c-\max_{c'\ne c}g_{c'},\,
\text{existing-consistency}_c,\,
r_{u,c},\,
-d_{\min,c},\,
d_{\text{other},c}-d_{\min,c},\,
|\log(T_q/T_{\text{best},c})|,\,
\log n_{u,c}
].
\]

加入必要的缺失掩码；没有第二类或第二模板时不能制造“无限大 margin”。现有 consistency 为一组统计特征，上式是概念性分组，不代表最终必须八维；特征精简通过内部消融决定。

以候选是否为正确注册类做二元监督：正确候选为 1，错类和 unknown query 的全部候选为 0；控制每个 query、用户和类的权重，避免候选数或大用户主导训练。训练时运行真实检索、模板选择和距离聚合。s 只用作排名证据，不直接作为业务 confidence。

所有候选使用同一组 w，而不是每个短语训练独立头。标准化器只在 head 训练集拟合。

相比历史动态 sigmoid gate，该模型便于分别估计各证据的贡献。gate 保留为消融，不能仅凭参数少便认定更优。

### 5.6 决策：区分“正确、错分、未知”

得到最终 top1 类别 c* 后，计算 query 级特征 ψ：

- 最终 top1/top2 分数与 margin；
- c* 自己的 global、DTW、LOO/consistency 证据；
- global 与 DTW 的赢家是否一致；
- 注册样本数、类别数、候选数、缺失标记；
- 必要的音频有效性证据。

按最终候选 class ID 对齐证据；DTW 改变 top1 后不能继续拿旧 global top1 的统计量解释新结果。

用共享、L2 正则的三分类 logistic：

\[
(p_C,p_W,p_U)=\operatorname{softmax}(W\psi+b).
\]

| 标签 | 真实事件 |
|---|---|
| C | query 属于注册类，且最终 top1 正确 |
| W | query 属于注册类，但最终 top1 错误，包括真类未进入候选 |
| U | query 不属于该用户当前注册集合 |

P(Known)=p_C+p_W；**confidence 应对应 p_C，而不是 P(Known)**。这三个标签描述决策事件，不是把动态短语库训练成固定 31 类分类器。

规则模板：

~~~text
有效输入且 p_C ≥ τ_accept，同时 p_U ≤ τ_unknown_guard → ACCEPT
p_U ≥ τ_reject                                      → REJECT
其余                                                → CONFIRM
~~~

约束 τ_unknown_guard < τ_reject；阈值由独立校准集选择。

- ACCEPT：intent_id + canonical_text 作为已接受结果交给 Agent。
- CONFIRM：给出候选并等待确认；按孩子能力采用按钮/照护者等可访问方式，不默认孩子能清晰口头说“是/否”。
- REJECT：提示重说或选择短语，不自动把 top1 当最终文本。
- 无有效音频返回对应 reason，概率可为空，不伪造 0.00。

三分类仍需可靠性验证。若独立错分样本不足，先用正则化二元“top1 正确/不正确”头与未知阈值回退，不强行展示精确三类概率。

## 6. 数据和训练协议

### 6.1 真正的个人 episode

每个 episode 选择用户 u 及当前注册词表 C_u：

1. support：该用户同一短语的 n 条有效录音。
2. known query：同用户、已注册短语、不同原始录音，优先来自另一天/session。
3. unknown query：同用户未注册语句，包括近音、否定、截断短语和自然讲话。
4. 辅助负例：环境声音、串入语音等，分别记录来源；身份识别不默认成为此 head 的已验证能力。

support 与 query 不能是同一原始音频的裁剪或增强副本。测试用户可以提供个人 support，但其 query 不能参与训练或选阈值。

### 6.2 分层拆分

- Head-fit：训练融合器和决策头。
- Calibration：必要时温度缩放、选择最终决策阈值。
- Held-out score / final test：冻结配置后评价。

融合器与决策头形成堆叠：先在 fit-A 训练排名，再用 fit-B 或交叉拟合产生 out-of-fold 排名，训练决策头，避免只看到过于容易的训练集结果。

若 head-fit 对 C/W/U 或 known/unknown 做重采样、类别加权，拟合出的概率不自动代表部署先验。概率校准应使用代表部署混合比例的独立数据，或进行明确的先验/截距校正并重新验证。单一 temperature scaling 不能一般地修复类别先验变化；没有代表性校准数据时应输出 score 而非宣称已校准的 confidence。

数据少时用嵌套、按用户/会话分组的交叉验证；头部训练与阈值选择在内层，外层用户只评分。反复切同一批录音不能替代新用户证据。

开放类按真实短语身份区分 exposure、calibration 与 held-out unknown；动态增删类额外测试训练时未见的词表组合。若声称可识别从未训练过的新短语，还需 phrase-disjoint 实验；它与新用户泛化是两个维度。

generic_test 继续封存，直到协议与模型选择完成。已反复用于表征选型的 DEV 不能重新命名为独立最终测试。

### 6.3 Unknown Exposure

现有候选 5307 条、2510 个真实短语、34 人，其中 control 3322 / dysarthria 1985；审计中与 Core30 和 open DEV 真实短语交集为 0。来源：[unknown exposure audit](D:/02_开发项目/02_AI视觉与机器人/02_语音识别/papr_ssl/artifacts/p6_unknown_exposure_audit_v5/p6_unknown_exposure_audit.json:27)。

先用于 head-fit 负例，保持 Backbone/Neck 固定。按用户、域和短语适当平衡，防止学成“标准语音是 unknown、障碍语音是 known”。现有数量与去重只证明候选数据可用，不能证明曝光训练有效或儿童泛化。

unknown 相对于当前注册集合定义：同一个短语在 episode A 可已注册，在 episode B 可未注册，不是永久 Unknown 标签。

## 7. 验收指标和消融

### 7.1 必须同时报告

- Global Recall@K：候选召回上限。
- Closed-set top1 / Macro-F1：纯已知类区分能力。
- Correct Accept：#(known 且 ACCEPT 且 top1 正确) / #known。
- Wrong Intent：#(known 且 ACCEPT 且 top1 错误) / #known。它不是标签 W 的总体占比；W 还可能被 CONFIRM 或 REJECT。
- Known Reject / known 和 CONFIRM / known，单独列出。
- Unknown 自动接受率：#(unknown 且 ACCEPT) / #unknown。
- Unknown CONFIRM 率：避免把未知全转成确认后隐藏问题。
- ACCEPT coverage：#ACCEPT / #全部 query。
- Selective risk：#(ACCEPT 但输出错误，包括 unknown) / #ACCEPT。
- 可靠性图、Brier/NLL/ECE；按用户/会话给出不确定性。
- 总体、每用户和不同发音难度的结果。
- p50/p95 延迟、DTW 次数、注册时间、内存。

CONFIRM 不算自动识别成功；人工确认后的交互成功率另报。

有效 known query 的 Correct Accept、Wrong Intent、Known Reject 和 Known CONFIRM 四项应合计为 1。数据处理失败应另行统计，不通过静默丢弃样本抬高成功率。

研究可预先设 FAR=1%/5%/10% 曲线，比较固定 FAR 下的 Correct Accept；这些是建议比较点，不是已达到的指标或产品承诺。阈值在校准集确定，在留出集评价，不能在最终测试上挑最优阈值。

概率和 selective risk 受真实 known/unknown 比例影响；报告测试构成和不同混合比例下的稳定性。逐条 WAV 的 FAR 不能替代连续监听的每小时误触发数。

### 7.2 最小实验矩阵

| 实验 | 改动 | 回答的问题 |
|---|---|---|
| B0 | 当前 Global+DTW 迁入个人 episode | 真正个人支持集上表现如何 |
| B1 | 原统计 vs LOO+收缩 | 个人统计是否带来增益 |
| B2 | 最近质心两模板 vs 时序双 medoid | 发音覆盖是否更好 |
| B3 | 固定 λ vs 共享线性融合 | 学习融合是否优于简单加权 |
| B4 | knownness vs 正确/错分/未知决策 | 同 FAR 下是否更可靠地接受 |
| B5 | 不用/使用 Unknown Exposure | 是否改善未见未知类 |
| B6 | 最终组合与去掉关键模块 | 组合增益来源是什么 |

另报 K 的召回/延迟曲线、shot 数曲线及跨会话结果。按外层用户/会话成对比较，不把重复录音当独立样本。

## 8. 工程接口与实施顺序

建议接口：

~~~text
enroll(user_id, intent_id, canonical_text, labeled_audio[])
predict(user_id, audio, memory_version)
update_enrollment(user_id, intent_id, verified_audio[])
remove_phrase(user_id, intent_id)
~~~

predict 输出：

~~~text
status: ACCEPT | CONFIRM | REJECT
intent_id: 接受的意图；非接受时为空
canonical_text: 接受的标准文本；非接受时为空
candidates: 候选 intent_id / 文本 / ranking_score
confidence: 已校准的 P(top1正确)，未校准时为空
unknown_probability: 已校准时提供，否则为空
reason
model_version / memory_version / calibration_version
~~~

排序 score 与 confidence 分开。只把已验证标签或可靠显式确认后的样本加入记忆，不用自动 ACCEPT 的伪标签持续更新，以免错误污染原型。

新增类别无需训练 C-way 输出层，但会改变 top1、margin、unknown 最大匹配值。需重算跨类统计，并使用覆盖该类别数/shot 数范围的已验证校准策略。修改 encoder/neck 后重算全部注册表征；修改 K、模板数或距离变换后重新验证校准。

实施顺序：

1. **协议与 Memory**：用户维度、跨会话 support/query、版本和原始录音去重。
2. **个人基线**：冻结现有表征及 Global+DTW，获得真正个人基线。
3. **LOO 与决策语义**：稳健统计，独立证据训练正确性/未知决策。
4. **模板与融合消融**：有稳定收益的组件才进入最终组合。
5. **终端/API Demo**：三态输出与标准文本查表，供前端和 Agent 联调。

适合凝练的研究问题：**在低注册成本、跨会话和真实未知语句下，支持集统计能否在固定未知误接受率下提高个人短语的自动正确接受率？**

LOO、收缩、DTW、logistic、三态交互各有相关先例；组合或命名不足以证明方法创新。贡献需要由明确的新机制、严格个人协议与消融证据建立。
