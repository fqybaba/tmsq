# 术语表（中英对照）

翻译中文版 post 和提纲时统一用这些译法。你自己写中文稿时也建议沿用，这样最后逐字直译成英文时，术语能对回原文。

| English | 中文 | 备注 |
|---|---|---|
| superposition | 叠加 | |
| Toy Models of Superposition (TMS) | 叠加玩具模型（TMS） | 论文名保留英文 |
| toy model | 玩具模型 | |
| feature | 特征 | |
| ground-truth feature | 真实特征 | 玩具模型里已知的特征 |
| antipodal pair | 反极对 | 两个特征方向相反，D_i ≈ 0.5 |
| crowded feature | 拥挤特征 | 和多个特征共享维度，D_i 小 |
| polytope | 多胞形 | 如五边形、四面体 |
| feature dimensionality D_i | 特征维度 D_i | |
| sparsity S | 稀疏度 S | 特征为 0 的概率 |
| frequency p_i | 频率 p_i | p_i = 1 − S_i |
| importance I_i | 重要性 I_i | |
| tied weights | 绑定权重 | 编码器和解码器共用 W |
| bias b_i | 偏置 b_i | |
| interference | 干扰 | positive / negative interference：正干扰 / 负干扰 |
| margin | 余量（margin） | 首次出现时附英文 |
| post-training quantization (PTQ) | 训练后量化（PTQ） | |
| round-to-nearest (RTN) | 就近取整量化（RTN） | |
| quantization-aware training | 量化感知训练 | |
| bit width | 位宽 | |
| scale | 缩放因子 | |
| per-row / per-column / per-tensor | 按行 / 按列 / 按整个张量（缩放） | |
| rounding error | 舍入误差 | |
| matched Gaussian noise | 匹配高斯噪声 | 每列误差范数与 RTN 相同的随机噪声 |
| survival | 存活 | Duan 的定义：Pearson 相关 > 0.9 |
| composite damaged label | 复合损伤标签 | Pearson < 0.9，或 R² 下降 > 0.1，或误激活上升 > 0.05 |
| false activation | 误激活 | 特征应为 0 时输出 > 0.05 |
| leakage | 泄漏 | |
| eligible feature | 合格特征 | 进入分析的特征 |
| eval set | 评估集 | |
| instance | 实例（模型实例） | |
| seed | 随机种子 | |
| pre-registration / pre-registered | 预注册 / 预注册的 | |
| post hoc | 事后（分析） | |
| exploratory | 探索性 | |
| confirmatory | 验证性 | |
| primary cell | 主单元 | 预注册时指定的 (S, 位宽, 粒度) 组合 |
| exploration set / confirmation set | 探索集 / 确认集 | E 和 K 两批模型 |
| reliability (ICC) | 可靠性（ICC） | |
| plateau | 平台 | D* 曲线上的平台 |
| cutoff / threshold | 阈值 | |
| threshold density (near_thr) | 阈值密度（near_thr） | |
| head / tail | 头部 / 尾部 | 高频 / 低频特征 |
| relative damage | 相对损伤 | |
| absolute damage | 绝对损伤 | |
| bit allocation | 比特分配 | |
| uniform 4-bit | 统一 4 位 | |
| loss-aware rounding | 感知损失的舍入 | |
| sparse autoencoder (SAE) | 稀疏自编码器（SAE） | |
| residual stream | 残差流 | |
| perplexity | 困惑度 | |
| peak activation | 峰值激活 | |
| firing rate / rarity | 激活率 / 稀有度 | |
| ablation | 消融 | |
| implementation note | 实现说明 | |
| replication | 复现 | |
| Spearman correlation ρ | Spearman 相关系数 ρ | |
| confidence interval | 置信区间 | |
| sign-flip test | 符号翻转检验 | |
| Holm correction | Holm 校正 | |
| AUC | AUC | 不译 |
