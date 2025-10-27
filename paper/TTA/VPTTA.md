# 📘 论文阅读记录

## 基本信息

- **论文标题**：Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation  
- **作者**：Ziyang Chen, Yongsheng Pan, Yiwen Ye, Mengkang Lu, Yong Xia  
- **机构**：Northwestern Polytechnical University, China  
- **发表会议**：CVPR 2024  
- **链接**：[https://github.com/Chen-Ziyang/VPTTA](https://github.com/Chen-Ziyang/VPTTA)  
- **研究领域**：医学图像分割、模型自适应、提示学习（Prompt Learning）

---

## 一、研究背景

在医学图像分割任务中，**跨域分布差异（distribution shift）** 是部署预训练模型的关键障碍。现有 **测试时自适应（Test-Time Adaptation, TTA）** 方法可缓解该问题，但普遍需要在测试时更新模型参数。

然而，当模型面对持续变化的目标域时（即 **持续测试时自适应，CTTA**），传统方法容易出现：

- **错误累积（Error Accumulation）**
- **灾难性遗忘（Catastrophic Forgetting）**

---

## 二、研究动机

作者认为：  

> “在推理阶段更新模型参数本身是不合适的。”  
> 因此，提出 **冻结模型参数**，通过 **视觉提示（Visual Prompt）** 对输入图像进行自适应调整，从而避免错误累积与遗忘。

---

## 三、主要贡献

1. **提出 VPTTA（Visual Prompt-based Test-Time Adaptation）框架**  
   - 通过为每个测试图像学习一个特定的视觉提示，实现轻量级、样本特定的自适应。
2. **设计低频提示（Low-frequency Prompt）**  
   - 利用图像低频分量（主要影响图像风格）进行调整，仅需极少参数。
3. **引入记忆库（Memory Bank）初始化策略**  
   - 通过历史提示与当前样本的相似性，为当前提示提供良好初始值。
4. **提出基于统计融合的预热机制（Warm-up Mechanism）**  
   - 混合源域与目标域的统计信息，平滑提示训练初期的困难。
5. **实验验证**  
   - 在两个 2D 医学分割任务上优于现有多种 SOTA 方法。

---

## 四、方法概述

### 1. 模型整体结构

冻结源模型参数，仅在输入空间学习提示 `P_i`，生成调整后的输入图像：

$\tilde{X}_i = F^{-1}([OnePad(P_i) \odot F_A(X_i), F_P(X_i)])$

其中：

- `F` 表示快速傅里叶变换（FFT）
- `P_i` 作用于低频分量（style 信息）

### 2. 记忆库初始化

- 记忆库存储历史样本的低频分量与对应提示。  
- 当前样本根据相似度检索最相近的 `K` 个提示加权求和，形成初始 `P_i`。

### 3. 提示训练（Statistics Alignment）

- 通过最小化 BN 层统计差异的绝对距离来优化提示：
  
  $L_p = \sum_j (|\mu_j^w - \mu_j^t| + |\sigma_j^w - \sigma_j^t|)$

- 其中 warm-up 统计量：
  
  $\mu^w = \lambda \mu^t + (1 - \lambda)\mu^s$

---

## 五、实验设计

### 数据集

- **OD/OC 分割任务**：RIM-ONE-r3、REFUGE、ORIGA、Drishti-GS 等五个数据集  
- **Polyp 分割任务**：BKAI、ClinicDB、ETIS、Kvasir-Seg 四个数据集  

### 模型

- ResUNet-34（OD/OC）
- PraNet（Polyp）

### 评价指标

- DSC（Dice Score）
- \($E_\phi^{max}$ \)
- \( $S_\alpha$ \)

---

## 六、主要结果

| 方法                      | 平均 DSC (OD/OC) | 平均 DSC (Polyp) |
| ----------------------- | -------------- | -------------- |
| Source Only             | 65.86          | 75.77          |
| CoTTA (CVPR’22)         | 68.98          | 71.33          |
| DLTTA (TMI’22)          | 69.84          | 69.44          |
| DomainAdaptor (CVPR’23) | 70.01          | 74.29          |
| **VPTTA (Ours)**        | **71.93**      | **80.46**      |

- 在 **持续自适应** 环境下，VPTTA 几乎无性能退化（仅 0.42%），优于所有对比方法。
- 可在 **单次迭代**、**极少参数**（仅几十个）下完成自适应。

---

## 七、消融实验结论

- **低频提示**：减少模型更新带来的灾难性遗忘。
- **记忆库 + 预热机制**：两者互补，显著提升性能。
- **参数敏感性分析**：α=0.01、S=40、K=16、τ=5 为最优配置。
- **低频提示 vs. 低秩提示（LoRA）**：低频提示更高效且性能更优。

---

## 八、结论与展望

- VPTTA 提供了一种全新的 **“冻结模型 + 样本级提示”** 的持续自适应范式。
- 未来可扩展至：
  - 3D 医学图像场景
  - 其他视觉任务（如分类或检测）
  - 多模态联合适应框架

---

## 九、个人思考

- 该方法以**输入层提示**代替**参数微调**，有效解决了持续学习中的遗忘问题。
- 其“单样本快速自适应”的特性为临床部署提供了高效且安全的解决方案。
- 未来可探讨如何将提示生成与上下文信息结合，实现更智能的跨域自适应。

---

## 🧩 参考文献

论文共引用 52 篇参考文献，涵盖 TTA、CTTA、Prompt Learning、BN 统计自适应等方向的最新研究。
