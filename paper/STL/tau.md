# 📘 论文阅读笔记：*Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning*

**作者：** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li  
**会议：** CVPR 2023  

---

## 一、研究背景

**时空预测学习（Spatiotemporal Predictive Learning）** 旨在根据历史帧预测未来帧，广泛应用于气候预测、人类动作预测、交通流量预测等任务:contentReference[oaicite:0]{index=0}。

传统方法多采用 **ConvLSTM**、**PredRNN** 等循环单元（RNN）捕捉时间依赖，但由于其顺序计算特性，**无法并行、计算效率低**。  
本文提出的 **Temporal Attention Unit (TAU)** 用 **并行化注意力机制** 替代循环结构，从而提高计算效率并保持或超越性能。

---

## 二、主要贡献

1. **提出 Temporal Attention Unit (TAU)**：
   - 将时序注意力分解为：
     - **Intra-frame Statical Attention（帧内静态注意力）**：捕捉空间内长程依赖；
     - **Inter-frame Dynamical Attention（帧间动态注意力）**：建模时间演化。
2. **提出 Differential Divergence Regularization (DDR)**：
   - 传统 $L_{MSE}$ 仅关注帧内像素误差；
   - DDR 约束帧间变化，使模型学习动态差异。
3. **实现端到端无监督预测模型**，不依赖循环结构即可获得与主流方法相当甚至更优的性能。
4. **在多数据集上验证优越性与泛化能力**，包括 Moving MNIST、TaxiBJ、KITTI→Caltech、KTH。

---

## 三、模型框架

![](/Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-27-17-10-18-image.png)

![](/Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-27-17-10-33-image.png)

### 3.1 总体结构

模型结构遵循通用三模块框架：

> **Encoder → Temporal Module (TAU) → Decoder**

- **Encoder/Decoder：** 使用简单的 2D Conv 与 ConvTranspose2D；
- **Temporal Module：** 堆叠 TAU 模块捕捉时间依赖；
- **Residual connection** 连接首末层，保留空间特征。

<div align="center">图示：Spatial Encoder → TAU Stack → Spatial Decoder</div>

---

### 3.2 Temporal Attention Unit (TAU)

输入特征张量 $H \in \mathbb{R}^{B \times (T \times C') \times H \times W}$  
TAU 包含两个核心模块：

$$
\begin{aligned}
SA &= Conv_{1 \times 1}(DW\text{-}DConv(DWConv(H))) \\
DA &= FC(AvgPool(H)) \\
H' &= (SA \otimes DA) \odot H
\end{aligned}
$$

- **$SA$**（静态注意力）：通过深度可分卷积 + 空洞卷积模拟大卷积核，建模帧内空间依赖；
- **$DA$**（动态注意力）：通过通道加权（Squeeze-and-Excitation）捕捉帧间动态变化；
- **最终特征融合：** 两者乘积加权原特征。

---

### 3.3 Differential Divergence Regularization (DDR)

传统 MSE 仅关注帧内差异，DDR 通过引入帧间差分约束提升动态一致性：

1. 计算预测帧与真实帧的前向差分：
   
   $$
   \Delta \hat{Y}_i = \hat{Y}_{i+1} - \hat{Y}_i, \quad
\Delta Y_i = Y_{i+1} - Y_i
   $$
2. Softmax 归一化得到概率分布：
   
   $$
   \sigma(\Delta \hat{Y}) = \frac{e^{\Delta \hat{Y}/\tau}}{\sum e^{\Delta \hat{Y}/\tau}}
   $$
3. 基于 KL 散度计算正则项：
   
   $$
   L_{reg} = D_{KL}(\sigma(\Delta \hat{Y}) \| \sigma(\Delta Y))
   $$
4. 最终损失：
   
   $$
   L = \sum_i \| \hat{Y}_i - Y_i \|_2^2 + \alpha L_{reg}
   $$

其中 $L_{reg}$ 关注帧间变化，补充了 $L_{MSE}$ 的不足。

---

## 四、实验设计与结果

### 4.1 数据集

| 数据集           | 类型   | 输入→输出帧   | 特征    |
|:------------- |:---- |:-------- |:----- |
| Moving MNIST  | 合成   | 10→10    | 两数字移动 |
| TaxiBJ        | 实际交通 | 4→4      | 出/入流量 |
| KITTI→Caltech | 跨域   | 10→1     | 泛化测试  |
| KTH           | 人类动作 | 10→20/40 | 长序列预测 |

---

### 4.2 实验结果总结

#### 🔹 Moving MNIST

- Ours: MSE 19.8, SSIM 0.957  
- 超越 ConvLSTM (103.3), PredRNN (56.8), SimVP (23.8)

#### 🔹 TaxiBJ

- Ours: MSE×100 = 34.4, SSIM 0.983  
- 明显优于 PhyDNet (41.9), SimVP (41.4)

#### 🔹 KITTI→Caltech 泛化

- Ours: SSIM 0.946, PSNR 33.7  
- 超越 SimVP (0.940, 33.1)

#### 🔹 KTH 长序列预测

| 任务    | SSIM  | PSNR  |
|:----- |:----- |:----- |
| 10→20 | 0.911 | 34.13 |
| 10→40 | 0.897 | 33.01 |

> 超越所有主流方法（如 PredRNN++、E3D-LSTM、STMFANet、SimVP）。

---

### 4.3 消融实验

| 变体            | MSE(↓)   | TaxiBJ×100 | FLOPs(G) |
|:------------- |:-------- |:---------- |:-------- |
| Conv Baseline | 58.9     | 43.5       | 6.1      |
| w/o SA        | 23.2     | 40.8       | 15.3     |
| w/o DA        | 22.4     | 38.4       | 16.0     |
| w/o DDR       | 21.1     | 37.7       | 16.0     |
| **Ours**      | **19.8** | **34.4**   | **16.0** |

> → 说明 **静态注意力、动态注意力与DDR均显著提升性能**。

---

## 五、模型效率

- **TAU 模块完全可并行化**，相比 CrevNet (30min/epoch) 与 PhyDNet (7min/epoch)，仅需 **2.5min/epoch**；
- 在 50 个 epoch 内即可达到 MSE 35.0，而对比模型尚未收敛。

---

## 六、总结与启发

| 优点     | 说明                  |
|:------ |:------------------- |
| 并行高效   | 替代RNN结构，实现显著加速      |
| 表现优异   | 多数据集超越主流时空预测模型      |
| 泛化良好   | KITTI→Caltech迁移性能突出 |
| 无监督可扩展 | 自监督框架下端到端训练         |

**关键创新点**：

1. 时间建模由“顺序记忆”转向“全局注意力”；
2. DDR使模型从“像素拟合”转向“动态理解”；
3. 以简单结构实现复杂时空动态捕捉，体现“Simple is Better”理念。

---

## 七、思考与未来方向

- 可进一步将 TAU 应用于更复杂的视频预测（如物理模拟、动作生成等）；
- 探索 **多尺度或分层时序注意力结构**；
- 与 **扩散模型或变分结构** 结合，提升生成多样性。
  
  

---

**关键词：** Spatiotemporal Prediction, Attention, Self-supervised Learning, Temporal Modeling, TAU, DDR



在c3vd上进行实验发现tau效果非常好，可能是由于c3vd不同切片之间非常接近，tau的深度卷积有着比较好的效果。

---
