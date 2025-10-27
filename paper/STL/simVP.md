# 📘 论文阅读笔记：SimVP — Simpler Yet Better Video Prediction

**论文标题**：SimVP: Simpler Yet Better Video Prediction  
**作者**：Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li  
**机构**：Westlake University  
**年份**：CVPR 2022  
**链接**：[论文 PDF 原文](#)  

---

## 🧭 一、研究背景与动机

视频预测（Video Prediction）旨在根据过去的视频帧预测未来帧，广泛应用于：

- 气候变化预测
- 人体动作预测
- 交通流量预测
- 表征学习（Representation Learning）

现有方法多依赖复杂架构：

- RNN 系列（如 ConvLSTM、PredRNN、MIM-LSTM）
- Transformer 系列（如 AViT、Latent AViT）
- 结合 CNN 与 ViT 的混合模型

> **问题**：这些复杂结构真的有必要吗？  
> **目标**：设计一个简单但高效的模型，仅基于 CNN，就能达到甚至超越 SOTA（state-of-the-art）。

---

## 🧩 二、任务定义

<img src="file:///Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-27-16-08-36-image.png" title="" alt="" width="392">

给定过去 $T$ 帧视频序列 $X_{t,T} = \{x_i\}_{t-T+1}^{t}$，预测未来 $T'$ 帧序列 $Y_{t,T'} = \{x_i\}_{t+1}^{t+T'}$。  
其中每帧 $x_i \in \mathbb{R}^{C \times H \times W}$。

模型目标是最小化预测误差：
$
\Theta^* = \arg\min_{\Theta} \mathcal{L}(F_\Theta(X_{t,T}), Y_{t,T'})
$

在 SimVP 中，损失函数采用 **MSE（Mean Squared Error）**：
$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \| \hat{Y}_i - Y_i \|_2^2
$

---

## 🧱 三、模型结构：SimVP 架构概览

![](/Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-27-16-09-18-image.png)

SimVP 完全基于 **CNN-CNN-CNN** 结构，由三部分组成：
Encoder → Translator → Decoder
### 1️⃣ Encoder

- 提取空间特征 (spatial features)
- 模块结构：`Conv2d → LayerNorm → LeakyReLU`
- 重复 $N_s$ 次堆叠

$$
z_i = \sigma(\text{LayerNorm}(\text{Conv2d}(z_{i-1}))), \quad 1 \le i \le N_s
$$

### 2️⃣ Translator

- 捕捉时间演化特征 (temporal evolution)
- 使用多个 **Inception 模块**
- 每个模块包括：1×1 bottleneck + 并行 GroupConv2d

$$
z_j = \text{Inception}(z_{j-1}), \quad N_s < j \le N_s + N_t
$$

### 3️⃣ Decoder

- 重建预测视频帧
- 模块结构：`ConvTranspose2d → GroupNorm → LeakyReLU`

$$
z_k = \sigma(\text{GroupNorm}(\text{ConvTranspose2d}(z_{k-1}))), \quad N_s+N_t < k \le 2N_s+N_t
$$

🟩 **特点总结：**

- 无 RNN、无 LSTM、无 Transformer
- 无对抗训练（GAN）、无教师蒸馏、无复杂策略
- 完全使用 CNN + MSE loss

---

## 🧪 四、实验结果

### ✅ 数据集

| Dataset            | Train | Test | Resolution    | Input T | Output T' |
| ------------------ | ----- | ---- | ------------- | ------- | --------- |
| Moving MNIST       | 10k   | 10k  | (1, 64, 64)   | 10      | 10        |
| TrafficBJ          | 19k   | 1.3k | (2, 32, 32)   | 4       | 4         |
| Human3.6           | 2.6k  | 1.1k | (3, 128, 128) | 4       | 4         |
| Caltech Pedestrian | 2k    | 2k   | (3, 128, 160) | 10      | 1         |
| KTH                | 5.2k  | 3.1k | (1, 128, 128) | 10      | 20 / 40   |

---

### 🧮 性能比较（Moving MNIST）

| Method    | MSE ↓    | SSIM ↑    | Framework    |
| --------- | -------- | --------- | ------------ |
| ConvLSTM  | 103.3    | 0.707     | RNN          |
| PredRNN   | 56.8     | 0.867     | RNN          |
| MIM       | 44.2     | 0.910     | RNN          |
| PhyDNet   | 24.4     | 0.947     | CNN-RNN      |
| CrevNet   | 22.3     | 0.949     | CNN-Flow     |
| **SimVP** | **23.8** | **0.948** | **Pure CNN** |

👉 **结论**：SimVP 以更简单的结构达到了与最优模型接近的性能，甚至在计算效率上显著优越。

---

### ⚙️ 计算效率

| Method    | Memory / sample | FLOPs / frame | Training time |
| --------- | --------------- | ------------- | ------------- |
| PhyDNet   | 200 MB          | 1.63G         | ≈10d          |
| CrevNet   | 224 MB          | 1.65G         | ≈10d          |
| **SimVP** | **412 MB**      | **1.68G**     | **≈2d**       |

---

### 🌍 泛化性能（Caltech Pedestrian）

| Method    | MSE ↓    | SSIM ↑    | PSNR ↑   |
| --------- | -------- | --------- | -------- |
| STMFANet  | -        | 0.927     | 29.1     |
| **SimVP** | **1.56** | **0.940** | **33.1** |

📈 **提升：**

- SSIM 提升 1.4%
- PSNR 提升 13%
- 训练时间仅 4 小时

---

### ⏱ 长期预测能力（KTH）

| Setting | SSIM ↑    | PSNR ↑    |
| ------- | --------- | --------- |
| (10→20) | **0.905** | **33.72** |
| (10→40) | **0.886** | **32.93** |

相比 E3D-LSTM (29.31 PSNR)，SimVP 提升 **>11%**。

---

## 🧩 五、消融实验 (Ablation Study)

研究了以下关键因素：

1. Spatial & Temporal UNet shortcuts  
2. Group Normalization / Group Convolution  
3. Inception kernel 大小 (3,5,7,11)
4. Hidden dimension

🧠 **主要发现：**

- GroupConv 对性能贡献最大  
- 大卷积核与多尺度结构显著提升性能  
- Encoder 擅长去除背景误差，Decoder 优化前景形状，Translator 控制物体位置与内容

---

## 🧾 六、结论与启示

> **结论：**

- SimVP 展示了“Simple is Powerful”的理念  
- 无需复杂结构即可实现 SOTA 性能  
- 计算效率高、易于扩展到真实场景  
- 可作为视频预测领域的 **新基线模型**

> **启示：**

- 深度学习的发展可能过度追求复杂性  
- 回归简单结构也能带来突破性进展  
- CNN 仍具备强大的时空建模潜力

---

## 📚 参考公式回顾

视频预测目标函数：
$$
\Theta^* = \arg\min_\Theta \|F_\Theta(X_{t,T}) - Y_{t,T'}\|_2^2
$$

Encoder 层：
$$
z_i = \sigma(\text{LayerNorm}(\text{Conv2d}(z_{i-1})))
$$

Translator 层：
$$
z_j = \text{Inception}(z_{j-1})
$$

Decoder 层：
$$
z_k = \sigma(\text{GroupNorm}(\text{ConvTranspose2d}(z_{k-1})))
$$

---

## 🧩 七、关键词

`Video Prediction` · `CNN` · `Inception Module` · `Simplicity` · `Efficiency` · `Baseline Model`

---
