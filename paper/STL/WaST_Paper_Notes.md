# Wavelet-Driven Spatiotemporal Predictive Learning (WaST) 论文笔记

**论文信息：**  
Nie 等, AAAI 2024  
**标题：** *Wavelet-Driven Spatiotemporal Predictive Learning: Bridging Frequency and Time Variations*  

---

## 🧠 一、研究背景与动机

传统的视频预测模型（如 ConvLSTM、SimVP、TAU）通常在**空间域**进行时空特征学习。  
但这类方法往往将 **时间变化（Temporal Variation）** 和 **频率变化（Frequency Variation）** 混合在一起，导致模型难以同时处理慢变化（低频）与快变化（高频）信号。

WaST 的核心思想是：

> **通过小波变换（Wavelet Transform）显式地将视频的时频信息解耦，并在低频部分进行时频联合建模。**

---

## ⚙️ 二、整体结构概览

<img title="" src="file:///Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-28-22-13-45-image.png" alt="" width="468" data-align="center">

整个网络流程如下：

$$
X_{input} \rightarrow TF\text{-}Aware\ Translator_1 \rightarrow 3D\text{-}Wavelet\ Embed \rightarrow TF\text{-}Aware\ Translator_2 \rightarrow Wavelet\ Bottleneck \rightarrow TF\text{-}Aware\ Translator_3 \rightarrow 3D\text{-}Wavelet\ Recon \rightarrow \hat{X}_{output}
$$

### 模块划分：

| 模块                             | 所在域       | 主要功能           |
| ------------------------------ | --------- | -------------- |
| **TF-Aware Translator ①**      | 空间域       | 初步学习时空依赖       |
| **3D-Wavelet Embed**           | 空间域 → 频域  | 小波分解，提取低频与高频分量 |
| **TF-Aware Translator ②**      | 频域（低频部分）  | 建模慢变化的全局动态     |
| **Wavelet Bottleneck**         | 频域（低频的低频） | 深层次全局建模        |
| **3D-Wavelet Recon**           | 频域 → 空间域  | 逆小波重建，融合高频信息   |
| **(可选) TF-Aware Translator ④** | 空间域       | 输出细化与误差修正      |

---

## 🌀 三、3D-Wavelet Embedding

**3D 离散小波变换（3D-DWT）** 对视频张量  
$X \in \mathbb{R}^{B \times T \times C \times H \times W}$  
进行时空域的分解，得到 8 个子带：

$$
\{ X_{LLL}, X_{LLH}, X_{LHL}, X_{LHH}, X_{HLL}, X_{HLH}, X_{HHL}, X_{HHH} \}
$$

- $X_{LLL}$ 表示低频近似部分（全局结构、慢变化）；  
- 其他 7 个为高频部分（局部纹理、边缘、快速运动）。

模型仅将 **低频部分 $X_{LLL}$** 送入后续主干建模，高频部分缓存以备重建。

---

## 🔺 四、TF-Aware Translator 模块

TF-Aware Translator 是 WaST 的核心构件，包含两部分：

### (1) Frequency Mixer（频率混合器）

通过大核卷积与小核卷积模拟低频与高频响应：

$$
Y_{LF} = Conv_{k=5}(X), \quad Y_{HF} = Conv_{k=1}(X)
$$

并通过频率注意力（Frequency Attention）进行融合，增强不同方向的高频特征响应。

---

### (2) Temporal Mixer（时间混合器）

在时间维度上进行注意力建模：

$$
A_t = Softmax(Q_t K_t^T / \sqrt{d}), \quad Y_t = A_t V_t
$$

结合 FFN 与残差结构，从而捕捉帧间依赖与时间动态。

> ✅ 简而言之，TF-Aware Translator = TAU + Frequency Attention。

---

## 🔁 五、Wavelet Bottleneck 模块

该模块进一步处理低频主干特征，步骤如下：

1. **再次进行 3D 小波分解：**
   
   $$
   (Ca, Cd) = DWT3D(X_{LLL})
   $$

2. **在 $Ca$（低频的低频）上应用 TF-Aware Translator。**

3. **再通过逆小波变换重建：**
   
   $$
   X'_{LLL} = IDWT3D(Ca', Cd)
   $$

通过这种方式，模型能够在深层低频空间中学习全局一致的时间变化。

---

## 🧩 六、3D-Wavelet Reconstruction 模块

最后，将经过 Wavelet Bottleneck 处理后的低频部分  
与先前缓存的高频部分 $\{ X_{LLH}, X_{LHL}, ..., X_{HHH} \}$  
一起进行逆小波重建：

$$
\hat{X} = IDWT3D(X'_{LLL}, \{ X_{LLH}, X_{LHL}, ..., X_{HHH} \})
$$

输出回到空间域，恢复出细节清晰的预测帧。

---

## 🧠 七、损失函数 — 高频聚焦损失（HFFL）

HFFL（High-Frequency Focal Loss）强化模型对高频区域（边缘、运动边界）的关注。

损失定义为：

$$
\mathcal{L}_{HFFL} = \sum_i w_i \cdot \| X_i - \hat{X}_i \|_1, \quad w_i = 1 + \alpha |DWT(X_i)|
$$

整体训练目标：

$$
\mathcal{L} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{HFFL}
$$

---

## 📊 八、消融实验结果（Ablation Study）

| 模型                    | MSE ↓    | PSNR ↑   | SSIM ↑    | 主要提升来源 |
| --------------------- | -------- | -------- | --------- | ------ |
| SimVP 基线              | 89.4     | 28.2     | 0.902     | —      |
| + TF-Aware Translator | 84.7     | 29.0     | 0.911     | 时间注意力  |
| + Wavelet 架构          | 78.3     | 29.9     | 0.923     | 时频分解   |
| + HFFL (完整 WaST)      | **73.5** | **30.6** | **0.933** | 高频细节强化 |

🔹 性能提升主要来自：

1. 小波分解（≈40% 贡献）  
2. TF-Aware Translator 注意力建模（≈35%）  
3. 高频聚焦损失（≈25%）

---

## 🎯 九、总结与创新点

- **时频域显式解耦：** 通过 3D 小波变换分离低/高频信息；
- **TF-Aware Translator：** 同时建模时间变化与频率特征；
- **Wavelet Bottleneck：** 分层建模全局与局部动态；
- **HFFL：** 优化阶段增强高频细节还原；
- **整体优势：** 提高预测帧清晰度与动态一致性。

---

## 📚 十、相关工作参考

- **Nie et al. (AAAI 2024)** — *Wavelet-Driven Spatiotemporal Predictive Learning*  
- **Tan et al. (CVPR 2023)** — *Temporal Attention Unit (TAU)*  
- **Gao et al. (CVPR 2022)** — *SimVP: Simpler yet Better Video Prediction*

---
