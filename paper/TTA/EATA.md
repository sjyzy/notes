# 论文阅读记录

## 基本信息

- **论文题目**: Efficient Test-Time Model Adaptation without Forgetting  
- **作者**: Yuejiang Niu, Yifan Zhang, Songyang Zhang, and Xinchao Wang  
- **会议**: ICML 2022  
- **阅读日期**: 2025-10-23

---

## 一、研究背景

在实际部署中，深度神经网络经常面临 **测试分布与训练分布不一致（domain shift）** 的问题，例如图像模糊、噪声或压缩失真。传统方法往往需要重新训练或访问源数据，而这在测试阶段通常是不可能的。

为了解决这一问题，**Test-Time Adaptation (TTA)** 成为研究热点，其中最具代表性的工作是 **Tent (Wang et al., ICLR 2021)**，通过最小化预测熵实现在线自适应。然而，Tent 存在两个关键问题：  

1. 容易在连续适应中 **灾难性遗忘（catastrophic forgetting）**；  
2. 对不确定样本的更新会导致 **错误累积（error accumulation）**。

---

## 二、主要贡献

本文提出 **EATA (Efficient Anti-forgetting Test-time Adaptation)**，在 Tent 的基础上加入防遗忘机制，使模型在多域测试时依然保持稳定。核心贡献包括：

1. **Entropy-based sample filtering**：仅对低熵、高置信样本进行更新，避免错误梯度传播。  
2. **Fisher Information Regularization**：利用 Fisher 信息矩阵约束重要参数的更新幅度，防止参数漂移。  
3. **Efficient selective adaptation**：在保证计算效率的同时，实现稳定的多轮 test-time 学习。

---

## 三、方法概述

EATA 的总体目标函数为：

$$
\min_\theta \mathbb{E}_{x \sim D_t}[H(p_\theta(y|x))] + \lambda \cdot R_{Fisher}(\theta, \theta_0)
$$

其中：  

- 第一项为熵最小化；  
- 第二项为基于 Fisher 信息的正则项，用于防止灾难性遗忘。  

算法仅更新 **BN 层参数（γ、β、均值、方差）**，并在每次迭代时筛选掉高熵样本。

### 4.2 Anti-Forgetting with Fisher Regularization

#### 🧩 背景动机

在测试时自适应（Test-Time Adaptation, TTA）中，模型会在连续的目标域（如不同类型的图像腐蚀）上逐步更新。  
由于每一步都会修改模型参数（尤其是 BN 层参数），如果没有约束，参数可能发生**漂移**，导致模型逐渐“忘记”之前学到的知识，这就是 **灾难性遗忘（catastrophic forgetting）**。

EATA 为了解决这个问题，引入了基于 **Fisher 信息矩阵（Fisher Information Matrix, FIM）** 的正则化约束，在测试阶段防止模型对关键参数的过度修改。

---

#### 🧠 核心思想：基于 EWC 的防遗忘机制

EATA 借鉴了终身学习（Continual Learning）中的 EWC（Elastic Weight Consolidation, Kirkpatrick et al., 2017）思想。  
EWC 的假设是：对旧任务重要的参数，其 **Fisher 信息值 \( F_i \)** 较大，因此在新任务中应避免大幅修改。

在 EATA 中，这一思想被用于测试时适应：  

> 对旧分布（原始模型）中重要的参数，测试时的更新应受到抑制，以防止遗忘。

---

#### 🧮 正则项公式

EATA 的优化目标为：

$$
\min_\theta \; \mathbb{E}_{x \sim D_t}[H(p_\theta(y|x))] + \lambda \cdot R_{\text{Fisher}}(\theta, \theta_0)
$$

其中：

- 第一项是 **熵最小化损失（entropy minimization）**；
- 第二项是 **Fisher 正则项（anti-forgetting regularizer）**。

正则项定义为：

$$
R_{\text{Fisher}}(\theta, \theta_0)
= \frac{1}{2} \sum_i F_i (\theta_i - \theta_{0,i})^2
$$

含义如下：

| 符号                 | 解释                     |
| ------------------ | ---------------------- |
| \( \theta_i \)     | 当前第 i 个参数（正在被更新）       |
| \( \theta_{0,i} \) | 该参数在原始模型中的初始值          |
| \( F_i \)          | 第 i 个参数的重要性（Fisher 信息） |
| \( \lambda \)      | 正则化强度的权重超参数            |

直觉：  

> 重要参数（高 \(F_i\)）不应被大幅修改，  
> 不重要的参数（低 \(F_i\)）可以自由调整以适应新分布。

---

#### ⚙️ Fisher 信息的计算方式

由于测试阶段无法访问源数据和真实标签，EATA 采用基于预测伪标签的 Fisher 信息近似：

$$
F_i \approx \mathbb{E}_{x \sim D_t} 
\left[\left(\frac{\partial \log p_\theta(\hat{y}|x)}{\partial \theta_i}\right)^2\right]
$$

其中：

- \( $\hat{y} = \arg\max_y p_\theta(y|x) $\) 是模型预测的伪标签；
- \( p_\theta(y|x) \) 为预测概率分布；
- 梯度平方表示输出对参数的敏感度。

在实现中，EATA 用一阶近似和指数滑动平均来更新 Fisher 值：

```python
# Fisher 信息的近似更新
for name, param in model.named_parameters():
    fisher[name] = (param.grad ** 2).detach().mean() + alpha * fisher[name].detach()
```

---

## 四、实验与结果

论文在 **CIFAR10-C、CIFAR100-C、ImageNet-C** 等数据集上验证了 EATA。  
相较于 Tent、BN Adapt 等方法，EATA 在所有测试域上都取得显著提升。  

### Figure 3 — Demonstration of Preventing Forgetting

- **左图**：EATA 在多种 corruption 类型上保持稳定的 clean accuracy，而 Tent 波动较大。  

- **右图**：在连续适应多个失真域的过程中，Tent 的 clean 与 corruption accuracy 均迅速下降至近 0%，而 EATA 始终稳定。  

- **结论**：EATA 能有效防止灾难性遗忘并保持对多域的泛化。
  
  ![](/Users/sjyzy/Library/Application%20Support/marktext/images/2025-10-23-20-55-18-image.png)

---

## 五、Tent 的崩溃机制分析

虽然 Tent 仅更新 BN 层参数，但这些参数决定了每层特征的归一化分布。  
当错误样本被反复用于熵最小化时：  

1. BN 均值和方差逐渐漂移；  
2. 激活分布错位，特征空间被扭曲；  
3. 预测置信度虚高但方向错误；  
4. 误差持续累积，最终模型输出完全崩溃。  

EATA 通过 **样本筛选 + Fisher 正则化** 避免了这一问题。

---

## 六、总结与启发

| 方面      | Tent | EATA      |
| ------- | ---- | --------- |
| 样本选择    | 全部样本 | 仅低熵样本     |
| 防遗忘机制   | 无    | Fisher 正则 |
| 连续适应稳定性 | 崩溃   | 稳定        |
| 计算开销    | 低    | 略高但高效     |

**总结**：EATA 提供了一种轻量且有效的测试时自适应策略，既能提升性能又能避免遗忘。其思想（选择性样本更新 + Fisher 约束）对未来的在线自适应与持续学习任务具有重要启示。

---

## 七、个人思考与问题

- BN 层参数虽然少，但主导了特征分布，因此其正确更新至关重要；  
- 未来可考虑结合 **自监督目标** 或 **特征一致性约束** 进一步提升稳定性；  
- 是否可以将 EATA 思想扩展到 **Transformer 架构** 或 **时序任务**，是一个值得探索的方向。
