# Active Memory Attention — 设计文档与对比实验方案

## 目录

- [1. 架构设计](#1-架构设计)
- [2. 核心公式](#2-核心公式)
- [3. 实现细节](#3-实现细节)
- [4. 对比实验设计](#4-对比实验设计)
- [5. 实验操作指南](#5-实验操作指南)

---

## 1. 架构设计

### 标准 MLA vs Active Memory MLA

```
标准 MLA:
  q·k^T/√d → softmax → 注意力权重 → 加权求和 V
  (纯匹配，没有记忆，每步独立)

Active Memory MLA:
  q·k^T/√d ──────────────────┐
       │                      │
       ↓                      ↓
  激活度动力学系统         注意力分数调制
  ┌──────────────────┐    ┌───────────────┐
  │ a_i *= γ (衰减)   │    │               │
  │ a_i += 扩散(邻居)  │───→│ score += λ·a  │→ softmax → 加权 V
  │ a_i += β·relevance │    │               │
  │ a_i += ε (噪声)    │    └───────────────┘
  └──────────────────┘
       ↓
  a_i 存入 KV Cache（跨步持久化）
```

### 生物学灵感

| 组件 | 生物对应 | 在 AI 中的作用 |
|---|---|---|
| 衰减 (γ) | 自然遗忘 | 旧记忆自动减弱 |
| 扩散 (w_ij) | 联想网络扩散激活 | 相邻 token 互相激活 |
| 线索加成 (β) | 外部线索催化记忆提取 | query 匹配加强激活 |
| 随机噪声 (ε) | 自发重激活 / "灵光一闪" | 训练时探索性激活 |
| 激活度权重 (λ) | 激活度对意识的影响力 | 高激活记忆更容易被注意 |

### 关键数字

| 指标 | 数值 |
|---|---|
| 新增参数量 | **36 个**（衰减率、扩散核、β、噪声、λ） |
| 相对开销 | **0.09%** |
| 所有参数有梯度 | ✅ 端到端可训练 |
| 接口兼容 | ✅ drop-in 替换 DeepseekV4Attention |

---

## 2. 核心公式

### 激活度更新

$$a_i^{(t+1)} = \underbrace{\gamma \cdot a_i^{(t)}}_{\text{衰减}} + \underbrace{\sum_{j \in \text{neighbors}} w_{ij} \cdot a_j^{(t)}}_{\text{扩散激活}} + \underbrace{\beta \cdot \text{relevance}(q, k_i)}_{\text{线索加成}} + \underbrace{\epsilon_i}_{\text{随机噪声}}$$

### 调制后的注意力

$$\alpha_i = \text{softmax}\left(\frac{q \cdot k_i^T}{\sqrt{d}} + \lambda \cdot a_i\right)$$

### 和标准 attention 的区别

| 标准 attention | Active Memory attention |
|---|---|
| α_i = softmax(q·k_i^T/√d) | α_i = softmax(q·k_i^T/√d + **λ·a_i**) |
| 每步独立，无状态 | 激活度跨步累积，有状态 |
| 纯被动检索 | 主动联想 + 被动检索 |

---

## 3. 实现细节

### 五个可学习参数

| 参数 | 形状 | 约束 | 初始值 | 说明 |
|---|---|---|---|---|
| decay_raw → γ | [H] per-head | sigmoid → (0,1) | 0.95 | 每步衰减 5% |
| spread_kernel | [H, 1, W] | 无 | 高斯形 | W=5 的 1D 卷积核 |
| beta_raw → β | [H] per-head | softplus → >0 | 0.5 | query 加成强度 |
| noise_log_scale → ε | [H] per-head | exp → >0 | 0.01 | 仅训练时生效 |
| lambda_raw → λ | [H] per-head | 无约束 | 0.1 | 可正可负，模型自学 |

### 使用方式

```python
# 方式 1: 替换已有模型
from active_memory_attention import replace_attention_with_active_memory
model = DeepseekV4ForCausalLM(config)
replace_attention_with_active_memory(model)  # 原有权重自动复制

# 方式 2: 单独使用
from active_memory_attention import ActiveMemoryAttention
attn = ActiveMemoryAttention(config, layer_idx=0)
output, cache = attn(hidden_states, freqs_cis=freqs)
```

### KV Cache 扩展

```
标准 KV Cache:  (K, V)                → 2 个张量
Active Memory:  (K, V, activation)     → 3 个张量（多了激活度）

activation shape: [B, H, T]
额外显存: B × H × T × 4 bytes (float32)
对于 B=1, H=8, T=2048: 仅 64 KB
```

---

## 4. 对比实验设计

### 第一层：基础指标（必测）

| 指标 | 怎么测 | 预期 |
|---|---|---|
| **困惑度 (PPL)** | held-out 文本 | Active Memory ≤ 标准版 |
| **训练 loss 曲线** | 对比收敛速度 | 可能收敛更快 |
| **参数效率** | 同 loss 所需步数 | 可能更少 |
| **训练速度** | wall-clock/step | Active Memory 会慢（手动 attention 替代 SDPA） |

### 第二层：针对性能力测试

#### 测试 1: 远距离信息回忆（Long-Range Recall）

```
输入:
  "小明喜欢吃苹果。
   [中间插入 N 个无关 token]
   小明今天午饭最可能吃什么？"

变量: N = 100, 500, 1000
指标: 回答准确率 vs 距离的衰减曲线
```

标准 attention 在距离增大时准确率快速下降。Active Memory 的扩散激活应使"小明"+"苹果"在被重新提及时仍保持较高激活度。

#### 测试 2: 隐式关联推理（Implicit Association）

```
1 跳: "张三是医生。王五生病了，找谁？" → 张三
2 跳: "张三的妻子在医院工作。王五生病了，可以找谁的妻子？" → 张三
3 跳: "张三的妻子在医院工作，医院对面是药店。王五要买药去哪？" → 药店
```

Active Memory 的扩散机制在多跳推理中优势更大——激活沿关联链传播。

#### 测试 3: 干扰信息抗干扰（Distractor Robustness）

```
输入:
  "答案是苹果。
   [大量提到香蕉、橘子、西瓜的文本]
   请问答案是什么？"

变量: 干扰项数量 = 0, 5, 20
指标: 准确率
```

标准 attention 中高频词可能干扰。Active Memory 中原始"答案是苹果"的记忆可通过"答案是什么"的 query 加成重新激活。

#### 测试 4: "灵光一闪"测试（Spontaneous Recall）

```
轮次 1: "今天学了个词叫 serendipity，意思是意外发现美好事物。"
轮次 2: "我要写科学发现的文章，有什么好的形容词？"
  (不直接提及 serendipity)
```

标准版需显式匹配。Active Memory 中 serendipity 可能通过噪声/扩散自发浮现。

### 第三层：激活度行为分析（可视化）

#### 测试 5: 激活度热力图

```python
# 收集每步每个 token 的激活度
activations = []
for step in generation_steps:
    activations.append(model.get_activation_snapshot())

# 画热力图
plt.imshow(activations, aspect='auto')
plt.xlabel('Token Position')
plt.ylabel('Generation Step')
plt.title('Activation Dynamics Over Time')
```

观察要点:
- 重要 token（实体名、数字）激活度是否自然更高
- 是否形成"激活簇"（相关 token 互相激活）
- 衰减速率是否合理

#### 测试 6: 学习到的参数分析

```python
for layer_idx, layer in enumerate(model.layers):
    d = layer.attn.activation_dynamics
    print(f"Layer {layer_idx}:")
    print(f"  decay (γ): {d.decay.data}")       # 每层遗忘不同？
    print(f"  lambda (λ): {d.lambda_act.data}")  # 哪些层更依赖联想？
    print(f"  beta (β): {d.beta.data}")          # query 加成多强？
    print(f"  noise: {d.noise_scale.data}")       # 噪声多大？
```

观察要点:
- 浅层 vs 深层的 λ 差异（深层可能更依赖联想）
- 不同 head 学到不同衰减率（有的专注近期，有的记忆更久）
- 噪声是否被模型学到接近 0（说明不需要）还是保持（说明有用）

### 第四层：消融实验（Ablation Study）

| 实验 | 改动 | 目的 |
|---|---|---|
| Full Active Memory | 全部开启 | baseline |
| 无衰减 | γ = 1.0 | 遗忘是否有帮助 |
| 无扩散 | spread_kernel = 0 | 联想扩散是否有用 |
| 无噪声 | ε = 0 | 自发激活是否有用 |
| 无线索加成 | β = 0 | query 反馈是否重要 |
| 只有 λ 偏置 | 去掉动力学，a_i 固定可学 | 动力学 vs 静态偏置 |

消融实验可以回答一个关键问题：**Active Memory 的收益来自哪个组件？** 如果去掉扩散后性能不变，说明联想不重要；如果去掉衰减后反而更好，说明"遗忘"在这个规模下不必要。

---

## 5. 实验操作指南

### 阶段 1: 快速验证（debug 配置）

```bash
# 标准版 baseline
python scripts/train_pretrain.py --config configs/debug.yaml \
  --output_dir checkpoints/baseline_debug

# Active Memory 版
python scripts/train_pretrain.py --config configs/debug.yaml \
  --output_dir checkpoints/active_memory_debug
# (需要在 train 脚本中添加 replace_attention_with_active_memory 调用)
```

50 步即可验证:
- 训练是否稳定（不出 NaN）
- loss 下降趋势有无差异
- 每步速度差多少

### 阶段 2: 正式对比（main_100m 配置）

```bash
# 标准版
python scripts/train_pretrain.py --config configs/main_100m.yaml \
  --output_dir checkpoints/baseline_100m

# Active Memory 版
python scripts/train_pretrain.py --config configs/main_100m.yaml \
  --output_dir checkpoints/active_memory_100m
```

两个模型各训 5000 步，然后运行全部测试。

### 阶段 3: 消融实验

在 Active Memory 版本基础上，逐个关闭组件重新训练。只需修改 `ActivationDynamics` 的初始化参数。

### 结果记录模板

```
| 实验 | PPL ↓ | Loss ↓ | 远距回忆(500) | 多跳推理(2跳) | 抗干扰(20) | ms/step |
|---|---|---|---|---|---|---|
| Standard MLA | - | - | - | - | - | - |
| Active Memory (full) | - | - | - | - | - | - |
| AM (no decay) | - | - | - | - | - | - |
| AM (no spread) | - | - | - | - | - | - |
| AM (no noise) | - | - | - | - | - | - |
| AM (no beta) | - | - | - | - | - | - |
| AM (static bias) | - | - | - | - | - | - |
```

---

*Active Memory Attention — nanowhale project, May 2026*
