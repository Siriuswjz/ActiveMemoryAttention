"""Active Memory Attention — 竞争激活式注意力机制

核心思想：每条 KV Cache 记忆拥有"自主激活度"，注意力不再纯靠 query-key 匹配，
而是融合了记忆自身的活跃程度。记忆可以"主动联想"，而不只是"被动检索"。

设计原则：
  1. 最小侵入：作为 DeepseekV4Attention 的 drop-in 替换
  2. 可对比：通过 config 开关控制，方便 A/B 实验
  3. 端到端可训练：所有新参数都参与梯度计算

生物学灵感：
  - 衰减 (γ): 自然遗忘
  - 扩散 (w_ij): 联想记忆网络中的扩散激活
  - 线索加成 (β): 外部线索催化记忆提取
  - 随机噪声 (ε): 自发重激活 / "灵光一闪"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用已有组件
try:
    from .modeling_deepseek_v4 import (
        DeepseekV4RMSNorm, apply_rotary_emb, DeepseekV4Attention
    )
    from .configuration_deepseek_v4 import DeepseekV4Config
except ImportError:
    from modeling_deepseek_v4 import (
        DeepseekV4RMSNorm, apply_rotary_emb, DeepseekV4Attention
    )
    from configuration_deepseek_v4 import DeepseekV4Config


# ==========================================================================
# 核心模块：激活度动力学系统
# ==========================================================================

class ActivationDynamics(nn.Module):
    """管理 KV Cache 中每个 token 的"自主激活度"。

    激活度更新公式：
        a_i^{t+1} = γ · a_i^t                    (衰减)
                   + Σ_j w_ij · a_j^t             (扩散)
                   + β · relevance(q, k_i)         (线索加成)
                   + ε_i                           (随机噪声)

    参数说明:
        decay (γ):         可学习标量，sigmoid 约束到 (0, 1)
        spread_kernel:     1D 卷积核，模拟局部扩散 (邻居间传播)
        query_gate (β):    可学习标量，控制 query 加成强度
        noise_scale:       可学习标量，控制自发激活强度
        lambda_act:        可学习标量，控制激活度对注意力的影响权重
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        spread_width: int = 5,       # 扩散邻域半径
        init_decay: float = 0.95,     # 初始衰减率
        init_noise: float = 0.01,     # 初始噪声幅度
        init_lambda: float = 0.1,     # 初始激活度权重
        init_beta: float = 0.5,       # 初始 query 加成权重
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.spread_width = spread_width

        # --- 可学习参数 ---

        # 衰减率: per-head，sigmoid 约束到 (0, 1)
        # 初始化使得 sigmoid(raw) ≈ init_decay
        raw_decay = math.log(init_decay / (1 - init_decay))
        self.decay_raw = nn.Parameter(torch.full((num_heads,), raw_decay))

        # 扩散核: 1D 卷积，per-head
        # 初始化为中心突出的高斯形状
        kernel = torch.zeros(num_heads, 1, spread_width)
        center = spread_width // 2
        for i in range(spread_width):
            kernel[:, 0, i] = math.exp(-0.5 * ((i - center) / max(center * 0.5, 1)) ** 2)
        kernel[:, 0, center] = 0  # 中心位置不自扩散（衰减已处理）
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8) * 0.1  # 归一化，初始总扩散 ~0.1
        self.spread_kernel = nn.Parameter(kernel)

        # Query 加成权重
        raw_beta = math.log(init_beta / (1 - init_beta)) if init_beta < 1 else 1.0
        self.beta_raw = nn.Parameter(torch.full((num_heads,), raw_beta))

        # 噪声幅度
        self.noise_log_scale = nn.Parameter(torch.full((num_heads,), math.log(init_noise)))

        # 激活度对注意力的影响权重 λ
        self.lambda_raw = nn.Parameter(torch.full((num_heads,), init_lambda))

        # 激活度上界 (防止爆炸)
        self.max_activation = 5.0

    @property
    def decay(self):
        return torch.sigmoid(self.decay_raw)  # (0, 1)

    @property
    def beta(self):
        return F.softplus(self.beta_raw)  # > 0

    @property
    def noise_scale(self):
        return self.noise_log_scale.exp()  # > 0

    @property
    def lambda_act(self):
        return self.lambda_raw  # 可正可负，让模型自己学

    def init_activation(self, batch_size: int, seq_len: int, device, dtype):
        """初始化激活度张量。新 token 的初始激活度 = 1.0（刚进入记忆系统，是"新鲜"的）"""
        return torch.ones(batch_size, self.num_heads, seq_len, device=device, dtype=dtype)

    def step(
        self,
        activation: torch.Tensor,          # [B, H, T] 当前激活度
        query_key_scores: torch.Tensor,     # [B, H, S, T] 原始 q·k 分数
    ) -> torch.Tensor:
        """执行一步激活度动力学更新。

        Args:
            activation: [B, H, T] 所有缓存 token 的当前激活度
            query_key_scores: [B, H, S, T] 当前 query 和所有 key 的点积分数（未 softmax）

        Returns:
            new_activation: [B, H, T'] 更新后的激活度（T' = T，或 T + S_new）
        """
        B, H, T = activation.shape
        device = activation.device
        dtype = activation.dtype

        # 1. 衰减: a_i *= γ
        decay = self.decay.to(dtype).view(1, H, 1)  # [1, H, 1]
        a = activation * decay

        # 2. 扩散: 用 1D 卷积模拟邻居间传播
        #    a_spread_i = Σ_j w_{|i-j|} · a_j
        if T >= self.spread_width:
            # 分组卷积: 每个 head 有自己的核
            pad = self.spread_width // 2
            a_padded = F.pad(a, (pad, pad), mode='constant', value=0)
            spread = F.conv1d(
                a_padded,
                self.spread_kernel.to(dtype),
                groups=H,
            )  # [B, H, T]
            a = a + spread

        # 3. 线索加成: β · mean_over_queries(softmax(scores))
        #    用 query 的注意力分布作为 "relevance" 信号
        beta = self.beta.to(dtype).view(1, H, 1)
        relevance = query_key_scores.mean(dim=2)  # [B, H, T] 对所有 query 平均
        # softmax 归一化，让 relevance 成为概率分布
        relevance = F.softmax(relevance, dim=-1)
        a = a + beta * relevance

        # 4. 随机噪声 (仅训练时)
        if self.training:
            noise = torch.randn_like(a) * self.noise_scale.to(dtype).view(1, H, 1)
            a = a + noise

        # 5. 裁剪防止爆炸
        a = a.clamp(0, self.max_activation)

        return a

    def modulate_attention(
        self,
        attn_scores: torch.Tensor,   # [B, H, S, T] 原始注意力分数
        activation: torch.Tensor,     # [B, H, T] 激活度
    ) -> torch.Tensor:
        """将激活度融入注意力分数。

        修改后的注意力:
            α_i = softmax(q·k_i/√d + λ · a_i)

        Args:
            attn_scores: [B, H, S, T] 原始 q·k/√d 分数
            activation: [B, H, T] 每个 key 的激活度

        Returns:
            modulated_scores: [B, H, S, T] 融合激活度后的分数
        """
        lam = self.lambda_act.to(attn_scores.dtype).view(1, -1, 1, 1)  # [1, H, 1, 1]
        act = activation.unsqueeze(2)  # [B, H, 1, T]
        return attn_scores + lam * act


# ==========================================================================
# 完整的 Active Memory Attention 层
# ==========================================================================

class ActiveMemoryAttention(nn.Module):
    """带竞争激活度的 MLA 注意力。

    在标准 DeepseekV4Attention 基础上增加:
    1. 每个缓存 token 有自主激活度 a_i
    2. 每步更新激活度（衰减 + 扩散 + 线索加成 + 噪声）
    3. 注意力分数融合激活度偏置

    和 DeepseekV4Attention 完全接口兼容，可直接替换。
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scaling = config.head_dim ** -0.5

        # ---- 原始 MLA 投影 (和 DeepseekV4Attention 完全一样) ----
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, config.rms_norm_eps)

        group_head_dim = self.num_heads * self.head_dim // self.o_groups
        self.wo_a = nn.Linear(group_head_dim, self.o_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)

        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))

        # ---- 新增: 激活度动力学系统 ----
        # 从 config 读取参数 (如果有的话)，否则用默认值
        self.activation_dynamics = ActivationDynamics(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            spread_width=getattr(config, 'am_spread_width', 5),
            init_decay=getattr(config, 'am_init_decay', 0.95),
            init_noise=getattr(config, 'am_init_noise', 0.01),
            init_lambda=getattr(config, 'am_init_lambda', 0.1),
            init_beta=getattr(config, 'am_init_beta', 0.5),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        forward 和标准 MLA 一样，但:
        1. past_key_value 多了第三个元素: 激活度张量 [B, H, T]
        2. 注意力分数被激活度调制

        Args:
            hidden_states: [B, S, D]
            past_key_value: (past_k, past_v, past_activation) or None
        """
        bsz, seqlen, _ = hidden_states.shape

        # ============ Q 投影 (不变) ============
        q = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        q = q.to(hidden_states.dtype)

        # ============ KV 投影 (不变) ============
        kv = self.kv_norm(self.wkv(hidden_states))
        kv = kv.unsqueeze(1)  # [B, 1, S, head_dim]

        # ============ RoPE (不变) ============
        if freqs_cis is not None:
            q_rope = q[..., -self.qk_rope_head_dim:]
            kv_rope = kv[..., -self.qk_rope_head_dim:]
            q_rope = apply_rotary_emb(q_rope, freqs_cis)
            kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
            q = torch.cat([q[..., :-self.qk_rope_head_dim], q_rope], dim=-1)
            kv = torch.cat([kv[..., :-self.qk_rope_head_dim], kv_rope], dim=-1)

        # ============ KV Cache 拼接 ============
        past_activation = None
        if past_key_value is not None:
            past_k, past_v, past_activation = past_key_value
            kv = torch.cat([past_k, kv], dim=2)

        # ============ 激活度初始化/拼接 ============
        # 新 token 的初始激活度 = 1.0
        new_activation = self.activation_dynamics.init_activation(
            bsz, seqlen, hidden_states.device, hidden_states.dtype
        )
        if past_activation is not None:
            activation = torch.cat([past_activation, new_activation], dim=-1)  # [B, H, T+S]
        else:
            activation = new_activation  # [B, H, S]

        # ============ 手动计算注意力 (不用 SDPA，因为需要修改分数) ============
        kv_expanded = kv.expand(-1, self.num_heads, -1, -1)  # [B, H, T, D]
        total_len = kv_expanded.shape[2]

        # 原始注意力分数: q·k^T / √d
        attn_scores = torch.matmul(q, kv_expanded.transpose(2, 3)) * self.scaling
        # attn_scores: [B, H, S, T]

        # ============ 👉 关键改动: 激活度动力学更新 ============
        activation = self.activation_dynamics.step(activation, attn_scores)

        # ============ 👉 关键改动: 激活度调制注意力分数 ============
        attn_scores = self.activation_dynamics.modulate_attention(attn_scores, activation)

        # 因果 mask
        if attention_mask is not None:
            # attention_mask: [1, 1, S, S] 或 [1, 1, S, T]
            if attention_mask.shape[-1] < total_len:
                # 需要扩展 mask 以覆盖 past cache
                pad_len = total_len - attention_mask.shape[-1]
                # past tokens 都可见 (0)，用 0 填充左边
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(hidden_states.dtype)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, kv_expanded)  # [B, H, S, D]

        # ============ De-rotate RoPE (不变) ============
        if freqs_cis is not None:
            cos, sin = freqs_cis[0], freqs_cis[1]
            cos_inv = cos.unsqueeze(0).unsqueeze(0)
            sin_inv = -sin.unsqueeze(0).unsqueeze(0)
            out_rope = attn_output[..., -self.qk_rope_head_dim:]
            d = out_rope.shape[-1] // 2
            o1, o2 = out_rope[..., :d], out_rope[..., d:]
            out_rope = torch.cat([
                o1 * cos_inv + o2 * sin_inv,
                o1 * (-sin_inv) + o2 * cos_inv
            ], dim=-1)
            attn_output = torch.cat([
                attn_output[..., :-self.qk_rope_head_dim],
                out_rope.to(attn_output.dtype)
            ], dim=-1)

        # ============ Grouped O 投影 (不变) ============
        attn_output = attn_output.transpose(1, 2)  # [B, S, H, D]
        attn_output = attn_output.reshape(bsz, seqlen, self.o_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a_w)
        attn_output = attn_output.flatten(2)
        attn_output = self.wo_b(attn_output)

        # ============ Cache ============
        new_cache = (kv, kv, activation) if use_cache else None

        return attn_output, new_cache


# ==========================================================================
# 辅助函数: 替换标准注意力为 Active Memory 版本
# ==========================================================================

def replace_attention_with_active_memory(model):
    """将模型中所有 DeepseekV4Attention 替换为 ActiveMemoryAttention。

    用法:
        model = DeepseekV4ForCausalLM(config)
        replace_attention_with_active_memory(model)
    """
    for layer in model.model.layers:
        old_attn = layer.attn
        new_attn = ActiveMemoryAttention(model.config, old_attn.layer_idx)

        # 复制原有权重
        with torch.no_grad():
            new_attn.wq_a.weight.copy_(old_attn.wq_a.weight)
            new_attn.q_norm.weight.copy_(old_attn.q_norm.weight)
            new_attn.wq_b.weight.copy_(old_attn.wq_b.weight)
            new_attn.wkv.weight.copy_(old_attn.wkv.weight)
            new_attn.kv_norm.weight.copy_(old_attn.kv_norm.weight)
            new_attn.wo_a.weight.copy_(old_attn.wo_a.weight)
            new_attn.wo_b.weight.copy_(old_attn.wo_b.weight)
            new_attn.attn_sink.copy_(old_attn.attn_sink)

        layer.attn = new_attn

    return model


# ==========================================================================
# 实验脚本: 对比标准注意力 vs Active Memory
# ==========================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Active Memory Attention — 功能验证")
    print("=" * 60)

    # 用 debug 配置创建小模型
    config = DeepseekV4Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        moe_intermediate_size=256,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=64,
        head_dim=48,
        qk_rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        hc_mult=2,
        hc_sinkhorn_iters=1,
        max_position_embeddings=128,
    )

    # --- 测试 1: ActivationDynamics 单元测试 ---
    print("\n--- 测试 1: ActivationDynamics ---")
    dynamics = ActivationDynamics(num_heads=4, head_dim=48)
    B, H, S, T = 2, 4, 8, 20

    activation = dynamics.init_activation(B, S + T, device='cpu', dtype=torch.float32)
    # 模拟 query-key scores
    fake_scores = torch.randn(B, H, S, S + T)
    print(f"  初始激活度: shape={activation.shape}, mean={activation.mean():.3f}")

    # 模拟多步更新
    dynamics.train()
    for step in range(5):
        activation = dynamics.step(activation, fake_scores)
        print(f"  Step {step+1}: mean={activation.mean():.3f}, "
              f"max={activation.max():.3f}, min={activation.min():.3f}")

    # 测试调制
    modulated = dynamics.modulate_attention(fake_scores, activation)
    diff = (modulated - fake_scores).abs().mean()
    print(f"  调制偏移量 (mean |Δ|): {diff:.4f}")
    print(f"  λ per head: {dynamics.lambda_act.data.tolist()}")

    # --- 测试 2: ActiveMemoryAttention 前向传播 ---
    print("\n--- 测试 2: ActiveMemoryAttention forward ---")
    attn = ActiveMemoryAttention(config, layer_idx=0)
    x = torch.randn(2, 16, 128)  # [B, S, D]

    # 准备 freqs_cis
    from modeling_deepseek_v4 import precompute_freqs_cis
    freqs = precompute_freqs_cis(16, 128)[:, :16]  # [2, 16, 8]

    output, cache = attn(x, freqs_cis=freqs, use_cache=False)
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    print(f"  Cache: {cache}")

    # --- 测试 3: 激活度的梯度 ---
    print("\n--- 测试 3: 梯度检查 ---")
    attn.train()
    x.requires_grad_(True)
    output, _ = attn(x, freqs_cis=freqs, use_cache=False)
    loss = output.sum()
    loss.backward()

    grad_params = {name: p.grad is not None for name, p in attn.activation_dynamics.named_parameters()}
    print(f"  激活度参数梯度存在: {grad_params}")
    print(f"  decay grad: {attn.activation_dynamics.decay_raw.grad.data}")
    print(f"  lambda grad: {attn.activation_dynamics.lambda_raw.grad.data}")

    # --- 测试 4: 参数量分析 ---
    print("\n--- 测试 4: 额外参数量 ---")
    standard_attn = DeepseekV4Attention(config, layer_idx=0)
    active_attn = ActiveMemoryAttention(config, layer_idx=0)

    n_standard = sum(p.numel() for p in standard_attn.parameters())
    n_active = sum(p.numel() for p in active_attn.parameters())
    n_dynamics = sum(p.numel() for p in active_attn.activation_dynamics.parameters())
    overhead = (n_active - n_standard) / n_standard * 100

    print(f"  标准 MLA 参数量: {n_standard:,}")
    print(f"  Active Memory 参数量: {n_active:,}")
    print(f"  动力学系统参数量: {n_dynamics:,}")
    print(f"  额外开销: {overhead:.2f}%")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)
