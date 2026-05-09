"""EarlyExitMoEBlock: 残差跳层 + MoE 路由融合实验模块。

核心想法:
  每层Transformer不仅有自己的Attention和MoE,还有一个Exit Gate。
  Gate决定每个token在这一层是"正常计算"还是"走残差跳过"。
  所有层的输出通过学习的混合权重汇聚到共享MoE,
  让简单token用浅层特征、复杂token用深层特征。

与已有工作的区别:
  - Mixture-of-Depths (Google 2024): 硬选择top-k, 没有跨层混合
  - CALM (Google 2022): 早退出到LM head, 没有MoE融合
  - 本模块: 残差跳层 + 跨层特征加权混合 + MoE路由, 三者统一

用法:
  python scripts/early_exit_moe.py                    # 运行测试
  python scripts/early_exit_moe.py --compare           # 对比原始模型
  python scripts/early_exit_moe.py --config configs/debug.yaml  # 用自定义配置
"""

import sys
import os
import math
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import (
    DeepseekV4RMSNorm,
    DeepseekV4Attention,
    DeepseekV4MoE,
    DeepseekV4Block,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4PreTrainedModel,
    hc_split_sinkhorn,
    precompute_freqs_cis,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin


# ---------------------------------------------------------------------------
# Exit Gate: 每层决定token走不走这层
# ---------------------------------------------------------------------------

class ExitGate(nn.Module):
    """学习每个token在当前层是否需要完整计算。

    输出 exit_prob ∈ (0, 1):
      - 接近 1: 跳过这层 (用残差)
      - 接近 0: 正常计算

    训练时用软混合 (可微), 推理时可硬阈值 (真省算力)。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        # 初始化偏置为负值, 让模型初期倾向于"不跳过"(正常计算)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] 隐藏状态
        Returns:
            exit_prob: [B, S, 1] 跳过概率
        """
        return torch.sigmoid(self.proj(x))


# ---------------------------------------------------------------------------
# LayerContribution: 跨层特征混合权重
# ---------------------------------------------------------------------------

class LayerContribution(nn.Module):
    """学习各层输出对最终MoE输入的贡献权重。

    每层产出一个特征, 所有层的特征通过学习权重加权混合后送入共享MoE。
    这实现了"浅层残差直连MoE输出"的效果。
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        # 每层一个可学习标量权重
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        # LayerNorm 对齐不同层的 feature scale
        self.norm = DeepseekV4RMSNorm(hidden_size)

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_outputs: list of [B, S, D], 各层的输出
        Returns:
            mixed: [B, S, D] 加权混合结果
        """
        weights = F.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(layer_outputs, dim=0)  # [L, B, S, D]
        # weights: [L] -> [L, 1, 1, 1]
        mixed = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return self.norm(mixed)


# ---------------------------------------------------------------------------
# EarlyExitMoEBlock: 带跳过门的Transformer层
# ---------------------------------------------------------------------------

class EarlyExitMoEBlock(nn.Module):
    """带 Exit Gate 的 Transformer Block。

    与标准 DeepseekV4Block 的区别:
    1. 多了 exit_gate: 决定每个token是否跳过
    2. 输出两路: 完整计算的结果 和 跳过(残差)的结果, 软混合
    3. 保留 HC (Hyper-Connections) 机制不变

    训练时: output = (1-p)*computed + p*residual (软混合, 可微)
    推理时: if p > 0.5: skip else: compute       (硬阈值, 省算力)
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.block = DeepseekV4Block(config, layer_idx)
        self.exit_gate = ExitGate(config.hidden_size * config.hc_mult)
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        hard_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], torch.Tensor]:
        """
        Args:
            x: [B, S, hc_mult, D] HC隐藏状态
            hard_exit: 推理时设为True, 启用硬阈值跳过

        Returns:
            output: [B, S, hc_mult, D]
            cache: KV cache
            exit_prob: [B, S, 1] 跳过概率 (用于分析和辅助loss)
        """
        bsz, seqlen = x.shape[:2]
        # Exit gate 在 flatten 的 HC 状态上决策
        x_flat = x.flatten(2)  # [B, S, hc_mult*D]
        exit_prob = self.exit_gate(x_flat)  # [B, S, 1]

        if hard_exit and not self.training:
            # 推理时: 跳过概率 > 0.5 的 token 直接走残差
            skip_mask = (exit_prob > 0.5).squeeze(-1)  # [B, S]
            if skip_mask.all():
                # 所有 token 都跳过, 完全不算这层
                return x, None, exit_prob

            if not skip_mask.any():
                # 没人跳过, 正常算
                computed, cache = self.block(
                    x, attention_mask=attention_mask, position_ids=position_ids,
                    freqs_cis=freqs_cis, past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                return computed, cache, exit_prob

            # 部分跳过: 只对需要计算的 token 跑 block
            # (完整实现需要 scatter/gather, 这里用软混合近似, 保持简单)
            computed, cache = self.block(
                x, attention_mask=attention_mask, position_ids=position_ids,
                freqs_cis=freqs_cis, past_key_value=past_key_value,
                use_cache=use_cache,
            )
            p = exit_prob.unsqueeze(2)  # [B, S, 1, 1]
            output = (1 - p) * computed + p * x
            return output, cache, exit_prob

        # 训练时: 全算, 软混合
        computed, cache = self.block(
            x, attention_mask=attention_mask, position_ids=position_ids,
            freqs_cis=freqs_cis, past_key_value=past_key_value,
            use_cache=use_cache,
        )
        p = exit_prob.unsqueeze(2)  # [B, S, 1, 1]
        output = (1 - p) * computed + p * x

        return output, cache, exit_prob


# ---------------------------------------------------------------------------
# 辅助 Loss: 防止 Gate 坍塌
# ---------------------------------------------------------------------------

def exit_balance_loss(exit_probs: List[torch.Tensor], target_skip_rate: float = 0.3) -> torch.Tensor:
    """鼓励模型跳过大约 target_skip_rate 比例的计算。

    如果不加这个 loss, gate 会坍塌到 "全跳" 或 "全不跳"。

    Args:
        exit_probs: list of [B, S, 1], 每层的跳过概率
        target_skip_rate: 目标跳过比例 (0.3 = 30% token 跳过)

    Returns:
        loss: 标量, 越小越好
    """
    total_loss = torch.tensor(0.0, device=exit_probs[0].device)
    for prob in exit_probs:
        mean_prob = prob.mean()
        # L2 惩罚: 偏离目标跳过率的程度
        total_loss = total_loss + (mean_prob - target_skip_rate) ** 2
    return total_loss / len(exit_probs)


def exit_entropy_loss(exit_probs: List[torch.Tensor]) -> torch.Tensor:
    """鼓励 gate 做出明确决策 (接近0或1), 而不是模糊的0.5。

    高熵 = 不确定 (p≈0.5) → 惩罚
    低熵 = 确定 (p≈0 or p≈1) → 鼓励

    Args:
        exit_probs: list of [B, S, 1]

    Returns:
        loss: 标量, 越小表示决策越明确
    """
    total_loss = torch.tensor(0.0, device=exit_probs[0].device)
    eps = 1e-7
    for prob in exit_probs:
        p = prob.clamp(eps, 1 - eps)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        total_loss = total_loss + entropy.mean()
    return total_loss / len(exit_probs)


# ---------------------------------------------------------------------------
# EarlyExitMoEModel: 完整模型
# ---------------------------------------------------------------------------

class EarlyExitMoEModel(DeepseekV4PreTrainedModel):
    """用 EarlyExitMoEBlock 替换标准 DeepseekV4Block 的模型。

    额外功能:
    1. 每层有 Exit Gate (跳过门)
    2. LayerContribution 跨层混合
    3. 辅助 loss 防止坍塌
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 用 EarlyExitMoEBlock 替换标准 Block
        self.layers = nn.ModuleList([
            EarlyExitMoEBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # 跨层特征混合
        self.layer_contrib = LayerContribution(config.num_hidden_layers, config.hidden_size)

        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        # HC head (同原模型)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta),
            persistent=False,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if module is self:
            nn.init.normal_(self.hc_head_fn, std=0.01)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

    def hc_head(self, x):
        """Contract hc_mult copies to 1."""
        dtype = x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.config.hc_eps
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hard_exit: bool = False,
    ):
        use_cache = False
        past_key_values = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seqlen = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(seqlen, device=inputs_embeds.device).unsqueeze(0)

        pos = position_ids.squeeze(0)
        freqs_cis = self.freqs_cis[:, pos].to(inputs_embeds.device)

        causal_mask = torch.full(
            (seqlen, seqlen), float("-inf"),
            device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        # Expand to hc_mult copies
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        exit_probs = []
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states, new_cache, exit_prob = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, causal_mask, position_ids, freqs_cis,
                    None, False, hard_exit,
                    use_reentrant=False,
                )
            else:
                hidden_states, new_cache, exit_prob = layer(
                    hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                    freqs_cis=freqs_cis, hard_exit=hard_exit,
                )

            exit_probs.append(exit_prob)
            # 收集每层 HC 收缩后的输出, 用于跨层混合
            layer_outputs.append(self.hc_head(hidden_states))

        # 两条路径混合:
        # 路径1: 标准 (最后一层 hc_head 输出)
        standard_output = layer_outputs[-1]
        # 路径2: 跨层加权混合
        mixed_output = self.layer_contrib(layer_outputs)
        # 最终输出 = 两条路径的均值 (也可以学习混合比例, 但先保持简单)
        hidden_states_out = 0.5 * (standard_output + mixed_output)

        hidden_states_out = self.norm(hidden_states_out)

        return hidden_states_out, exit_probs


class EarlyExitMoEForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    """带 Early Exit 的因果语言模型。"""

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = EarlyExitMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 辅助 loss 权重
        self.balance_loss_weight = 0.01
        self.entropy_loss_weight = 0.01
        self.target_skip_rate = 0.3

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hard_exit: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, exit_probs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            hard_exit=hard_exit,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 主 loss: 语言模型交叉熵
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 辅助 loss: 防止 gate 坍塌
            bal_loss = exit_balance_loss(exit_probs, self.target_skip_rate)
            ent_loss = exit_entropy_loss(exit_probs)

            loss = lm_loss + self.balance_loss_weight * bal_loss + self.entropy_loss_weight * ent_loss

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}


# ---------------------------------------------------------------------------
# 从标准模型迁移权重
# ---------------------------------------------------------------------------

def convert_from_standard_model(
    standard_model: DeepseekV4ForCausalLM,
    config: Optional[DeepseekV4Config] = None,
) -> EarlyExitMoEForCausalLM:
    """从标准 DeepseekV4ForCausalLM 迁移权重到 EarlyExitMoEForCausalLM。

    标准模型的所有权重都会被复制, 新增的 exit_gate 和 layer_contrib
    使用默认初始化。
    """
    if config is None:
        config = standard_model.config

    ee_model = EarlyExitMoEForCausalLM(config)

    # 复制 embed + lm_head
    ee_model.model.embed_tokens.load_state_dict(
        standard_model.model.embed_tokens.state_dict()
    )
    ee_model.lm_head.load_state_dict(
        standard_model.lm_head.state_dict()
    )

    # 复制每层的 block 权重
    for i in range(config.num_hidden_layers):
        ee_model.model.layers[i].block.load_state_dict(
            standard_model.model.layers[i].state_dict()
        )

    # 复制 HC head
    ee_model.model.hc_head_fn.data.copy_(standard_model.model.hc_head_fn.data)
    ee_model.model.hc_head_base.data.copy_(standard_model.model.hc_head_base.data)
    ee_model.model.hc_head_scale.data.copy_(standard_model.model.hc_head_scale.data)

    # 复制 norm
    ee_model.model.norm.load_state_dict(standard_model.model.norm.state_dict())

    # 复制 freqs_cis
    ee_model.model.freqs_cis = standard_model.model.freqs_cis

    print(f"[convert] 迁移完成. 新增参数:")
    new_params = 0
    for name, p in ee_model.named_parameters():
        if "exit_gate" in name or "layer_contrib" in name:
            new_params += p.numel()
            print(f"  {name}: {list(p.shape)}")
    print(f"  总新增: {new_params:,} ({new_params/1e6:.3f}M)")

    return ee_model


# ---------------------------------------------------------------------------
# 分析工具
# ---------------------------------------------------------------------------

def analyze_exit_patterns(model: EarlyExitMoEForCausalLM, input_ids: torch.Tensor):
    """分析每层的跳过模式。"""
    model.eval()
    with torch.no_grad():
        hidden_states, exit_probs = model.model(input_ids=input_ids)

    print("\n=== Exit Pattern Analysis ===")
    print(f"{'Layer':>6} | {'Mean Skip%':>10} | {'Min':>6} | {'Max':>6} | {'Std':>6}")
    print("-" * 50)
    total_skip = 0
    for i, prob in enumerate(exit_probs):
        mean_p = prob.mean().item()
        total_skip += mean_p
        print(f"{i:>6} | {mean_p*100:>9.1f}% | {prob.min().item():>6.3f} | {prob.max().item():>6.3f} | {prob.std().item():>6.3f}")

    avg_skip = total_skip / len(exit_probs)
    potential_speedup = 1 / (1 - avg_skip) if avg_skip < 1 else float('inf')
    print(f"\n平均跳过率: {avg_skip*100:.1f}%")
    print(f"理论加速比: {potential_speedup:.2f}x (推理时硬阈值)")

    # Layer contribution weights
    weights = F.softmax(model.model.layer_contrib.layer_weights, dim=0)
    print(f"\n=== Layer Contribution Weights ===")
    for i, w in enumerate(weights):
        bar = "█" * int(w.item() * 50)
        print(f"Layer {i:>2}: {w.item():.3f} {bar}")


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def _patch_sdpa_scale():
    """PyTorch < 2.1 的 scaled_dot_product_attention 不支持 scale 参数, 打 patch 兼容。"""
    orig_sdpa = F.scaled_dot_product_attention
    try:
        # 尝试调用带 scale 参数, 如果不报错则不需要 patch
        q = torch.zeros(1, 1, 1, 1)
        orig_sdpa(q, q, q, scale=1.0)
    except TypeError:
        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if scale is not None:
                query = query * (scale * math.sqrt(query.shape[-1]))
            return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        F.scaled_dot_product_attention = patched_sdpa
        print("[patch] SDPA scale 参数已打 patch (PyTorch < 2.1)")


def run_tests(config=None):
    """运行完整测试套件。"""
    _patch_sdpa_scale()

    if config is None:
        config = DeepseekV4Config(
            vocab_size=1024,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            moe_intermediate_size=128,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=1,
            hc_mult=4,
            max_position_embeddings=256,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print("=" * 60)
    print("EarlyExitMoEBlock Test Suite")
    print("=" * 60)

    # --- Test 1: ExitGate ---
    print("\n[Test 1] ExitGate 基础测试")
    gate = ExitGate(config.hidden_size * config.hc_mult).to(device, dtype)
    x = torch.randn(2, 8, config.hidden_size * config.hc_mult, device=device, dtype=dtype)
    prob = gate(x)
    assert prob.shape == (2, 8, 1), f"Shape mismatch: {prob.shape}"
    assert (prob >= 0).all() and (prob <= 1).all(), "Prob out of range"
    # 初始偏置为-2, sigmoid(-2)≈0.12, 应该倾向不跳过
    assert prob.mean() < 0.2, f"Initial skip rate too high: {prob.mean():.3f}"
    print(f"  ✓ Shape: {prob.shape}, mean: {prob.mean():.3f} (初始倾向不跳过)")

    # --- Test 2: LayerContribution ---
    print("\n[Test 2] LayerContribution 测试")
    contrib = LayerContribution(4, config.hidden_size).to(device, dtype)
    layers_out = [torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype) for _ in range(4)]
    mixed = contrib(layers_out)
    assert mixed.shape == (2, 8, config.hidden_size), f"Shape mismatch: {mixed.shape}"
    weights = F.softmax(contrib.layer_weights, dim=0)
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights don't sum to 1"
    print(f"  ✓ Shape: {mixed.shape}, weights sum: {weights.sum():.4f}")

    # --- Test 3: EarlyExitMoEBlock ---
    print("\n[Test 3] EarlyExitMoEBlock forward")
    block = EarlyExitMoEBlock(config, layer_idx=0).to(device, dtype)
    x_hc = torch.randn(2, 8, config.hc_mult, config.hidden_size, device=device, dtype=dtype)
    freqs = precompute_freqs_cis(config.qk_rope_head_dim, 256, config.rope_theta).to(device)
    freqs_slice = freqs[:, :8]
    causal = torch.full((8, 8), float("-inf"), device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    out, cache, exit_prob = block(x_hc, attention_mask=causal, freqs_cis=freqs_slice)
    assert out.shape == x_hc.shape, f"Shape mismatch: {out.shape} vs {x_hc.shape}"
    assert exit_prob.shape == (2, 8, 1), f"Exit prob shape: {exit_prob.shape}"
    print(f"  ✓ Output: {out.shape}, exit_prob: {exit_prob.shape}")

    # --- Test 4: 完整模型 forward ---
    print("\n[Test 4] EarlyExitMoEForCausalLM forward + loss")
    model = EarlyExitMoEForCausalLM(config).to(device, dtype)
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    assert output.loss is not None, "No loss"
    assert output.logits.shape == (2, 16, config.vocab_size), f"Logits shape: {output.logits.shape}"
    print(f"  ✓ Loss: {output.loss.item():.4f}, logits: {output.logits.shape}")

    # --- Test 5: 梯度检查 ---
    print("\n[Test 5] 梯度回传检查")
    output.loss.backward()
    exit_gate_grad = False
    contrib_grad = False
    block_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if "exit_gate" in name:
                exit_gate_grad = True
            if "layer_contrib" in name:
                contrib_grad = True
            if "block.attn" in name or "block.ffn" in name:
                block_grad = True
    assert exit_gate_grad, "Exit gate has no gradient!"
    assert contrib_grad, "Layer contrib has no gradient!"
    assert block_grad, "Block has no gradient!"
    print(f"  ✓ exit_gate 梯度: {exit_gate_grad}")
    print(f"  ✓ layer_contrib 梯度: {contrib_grad}")
    print(f"  ✓ block 核心参数梯度: {block_grad}")

    # --- Test 6: 辅助 loss ---
    print("\n[Test 6] 辅助 loss 测试")
    probs = [torch.full((2, 8, 1), 0.3, device=device)]
    bal = exit_balance_loss(probs, target_skip_rate=0.3)
    assert bal.item() < 1e-5, f"Balance loss should be ~0 at target: {bal.item()}"
    probs_extreme = [torch.full((2, 8, 1), 0.9, device=device)]
    bal_extreme = exit_balance_loss(probs_extreme, target_skip_rate=0.3)
    assert bal_extreme.item() > 0.1, f"Balance loss should be large at extreme: {bal_extreme.item()}"

    probs_sharp = [torch.full((2, 8, 1), 0.01, device=device)]
    ent_sharp = exit_entropy_loss(probs_sharp)
    probs_fuzzy = [torch.full((2, 8, 1), 0.5, device=device)]
    ent_fuzzy = exit_entropy_loss(probs_fuzzy)
    assert ent_sharp < ent_fuzzy, f"Sharp should have lower entropy: {ent_sharp:.4f} vs {ent_fuzzy:.4f}"
    print(f"  ✓ Balance loss @target: {bal.item():.6f}, @extreme: {bal_extreme.item():.4f}")
    print(f"  ✓ Entropy loss @sharp: {ent_sharp.item():.4f}, @fuzzy: {ent_fuzzy.item():.4f}")

    # --- Test 7: 参数统计 ---
    print("\n[Test 7] 参数统计")
    total = sum(p.numel() for p in model.parameters())
    exit_params = sum(p.numel() for n, p in model.named_parameters() if "exit_gate" in n)
    contrib_params = sum(p.numel() for n, p in model.named_parameters() if "layer_contrib" in n)
    new_params = exit_params + contrib_params
    print(f"  总参数: {total:,} ({total/1e6:.2f}M)")
    print(f"  Exit Gate 参数: {exit_params:,}")
    print(f"  Layer Contrib 参数: {contrib_params:,}")
    print(f"  新增参数: {new_params:,} (占比 {new_params/total*100:.2f}%)")

    # --- Test 8: Exit pattern 分析 ---
    print("\n[Test 8] 跳过模式分析")
    analyze_exit_patterns(model, input_ids)

    # --- Test 9: 权重迁移 ---
    print("\n[Test 9] 从标准模型迁移权重")
    standard = DeepseekV4ForCausalLM(config).to(device, dtype)
    converted = convert_from_standard_model(standard, config)
    converted = converted.to(device, dtype)
    # 验证迁移后 embedding 相同
    assert torch.allclose(
        converted.model.embed_tokens.weight,
        standard.model.embed_tokens.weight,
    ), "Embedding weights mismatch after conversion!"
    print("  ✓ 权重迁移验证通过")

    # 迁移后的模型也能正常 forward
    out2 = converted(input_ids=input_ids, labels=labels)
    assert out2.loss is not None
    print(f"  ✓ 迁移后 forward 正常, loss: {out2.loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def compare_models(config=None):
    """对比标准模型和 EarlyExit 模型。"""
    _patch_sdpa_scale()

    if config is None:
        config = DeepseekV4Config(
            vocab_size=1024,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            moe_intermediate_size=128,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=1,
            hc_mult=4,
            max_position_embeddings=256,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Model Comparison: Standard vs EarlyExit")
    print("=" * 60)

    std_model = DeepseekV4ForCausalLM(config).to(device)
    ee_model = EarlyExitMoEForCausalLM(config).to(device)

    std_params = sum(p.numel() for p in std_model.parameters())
    ee_params = sum(p.numel() for p in ee_model.parameters())

    print(f"\n标准模型参数: {std_params:,} ({std_params/1e6:.2f}M)")
    print(f"EarlyExit参数: {ee_params:,} ({ee_params/1e6:.2f}M)")
    print(f"新增: {ee_params - std_params:,} (+{(ee_params-std_params)/std_params*100:.2f}%)")

    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = input_ids.clone()

    # Forward comparison
    import time
    std_model.eval()
    ee_model.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(10):
            std_out = std_model(input_ids=input_ids, labels=labels)
        std_time = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            ee_out = ee_model(input_ids=input_ids, labels=labels)
        ee_time = (time.perf_counter() - t0) / 10

    print(f"\n标准模型 forward: {std_time*1000:.1f}ms")
    print(f"EarlyExit forward: {ee_time*1000:.1f}ms")
    print(f"开销: +{(ee_time/std_time - 1)*100:.1f}% (训练时, 软混合)")

    print(f"\n标准模型 loss: {std_out.loss.item():.4f}")
    print(f"EarlyExit loss: {ee_out.loss.item():.4f}")

    # Analyze exit patterns
    analyze_exit_patterns(ee_model, input_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EarlyExitMoEBlock 实验")
    parser.add_argument("--compare", action="store_true", help="对比标准模型和EarlyExit模型")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if "model" in cfg:
            config = DeepseekV4Config(**cfg["model"])

    if args.compare:
        compare_models(config)
    else:
        run_tests(config)
