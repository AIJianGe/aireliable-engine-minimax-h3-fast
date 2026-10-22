"""LightX2V Turbo 蒸馏 LoRA 加载器（S3 实装）。

LightX2V Turbo（ModelTC，`lightx2v/Minimax-h3-Turbo`）是**标准 diffusers 布局
bf16 步数蒸馏 LoRA**，其 safetensors 用 ComfyUI 前缀 `diffusion_model.*` 打包。
本模块把该 key 布局映射到**满血 FP16 标准 diffusers ``MinimaxH3Transformer3DModel``**，
以**手写旁路 adapter** 方式注入（推理速度 ~普通 LoRA 一个 matmul，几乎无感）。

为什么复用/沿用旁路而非 PEFT / 数学合并（与既有 `minimax_h3_video/lora_loader.py`
同源，v0.7.2 已定稿的教训）：
    - 数学合并进 backbone 权重会让"仅此引擎固定蒸馏"失去可逆性（unload 需重载模型）
    - 旁路子模块注册为被注入层（attn/ff 的 Linear）的**子模块**，`.to(device)`
      时自动跟随 host 层移动，无 device mismatch
    - fp16 旁路在 Turing（SM7.5）有 Tensor Core 路径；bf16 激活下 forward 包装
      内自动做 dtype 桥接

ComfyUI key → diffusers 模块映射（与参考引擎实测定稿一致）：
    diffusion_model.blocks.{i}.attn.qkv_proj    → transformer_blocks.{i}.attn.to_q/to_k/to_v
                                                  （fused 拆分：lora_B 按 out 维均分三段，
                                                   假定顺序 q/k/v，lora_A 三份共用）
    diffusion_model.blocks.{i}.attn.out_proj    → transformer_blocks.{i}.attn.to_out.0
    diffusion_model.blocks.{i}.mlp.fc1          → transformer_blocks.{i}.ff.net.0.proj
    diffusion_model.blocks.{i}.mlp.fc2          → transformer_blocks.{i}.ff.net.2
    diffusion_model.token_refiner.blocks.{i}.*  → token_refiner.refiner_blocks.{i}.*

标准 diffusers ``MinimaxH3Transformer3DModel`` 的模块布局即 ``transformer_blocks.*`` /
``token_refiner.refiner_blocks.*``，与上述映射**完全咬合**（鲁棒性：注入 0 模块判失败，
文件/结构不匹配绝不静默）。

注入失败**降级不阻断引擎启动**（只记 warning，跳过蒸馏仍可按满血步数跑）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# 旁路 adapter 的子模块属性名（挂到被注入层实例上，随 .to(device) 移动）
_LORA_A_ATTR = "_aireliable_lora_a"
_LORA_B_ATTR = "_aireliable_lora_b"

# 旁路计算精度：fp16（Turing SM7.5 Tensor Core 路径；forward 内做 dtype 桥接）
_BYPASS_DTYPE = torch.float16


@dataclass
class LoRAContext:
    """LoRA 旁路注入上下文（restore 时逐个恢复）。"""

    transformer: Any
    lora_name: str
    lora_path: str
    scale: float = 1.0
    wrapped: list = field(default_factory=list)
    injected: int = 0   # 成功注入的模块数（>0 才算注入成功）
    skipped: int = 0    # 跳过数（模块缺失 / 维度不匹配）


def _mod_name(module: nn.Module) -> str:
    """尽力获取模块名（用于日志）。"""
    return getattr(module, "_aireliable_lora_mod_path", type(module).__name__)


# ---------------------------------------------------------------------------
# key 解析与映射
# ---------------------------------------------------------------------------

def _parse_lora_keys(tensors: dict) -> dict[str, dict]:
    """把 safetensors 的 key 按 ComfyUI 模块路径分组。

    Returns:
        {comfyui_module_path: {"alpha": float, "A": Tensor, "B": Tensor}}
        其中 comfyui_module_path 形如 "blocks.0.attn.qkv_proj"（已剥掉
        diffusion_model. 前缀）。
    """
    groups: dict[str, dict] = {}
    for key, tensor in tensors.items():
        m = re.match(r"^diffusion_model\.(.+)\.(lora_A\.weight|lora_B\.weight|alpha)$", key)
        if not m:
            continue
        mod_path, leaf = m.group(1), m.group(2)
        entry = groups.setdefault(mod_path, {})
        if leaf == "alpha":
            entry["alpha"] = float(tensor.item()) if tensor.numel() == 1 else 1.0
        elif leaf == "lora_A.weight":
            entry["A"] = tensor
        else:
            entry["B"] = tensor
    return groups


def _map_targets(comfyui_mod: str) -> list[tuple[str, Optional[int]]]:
    """ComfyUI 模块路径 → [(标准 diffusers 模块路径, 段索引)]。

    fused qkv_proj 拆成 to_q/to_k/to_v 三份（B 按 out 维均分三段，假定
    q/k/v 顺序；A 三份共用）。段索引 0/1/2，None 表示整块映射。
    token_refiner 与主 blocks 同规则（blocks → refiner_blocks）。
    """
    m = re.match(r"^token_refiner\.blocks\.(\d+)\.(.+)$", comfyui_mod)
    if m:
        tb = f"token_refiner.refiner_blocks.{m.group(1)}"
        return _map_rest(tb, m.group(2))

    m = re.match(r"^blocks\.(\d+)\.(.+)$", comfyui_mod)
    if not m:
        return []
    tb = f"transformer_blocks.{m.group(1)}"
    return _map_rest(tb, m.group(2))


def _map_rest(tb: str, rest: str) -> list[tuple[str, Optional[int]]]:
    """按模块后缀（attn/mlp）映射目标（主 blocks 与 token_refiner 共用）。"""
    if rest == "attn.qkv_proj":
        # fused qkv → 三个独立模块，B 均分三段（q/k/v 顺序）
        return [
            (f"{tb}.attn.to_q", 0),
            (f"{tb}.attn.to_k", 1),
            (f"{tb}.attn.to_v", 2),
        ]
    if rest == "attn.out_proj":
        return [(f"{tb}.attn.to_out.0", None)]
    if rest == "mlp.fc1":
        return [(f"{tb}.ff.net.0.proj", None)]
    if rest == "mlp.fc2":
        return [(f"{tb}.ff.net.2", None)]
    return []


# ---------------------------------------------------------------------------
# 注入与恢复
# ---------------------------------------------------------------------------

def _inject_bypass(
    module: nn.Module,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    alpha: float,
    user_scale: float,
    log: logging.Logger,
) -> bool:
    """向单个 Linear 注入 LoRA 旁路。

    旁路公式：y = orig(x) + (alpha / r) * user_scale * lora_b(lora_a(x))
    """
    if hasattr(module, _LORA_A_ATTR):
        log.warning("模块已有 LoRA 旁路（重复注入防护），跳过: %s", _mod_name(module))
        return False

    r, in_features = a_weight.shape
    out_features = b_weight.shape[0]

    m_in = getattr(module, "in_features", None)
    m_out = getattr(module, "out_features", None)
    if m_in is not None and m_in != in_features:
        log.warning("LoRA 维度不匹配（in）: 模块=%s, 期望=%s, LoRA=%s",
                    _mod_name(module), m_in, in_features)
        return False
    if m_out is not None and m_out != out_features:
        log.warning("LoRA 维度不匹配（out）: 模块=%s, 期望=%s, LoRA=%s",
                    _mod_name(module), m_out, out_features)
        return False

    # 设备：nn.Linear 默认 CPU 创建，add_module 不会迁移——显式对齐宿主层设备，
    # 避免 L2/L3 下旁路在 CPU、主干在 GPU 导致 "mat2 is on cpu" 静默回退纯主干。
    try:
        host_device = next(module.parameters()).device
    except StopIteration:
        host_device = a_weight.device

    # dtype：先转 fp16 + 宿主设备，再 copy_（目标 fp16，源 bf16 安全降精度）——
    # 否则 copy_ 会把源转回目标 dtype（Linear 默认 fp32）→ 与 fp16 输入 "Half != float"。
    lora_a = nn.Linear(in_features, r, bias=False).to(device=host_device, dtype=_BYPASS_DTYPE)
    lora_b = nn.Linear(r, out_features, bias=False).to(device=host_device, dtype=_BYPASS_DTYPE)
    with torch.no_grad():
        lora_a.weight.copy_(a_weight)
        lora_b.weight.copy_(b_weight)

    module.add_module(_LORA_A_ATTR, lora_a)
    module.add_module(_LORA_B_ATTR, lora_b)

    combined_scale = (alpha / r) * user_scale

    orig_forward = module.forward

    def _lora_forward(x, *args, **kwargs):
        y = orig_forward(x, *args, **kwargs)
        try:
            # 惰性设备守卫：offload 策略动态移动设备时兜底
            if lora_a.weight.device != x.device:
                lora_a.to(x.device)
                lora_b.to(x.device)
            delta = lora_b(lora_a(x.to(_BYPASS_DTYPE)))
            y = y + (combined_scale * delta).to(y.dtype)
        except RuntimeError as e:
            log.warning("LoRA 旁路计算失败（回退纯主干）: %s: %s", _mod_name(module), e)
        return y

    module.forward = _lora_forward
    module._aireliable_lora_orig_forward = orig_forward  # restore 用
    return True


def _do_inject(transformer: nn.Module, lora_path: Path, scale: float, log: logging.Logger) -> Optional[LoRAContext]:
    """读 LoRA 权重并按 ComfyUI 分组映射注入到标准 diffusers transformer。"""
    try:
        from safetensors.torch import load_file
        tensors = load_file(str(lora_path))
    except Exception as e:
        log.warning("LoRA 文件读取失败: %s: %s", lora_path, e)
        return None

    groups = _parse_lora_keys(tensors)
    if not groups:
        log.warning("LoRA 文件无有效 key（非 ComfyUI diffusion_model 格式）: %s", lora_path)
        return None

    named = dict(transformer.named_modules())

    ctx = LoRAContext(
        transformer=transformer,
        lora_name=lora_path.stem,
        lora_path=str(lora_path),
        scale=scale,
    )

    for comfyui_mod, entry in sorted(groups.items()):
        a = entry.get("A")
        b = entry.get("B")
        if a is None or b is None:
            ctx.skipped += 1
            continue
        alpha = entry.get("alpha", 1.0)

        targets = _map_targets(comfyui_mod)
        if not targets:
            ctx.skipped += 1
            continue

        for local_path, seg_idx in targets:
            module = named.get(local_path)
            if module is None:
                ctx.skipped += 1
                log.warning("LoRA 目标模块不存在，跳过: %s", local_path)
                continue
            module._aireliable_lora_mod_path = local_path  # 日志用

            if seg_idx is not None:
                out_dim = b.shape[0] // 3
                seg_b = b[seg_idx * out_dim: (seg_idx + 1) * out_dim, :]
            else:
                seg_b = b

            if _inject_bypass(module, a, seg_b, alpha, scale, log):
                ctx.injected += 1
                ctx.wrapped.append(module)

    # 禁止静默零注入——注入 0 个模块视为失败
    if ctx.injected == 0:
        log.error(
            "Turbo LoRA 注入 0 个模块（全部跳过，skipped=%d）——文件与模型结构不匹配: %s",
            ctx.skipped, lora_path,
        )
        return None

    log.info(
        "Turbo LoRA 注入成功: %s, 注入模块数=%d, 跳过=%d, 路径=%s",
        lora_path.stem, ctx.injected, ctx.skipped, lora_path,
    )
    return ctx


def inject_turbo_lora(
    transformer: nn.Module,
    lora_path: str,
    scale: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Optional[LoRAContext]:
    """向满血 FP16 标准 diffusers transformer 注入 LightX2V Turbo 蒸馏 LoRA。

    Args:
        transformer: ``pipe.transformer``（或 ``pipe.transformer_ref``）——标准
            diffusers ``MinimaxH3Transformer3DModel``。
        lora_path: LoRA safetensors 路径（绝对路径）。
        scale: LoRA 有效缩放（默认 1.0）。

    Returns:
        注入成功的 LoRAContext；文件缺失 / 结构不匹配返回 None（不阻断引擎启动）。
    """
    log = logger or logging.getLogger(__name__)

    p = Path(lora_path)
    if not p.is_absolute():
        # 相对路径兜底：引擎目录 / models/video/minimax-h3-fast/
        candidates = [
            Path(lora_path),
            Path("models") / "video" / "minimax-h3-fast" / p.name,
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            log.warning(
                "LoRA 文件不存在: %s（已尝试 %s），跳过注入（请到模型管理界面下载 turbo LoRA）",
                lora_path, [str(c) for c in candidates],
            )
            return None

    log.info("准备注入 LightX2V Turbo LoRA: %s, scale=%.2f", p, scale)
    return _do_inject(transformer, p, scale, log)


def restore_lora(
    ctx: Optional[LoRAContext],
    logger: Optional[logging.Logger] = None,
) -> None:
    """卸载 LoRA（恢复原 forward、移除旁路子模块）。与 inject_turbo_lora 对称。"""
    log = logger or logging.getLogger(__name__)

    if ctx is None:
        return

    restored = 0
    for module in ctx.wrapped:
        try:
            orig = getattr(module, "_aireliable_lora_orig_forward", None)
            if orig is not None:
                module.forward = orig
                del module._aireliable_lora_orig_forward
            if hasattr(module, _LORA_A_ATTR):
                del module._aireliable_lora_a
            if hasattr(module, _LORA_B_ATTR):
                del module._aireliable_lora_b
            if hasattr(module, "_aireliable_lora_mod_path"):
                del module._aireliable_lora_mod_path
            restored += 1
        except Exception as e:
            log.warning("LoRA 旁路恢复失败（模块随 pipe 释放，可忽略）: %s", e)

    log.info(
        "Turbo LoRA 已卸载: %s, 恢复模块数=%d/%d",
        ctx.lora_name, restored, len(ctx.wrapped),
    )