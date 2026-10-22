"""MiniMaxH3Fast 满血 FP16 社区蒸馏加速视频引擎配置。

从环境变量 + 命令行参数加载引擎运行所需的全部配置。

设计参考（同 videoUpEnlarge/VSR 引擎的双源真相消除策略）：
    - MODEL_SUBDIR 默认读环境变量 ``AIRELIABLE_MODEL_LAYOUT_BASE``
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback

命名与端口：
    - engine_id  : minimax_h3_fast_a7c9d3e2（目录 / manifest / config 三同名契约）
    - engine_name: MiniMaxH3Fast（用户指定）
    - capability : video_generation（视频生成能力）
    - 端口       : 7014（videoUpEnlarge 已在 7013，VSR 在 7012）
    - 模型布局   : video/minimax-h3-fast/

⚠️ 骨架阶段说明：
    本引擎定位为"满血 FP16 + 社区蒸馏加速"的 MiniMax-H3 生态视频生成引擎。
    当前为**骨架空壳**：模型加载与真实推理均为占位（清晰 TODO），
    只有 8 个标准端点 + WebUI 路由可跑通。评测变体（fp16-full-8step）未定档，
    模型仓库真实 repo_id 未定，源 URL 为占位。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ===== 常量 =====

ENGINE_ID = "minimax_h3_fast_a7c9d3e2"
ENGINE_NAME = "MiniMaxH3Fast"
CAPABILITY = "video_generation"
ENGINE_TYPE = "video"

# 默认端口（与 config/engines/minimax_h3_fast_a7c9d3e2.yaml 一致）
DEFAULT_PORT = 7014

# 满血 FP16 引擎依赖 torch + diffusers（env_config.requires_torch: true）
REQUIRES_TORCH = True

# 模型目录布局（相对 MODELS_ROOT）
# 优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE（平台注入，双源真相消除）
_DEFAULT_MODEL_SUBDIR = "video/minimax-h3-fast/"

# ===== 模型仓库（S3 定档，与 manifest.yaml source.url / model_scanner 三处一致）=====
# 满血 FP16 基座：MiniMax 官方 diffusers 仓库（compound pipeline，含
#   text_encoder / transformer / token_refiner / vae / audio_vae 等子目录）
MH3FAST_MARKER_ID = "mh3fast-fp16-base"
MH3FAST_REPO_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3"

# 蒸馏加速 LoRA：LightX2V Turbo（ModelTC / lightx2v/Minimax-h3-Turbo，Apache-2.0）
# ✔ 2026-09-25 复用决策：本机已下载 ComfyUI 布局 8 步 v1.0（reference 引擎
#   minimax_h3_video_a1b2c3 的 loras/ 下），lora_loader 正是做 ComfyUI→diffusers
#   key 映射，故改用现成 ComfyUI 文件，免去 1.8GB diffusers 版下载。
TURBO_LORA_REPO_URL = "https://huggingface.co/lightx2v/Minimax-h3-Turbo"
TURBO_LORA_FL2V_FILE = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TURBO_LORA_REF2V_FILE = "minimax_h3_ref2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
# LoRA 复用来源目录（reference 引擎已下载的 ComfyUI LoRA 所在处）
TURBO_LORA_REUSE_DIR = "video/minimax-h3/loras"

# ===== 采样契约（S3 定稿，写入推理前必对齐）=====
# 官方 turbo 8 步蒸馏：推理 NFE ≈ 8（可 +1 音频退火）；Euler 系调度 + 指数 sigma shift。
# flow_shift（video）：12 通用 / 6（768p 训练档）；audio_flow_shift：3。
# 蒸馏后 guidance≈1.0（CFG 大档无意义，忽略 negative_prompt）。
TURBO_SAMPLING_STEPS = 8
TURBO_FLOW_SHIFT = 12.0           # video 通用档
TURBO_FLOW_SHIFT_768P = 6.0       # video 768p 档
TURBO_AUDIO_FLOW_SHIFT = 3.0      # audio 恒 3
TURBO_GUIDANCE_SCALE = 1.0


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量，双源真相消除）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()
# 公有别名（供 model_scanner 引用，避免依赖私有下划线名）
DEFAULT_MODEL_SUBDIR = MODEL_SUBDIR


@dataclass
class EngineConfig:
    """MiniMaxH3Fast 引擎运行配置。"""

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # ===== 路径 =====
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    output_root: Path = Path("output")

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== 设备 =====
    # auto: 有 CUDA 则 cuda:0，否则 cpu（视频生成走 torch+diffusers，必须有具体设备）
    device: str = "auto"

    # ===== 生成参数默认值（S3 实装：对齐官方 turbo 8 步采样契约）=====
    width: int = 1280
    height: int = 720
    num_frames: int = 121
    num_inference_steps: int = TURBO_SAMPLING_STEPS  # 8 步蒸馏（4 步留待验证）
    seed: int = -1                                   # -1 = 随机
    flow_shift: float = TURBO_FLOW_SHIFT             # video：12 通用 / 6（768p）
    audio_flow_shift: float = TURBO_AUDIO_FLOW_SHIFT  # audio：3
    guidance_scale: float = TURBO_GUIDANCE_SCALE     # 蒸馏后 ≈1.0

    # ===== LoRA（LightX2V Turbo，diffusers bf16 文件落在模型目录）=====
    lora_fl2v_file: str = TURBO_LORA_FL2V_FILE
    lora_ref2v_file: str = TURBO_LORA_REF2V_FILE
    lora_scale: float = 1.0                          # LoRA 有效缩放

    # ===== 日志 =====
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """推导子路径、统一绝对化并创建运行时目录。

        ⚠️ 必须绝对化（参考 videoUpEnlarge v1.1 教训）：
            引擎可能以非项目根 cwd 被拉起，任何相对路径都要在这里 `.resolve()`，
            避免后续推理/产物写入落到错误目录。diffusers 推理会把产物写到
            output_root，因此需要预先创建。
        """
        # 统一绝对化（对已是绝对路径的输入，resolve() 是幂等的）
        self.project_root = Path(self.project_root).resolve()
        self.models_root = Path(self.models_root).resolve()
        self.output_root = Path(self.output_root).resolve()

        # 模型目录（video/minimax-h3-fast/）
        self.model_dir = (self.models_root / MODEL_SUBDIR).resolve()

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = (self.project_root / "logs" / "engines").resolve()

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ===== 设备解析 =====

    def resolve_device(self) -> str:
        """解析推理设备（在传给后端之前完成，不留 "auto" 歧义）。

        Returns:
            "cuda:0" / "cpu"

        说明：满血 FP16 diffusers pipeline 调度到 GPU 才算真正"满血"；
        CPU 仅作兜底（评测/无卡场景）。
        """
        device = (self.device or "auto").strip()
        if device == "cpu":
            return "cpu"
        if device.startswith(("cuda", "gpu")):
            return device
        # auto
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    # ===== 蒸馏采样契约辅助 =====

    def pick_flow_shift(self, width: int, height: int) -> float:
        """按输出分辨率选 video flow_shift：768p 走 6，其余走 12（官方 turbo 语义）。

        Args:
            width / height: 请求分辨率（像素）。

        Returns:
            video flow_shift（float）。
        """
        # 768 线（720P/768P 级）用 shift=6，更高如 1080P 用 shift=12（官方 turbo）。
        return TURBO_FLOW_SHIFT_768P if max(width, height) <= 768 else TURBO_FLOW_SHIFT

    def _lora_candidates(self, filename: str) -> list[Path]:
        """LoRA 候选路径（优先复用 reference 已下载目录，回退到本引擎 loras/）。"""
        return [
            (self.models_root / TURBO_LORA_REUSE_DIR / filename).resolve(),
            (self.model_dir / "loras" / filename).resolve(),
        ]

    def lora_fl2v_path(self) -> Path:
        """FL2VA 蒸馏 LoRA 绝对路径（优先现成 ComfyUI 文件）。"""
        for p in self._lora_candidates(self.lora_fl2v_file):
            if p.is_file():
                return p
        return self._lora_candidates(self.lora_fl2v_file)[0]

    def lora_ref2v_path(self) -> Path:
        """Ref2VA 蒸馏 LoRA 绝对路径（优先现成 ComfyUI 文件）。"""
        for p in self._lora_candidates(self.lora_ref2v_file):
            if p.is_file():
                return p
        return self._lora_candidates(self.lora_ref2v_file)[0]

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "log_level": self.log_level,
        }


def parse_args(argv: Optional[list[str]] = None) -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。"""
    parser = argparse.ArgumentParser(description="MiniMaxH3Fast 视频生成引擎（满血 FP16 + 社区蒸馏加速）")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument("--output-root", type=str, default=None,
                        help="产物根目录（默认 output/）")

    args = parser.parse_args(argv)

    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))
    output_root = Path(args.output_root or os.environ.get("AIRELIABLE_OUTPUT_ROOT", "output"))

    return EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
        output_root=output_root,
    )