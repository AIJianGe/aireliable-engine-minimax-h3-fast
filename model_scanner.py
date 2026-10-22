"""MiniMaxH3Fast 模型扫描器（1 档位 variant 版，骨架占位）。

扫描 ``models/video/minimax-h3-fast/`` 目录，识别可用的生成档位。

1 variant 设计（评测未定，当前占位）：
    - MiniMaxH3Fast/fp16-full-8step   ：满血 FP16 + 社区蒸馏 8 步高低清

v0.2.11+ 方案 B 约定：
    优先调用平台 API（``engine.get_variants()``）获取 variant 可用性，
    平台不可达时 fallback 到本地扫描。

本地 fallback 检查策略（project-rules §第二部分 5）：
    - 只检查标志文件存在性，**不校验大小**
    - 完整性校验由平台 API（key_files 机制）或 ``load()`` 兜底

⚠️ 骨架阶段：
    model_files 声明为 marker 且 ``key_files=[]``（真实 FP16 仓库未定，S3 补全），
    因此本地没有任何"必须存在"的具体文件。S3 阶段补全 key_files 后，
    此处 _check_variant_ready 会改为按真实标志文件判断，trainer 只替代 marker
    存在性。当前扫描结果恒置 available=False 的占位语义由平台 API/本地目录
    决定——目录存在即视为可见（可触发下载/更新入口），满足"variant 在 UI 可见"。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import DEFAULT_MODEL_SUBDIR
from .schemas import MiniMaxH3VariantInfo

logger = logging.getLogger(__name__)


# ===== 已知变体定义 =====
# ⚠️ "name" 必须与 manifest.yaml 的 models.variants[].name 逐字一致
#    （variant 扫描匹配契约，双处不一致 → UI 扫不到 → 无法触发下载）
_KNOWN_VARIANTS: list[dict] = [
    # ============================================================
    # 1. 满血 FP16 + LightX2V Turbo 蒸馏 8 步（S3 实装）
    # ============================================================
    {
        "name": "MiniMaxH3Fast/fp16-full-8step",
        "alias": "FP16 满血 8步（LightX2V Turbo 蒸馏）",
        "base_model": "MiniMax-H3 (FP16 full)",
        "precision": "fp16",
        "loader": "diffusers",
        "vram_required": "20GB",
        "backend": "diffusers-fp16",
        "distil_steps": "8 步（LightX2V Turbo）",
        "video_format": "mp4",
        "description": "满血 FP16 diffusers 管线 + LightX2V Turbo 8 步蒸馏 LoRA 的高清视频生成档位",
    },
]


class ModelScanner:
    """MiniMaxH3Fast 模型扫描器（1 档位 variant 版）。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/）
            model_dir: 本引擎模型目录（video/minimax-h3-fast/）
            engine: 可选的 BaseEngine 实例（方案 B 注入）。
                传入时优先调用平台 API 检测 variant；为 None 时走本地扫描。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)

    # ===== 扫描入口 =====

    async def scan_variants_async(self) -> list[MiniMaxH3VariantInfo]:
        """方案 B：异步扫描，优先调用平台 API。"""
        if self.engine is not None:
            try:
                platform_variants = await self.engine.get_variants()
            except Exception as e:
                self._logger.warning("平台 API 获取 variants 失败，回退本地扫描: %s", e)
                platform_variants = None
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        return self._scan_locally()

    def scan_variants(self) -> list[MiniMaxH3VariantInfo]:
        """同步扫描接口（fallback 本地扫描）。"""
        return self._scan_locally()

    # ===== 平台 API 合并 =====

    def _merge_with_known(self, platform_variants: list[dict]) -> list[MiniMaxH3VariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。"""
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[MiniMaxH3VariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            if not kv:
                # 平台声明了引擎不认识（或已下线）的 variant：跳过，避免误报可用
                self._logger.debug("忽略未知 variant（不在 _KNOWN_VARIANTS 中）: %s", v_name)
                continue

            available = bool(pv.get("available", False))
            result.append(MiniMaxH3VariantInfo(
                name=v_name,
                base_model=pv.get("base_model", kv.get("base_model", "")),
                precision=pv.get("precision", kv.get("precision", "fp16")),
                loader=pv.get("loader", kv.get("loader", "diffusers")),
                vram_required=pv.get("vram_required", kv.get("vram_required", "20GB")),
                available=available,
                backend=kv.get("backend", "diffusers-fp16"),
                distil_steps=kv.get("distil_steps", ""),
                video_format=kv.get("video_format", "mp4"),
            ))
            if available:
                self._logger.info("发现模型变体: %s (%s)", v_name, kv.get("alias", ""))

        if not any(v.available for v in result):
            self._warn_nothing_found()
        return result

    # ===== 本地扫描 =====

    def _scan_locally(self) -> list[MiniMaxH3VariantInfo]:
        """fallback 本地扫描（骨架：目录存在即视为就位，不校验大小）。

        与平台 API 行为一致（project-rules §第三部分 3）：本地 fallback 也必须
        **列出全部已声明 variant**（available 反映文件是否就位），而非跳过未就位
        的 variant。否则会出现"平台返回 N 档、本地返回 0 档"的不一致，UI 扫不到
        档位 → 无法触发下载入口。
        """
        capabilities: list[MiniMaxH3VariantInfo] = []

        for variant_def in _KNOWN_VARIANTS:
            ready = self._check_variant_ready(variant_def["name"])
            if not ready:
                self._logger.debug(
                    "变体 %s 未就位（模型目录 %s 不存在，仍列出以触发下载入口）: %s",
                    variant_def["name"],
                    self.model_dir,
                    "骨架 marker 无 key_files，S3 实装后按真实标志文件判断",
                )
            cap = MiniMaxH3VariantInfo(
                name=variant_def["name"],
                base_model=variant_def["base_model"],
                precision=variant_def["precision"],
                loader=variant_def["loader"],
                vram_required=variant_def["vram_required"],
                available=ready,
                backend=variant_def["backend"],
                distil_steps=variant_def["distil_steps"],
                video_format=variant_def["video_format"],
            )
            if ready:
                self._logger.info(
                    "发现模型变体: %s (%s, backend=%s)",
                    cap.name, variant_def["alias"], cap.backend,
                )
            capabilities.append(cap)

        if not any(v.available for v in capabilities):
            self._warn_nothing_found()

        return capabilities

    def _check_variant_ready(self, variant_name: str) -> bool:
        """检查 variant 资源是否就位（本地 fallback，S3）。

        策略（project-rules §第二部分 5）：只检查**标志文件**存在性，不校验大小——
        dir 存在 + 满血基座 marker 的 model_index.json 就位即可视为"可见"（可触发
        下载/更新入口）。大权重完整性交给平台 API（key_files）或 load() 兜底。
        """
        if variant_name not in {v["name"] for v in _KNOWN_VARIANTS}:
            return False
        # 满血基座 marker 标志文件（manifest model_files[0].filename）
        return (self.model_dir / "model_index.json").exists()

    # ===== 路径与定义解析 =====

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型资源目录。

        Returns:
            模型根目录（如 models/video/minimax-h3-fast/），已知 variant 一律返回
            （目录是否存在不阻断解析，实时文件就位由平台 key_files 或 bulk load
            校验；骨架阶段 marker 无 key_files，目录不存在也允许空载 load）。
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                return self.model_dir
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取 variant 的完整定义（供 load() 定位模型与校验文件）。"""
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None

    # ===== 诊断 =====

    def _warn_nothing_found(self) -> None:
        """扫描无结果时的统一告警（含具体路径 + 仓库 ID + 下载指引）。"""
        self._logger.warning(
            "未在 %s 中找到可用模型（缺少 model_index.json），请到「模型管理」下载：\n"
            "  满血 FP16 基座：MiniMaxAI/MiniMax-H3\n"
            "  蒸馏 LoRA    ：lightx2v/Minimax-h3-Turbo（FL2VA 8 步）",
            self.model_dir,
        )