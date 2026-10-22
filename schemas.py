"""MiniMaxH3Fast 满血 FP16 视频生成引擎 Pydantic 数据模型。

扩展 SDK 的 VariantInfo / InferRequest.inputs，定义视频生成专属字段，
并为 REST 契约声明 infer 入参。

variant = 生成档位（S3 实装 1 个）：
    - MiniMaxH3Fast/fp16-full-8step   ：满血 FP16 + LightX2V Turbo 蒸馏 8 步

任务类型（task_type）：video_generation（视频生成，任务中心分类语义，
    与网关 video_endpoints.py 的 task_type 逐字一致）。

infer 入参（放入 InferRequest.inputs，由网关 / WebUI 透传）：
    - prompt                ：正向提示词
    - negative_prompt       ：负向提示词（蒸馏后 ~1.0 guidance，一般忽略）
    - width / height        ：输出分辨率（默认 1280x720）
    - num_frames            ：帧数
    - num_inference_steps   ：推理步数（蒸馏 8 步）
    - seed                  ：随机种子（-1 = 随机）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo

# ===== 能力声明 =====
CAPABILITY = "video_generation"
ENGINE_TYPE = "video"
# 与 manifest.models.model_layout.base 保持一致
DEFAULT_MODEL_SUBDIR = "video/minimax-h3-fast/"

# ===== 任务类型（与网关 video_endpoints.py 的 task_type 逐字一致）=====
TASK_TYPE = "video_generation"


class MiniMaxH3VariantInfo(VariantInfo):
    """MiniMaxH3Fast 变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充视频生成特有信息：
        - backend        ：后端标识（骨架固定 "diffusers-fp16"，实际管线 S3 定）
        - distil_steps   ：社区蒸馏加速等效步数描述
        - video_format   ：输出视频封装
    """

    model_config = ConfigDict(extra="allow")

    backend: str = Field(
        "diffusers-fp16",
        description="后端标识（满血 FP16 diffusers 管线 + LightX2V Turbo 蒸馏 LoRA）",
    )
    distil_steps: str = Field(
        "8 步（LightX2V Turbo）",
        description="社区蒸馏加速等效步数描述",
    )
    video_format: str = Field(
        "mp4",
        description="输出视频封装格式",
    )


class MiniMaxH3Inputs(BaseModel):
    """MiniMaxH3Fast 推理输入参数（放入 InferRequest.inputs）。

    所有字段均有默认值，网关 / WebUI 按需覆盖。
    ``extra="allow"`` 保证网关透传的未知字段不会导致解析失败。
    """

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(
        "",
        description="正向提示词（视频内容描述）",
    )
    negative_prompt: str = Field(
        "",
        description="负向提示词（不希望出现的内容）",
    )
    width: int = Field(
        1280, ge=64, le=4096, description="输出宽度（像素）"
    )
    height: int = Field(
        720, ge=64, le=4096, description="输出高度（像素）"
    )
    num_frames: int = Field(
        121, ge=1, le=2000, description="输出帧数"
    )
    num_inference_steps: int = Field(
        8, ge=1, le=200,
        description="推理步数（LightX2V Turbo 蒸馏 8 步）",
    )
    seed: int = Field(
        -1, description="随机种子（-1 = 随机）"
    )


class InferRequest(BaseModel):
    """MiniMaxH3Fast 推理请求契约（骨架 REST 声明）。

    与平台网关→引擎的 SDK ``aireliable_engine.InferRequest`` 形状同源：
    variant / task_id 放顶层，生成参数放 inputs。
    骨架阶段提供此模型用于：
        1. MCP/文档化 /capabilities 的 input_schema 导出
        2. 明确 S3 实装后 /infer 的参数契约
    """

    variant: Optional[str] = Field(
        None, description="生成档位（None = 用 default_variant）"
    )
    task_id: Optional[str] = Field(None, description="任务 ID（可选）")
    inputs: MiniMaxH3Inputs = Field(
        default_factory=MiniMaxH3Inputs,
        description="生成参数（prompt / negative_prompt / width / height / num_frames / steps / seed）",
    )


class InferResponse(BaseModel):
    """MiniMaxH3Fast 推理响应（保留为 REST 声明兼容；实际推理走 SSE）。"""

    task_id: Optional[str] = None
    status: str = "ok"
    code: int = 0
    message: str = Field(
        "",
        description="提示信息",
    )
    extra: dict[str, Any] = Field(default_factory=dict)