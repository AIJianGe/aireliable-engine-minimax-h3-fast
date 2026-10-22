"""MiniMaxH3Fast 满血 FP16 视频生成引擎推理核心（S3 实装）。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法。

S3 实装内容：
    1. load()：满血 FP16 ``ModularPipeline``（通用 modular 类，配合
       ``modular_model_index.json``）``load_components(names=[...], dtype=float16)``
       逐组件本地加载（diffusers，固定 FP16 + PyTorch SDPA，Turing 老卡无 BF16/FP8/FA）
       + LightX2V Turbo 蒸馏 LoRA 旁路注入（复用 accelerators/lora_loader）
    2. infer()：社区蒸馏加速采样（8 步），Euler 系调度 + 指数 sigma shift
       （flow_shift 12/6、audio_flow_shift 3、guidance≈1.0）生成视频帧，
       经 ffmpeg/imageio 编码 mp4
    3. 产物契约：completed 帧 result 为
       {"videos":[{"video_path","video_url","width","height","duration"}], ...}
       供网关 video_endpoints.py 的 _normalize_video_result 解析

⚠️ CUDA 实机推理留待用户；本文件保证结构正确 + 引擎可空载运行（模型文件
   缺失时 load() 抛 RuntimeError + 下载指引，不清真推理）。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.state_machine import RuntimeState

from .accelerators.lora_loader import inject_turbo_lora, restore_lora
from .config import EngineConfig
from .model_scanner import ModelScanner
from .schemas import CAPABILITY, TASK_TYPE, MiniMaxH3Inputs, MiniMaxH3VariantInfo

logger = logging.getLogger(__name__)


class MiniMaxH3FastEngine(CustomEngine):
    """MiniMaxH3Fast 满血 FP16 视频生成引擎（S3：FP16 管线 + LightX2V Turbo 蒸馏）。"""

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 启动时扫描（同步 fallback，异步扫描在 lifespan 中触发）
        self._available_variants: list[MiniMaxH3VariantInfo] = []
        self._refresh_scan()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        _manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant = self._read_default_variant_from_manifest(_manifest_path)

        # 任务类型（任务中心筛选语义）
        self.task_type = TASK_TYPE

        # ===== 模型实例（S3 实装）=====
        # load() 后 self._pipe = MinimaxH3Pipeline（满血 FP16），
        # self._lora_ctx 记录 LightX2V Turbo LoRA 注入上下文（unload 时对称恢复）。
        self._pipe = None
        self._lora_ctx = None

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新资源扫描结果（同步 fallback，用于 __init__ 阶段）。"""
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """方案 B：异步刷新扫描结果（优先平台 API），lifespan startup 阶段调用。"""
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用生成档位（1 档位）。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(MiniMaxH3VariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                backend=v.backend,
                distil_steps=v.distil_steps,
                video_format=v.video_format,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=[],  # 视频生成骨架不支持 LoRA
            supported_sizes=["1280x720", "1024x576", "720x1280"],
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema
            input_schema=MiniMaxH3Inputs.model_json_schema(),
            # 多能力引擎声明：本引擎仅 video_generation
            capabilities=[self.capability],
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load：满血 FP16 diffusers 管线加载 + LightX2V Turbo LoRA 注入。

        流程：
            1. variant 路径解析（不存在抛 FileNotFoundError）
            2. 模型文件**只校验不下载**：满血基座 marker 标志文件 model_index.json
               + FL2VA 蒸馏 LoRA 文件存在性；缺失抛 RuntimeError + 下载指引
            3. ModularPipeline.from_pretrained() + load_components(
               names=[transformer/...], dtype=fp16)（固定 FP16 + PyTorch SDPA，Turing 兼容）
            4. inject_turbo_lora() 以旁路 adapter 注入蒸馏 LoRA（失败降级不阻断）
        """
        # variant 缺省时回退（SDK LoadRequest 契约）：
        if not variant:
            variant = self.default_variant or next(
                (v.name for v in self._available_variants), None
            )
        if not variant:
            raise RuntimeError(
                "未指定 variant，且 manifest 无 default_variant / 无可用 variant，无法加载"
            )

        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在或资源目录缺失: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}；"
                f"期望目录: {self.config.model_dir}）"
            )

        vdef = self.scanner.get_variant_definition(variant)
        if not vdef:
            raise RuntimeError(f"variant 定义缺失（manifest 与 _KNOWN_VARIANTS 不一致?）: {variant}")

        device = self.config.resolve_device()
        model_dir = self.config.model_dir
        lora_path = self.config.lora_fl2v_path()

        # ===== 模型文件完整性校验（只校验不下载）=====
        flag_file = model_dir / "model_index.json"
        missing: list[str] = []
        if not flag_file.is_file():
            missing.append(f"{flag_file}（满血 FP16 基座标志文件）")
        if not lora_path.is_file():
            missing.append(f"{lora_path}（LightX2V Turbo 蒸馏 LoRA）")
        if missing:
            raise RuntimeError(
                f"模型文件不完整（缺失 {len(missing)} 项）：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」，选择镜像源后下载。\n"
                f"本地路径：{model_dir}\n"
                f"满血 FP16 基座：MiniMaxAI/MiniMax-H3\n"
                f"蒸馏 LoRA    ：lightx2v/Minimax-h3-Turbo（FL2VA 8 步）"
            )

        logger.info("模型完整性校验通过: variant=%s, device=%s, model_dir=%s", variant, device, model_dir)

        start_time = time.time()

        # ===== 满血 FP16 load（torch.load monkey-patch 覆盖全流程 + map_location=cpu）=====
        self._pipe = None
        self._lora_ctx = None
        try:
            import os
            import torch
            from diffusers import ModularPipeline

            # 离线模式：跳过 huggingface.co 文件更新检查（权重已本地化）
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            _orig_torch_load = torch.load

            def _patched_torch_load(*a, **kw):
                kw.setdefault("map_location", "cpu")
                kw.setdefault("weights_only", False)
                return _orig_torch_load(*a, **kw)

            torch.load = _patched_torch_load
            try:
                # 满血 FP16 用通用 ModularPipeline（目标 env diffusers 0.40.0.dev0
                # 仅提供 modular 类，无标准 MinimaxH3Pipeline）。
                # 不传 workflow 保留所有条件分支；随后 load_components 从本地逐组件加载。
                pipe = ModularPipeline.from_pretrained(str(model_dir))
                pipe.load_components(
                    names=[
                        "text_encoder", "tokenizer", "processor",
                        "vae", "audio_vae",
                        "scheduler", "audio_scheduler",
                        "transformer",  # FL2VA 分区（本引擎固定 FL2VA）
                    ],
                    dtype=torch.float16,  # 固定 FP16 + PyTorch SDPA（Turing 兼容）
                    pretrained_model_name_or_path=str(model_dir),
                )
                # 满血 FP16 基座 + 蒸馏 LoRA 旁路注入（失败降级，不阻断启动）
                self._lora_ctx = inject_turbo_lora(
                    pipe.transformer,
                    str(lora_path),
                    scale=self.config.lora_scale,
                    logger=logger,
                )
                if self._lora_ctx is None:
                    logger.warning(
                        "LightX2V Turbo 蒸馏 LoRA 注入失败（已降级为满血步数推理）——"
                        "请确认 lora 文件是 LightX2V 官方 diffusers 布局，与 MiniMaxH3Transformer3DModel key 匹配"
                    )
                # 满血调度到 GPU（Turing FP16 + SDPA）
                pipe.to(device)
                self._pipe = pipe
            finally:
                torch.load = _orig_torch_load

            if self._pipe is None:
                raise RuntimeError("ModularPipeline 加载失败（未得到有效 pipe 实例）")

        except RuntimeError:
            raise
        except Exception as e:
            logger.exception("满血 FP16 管线加载失败")
            raise RuntimeError(
                f"满血 FP16 diffusers 管线加载失败: {e}\n"
                f"本地路径：{model_dir}\n"
                f"请确认文件完整（到模型管理界面补全/重新下载）"
            ) from e

        load_time = time.time() - start_time
        state = "ready"
        logger.info(
            "满血 FP16 管线 + 蒸馏 LoRA 加载完成: variant=%s, device=%s, lora_injected=%s, 耗时=%.2fs",
            variant, device, self._lora_ctx is not None, load_time,
        )
        return LoadResponse(
            variant=variant,
            state=state,
            vram_used="未测量（满血 FP16 管线已驻留 GPU）",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：释放模型。

        即使未加载（如 ERROR 状态下），也要确保状态机回到 IDLE。
        """
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:
                logger.warning("unload()：reset 状态机失败: %s", e)

        # S3：先对称恢复 LoRA 旁路，再释放管线（对称注入/卸载）
        if self._lora_ctx is not None:
            restore_lora(self._lora_ctx, logger=logger)
            self._lora_ctx = None

        if self._pipe is not None:
            try:
                import torch
                self._pipe = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning("释放管线时清理显存失败: %s", e)
                self._pipe = None

        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:
                logger.warning("unload()：状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成（满血 FP16 管线 + 蒸馏 LoRA 已释放）")
        return UnloadResponse(state="idle", vram_freed="按管线占用清空（显存已释放）")

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：视频生成推理（S3：蒸馏 8 步采样 → 编码 → completed 帧）。

        流程：
            1. 解析 MiniMaxH3Inputs（prompt / width / height / num_frames / steps / seed）
            2. 蒸馏加速采样：Euler 系调度 + sigma shift（flow_shift 12/6、audio 3）、
               guidance≈1.0，``pipe(**pipe_kwargs)`` 产出帧序列 + 可选音频
            3. 编码 mp4（imageio / ffmpeg），音频可选 mux
            4. 推送 completed 帧 result.videos[].video_path / video_url / width / height / duration

        异常路径：推理/编码失败 → 推 error 帧后 return（不重新抛出，避免状态机 ERROR）。
        """
        task_id = request.task_id or f"mh3fast_{uuid.uuid4().hex[:16]}"
        logger.info("MMMM 收到 /infer 请求 task_id=%s", task_id)

        try:
            inputs = MiniMaxH3Inputs(**(request.inputs or {}))
        except Exception as e:
            logger.error("MiniMaxH3Inputs 解析失败: %s", e)
            stream.put_threadsafe(_make_error_frame(task_id, f"入参解析失败: {e}", "bad_input"))
            return

        if self._pipe is None:
            stream.put_threadsafe(_make_error_frame(
                task_id, "模型未加载，请先 POST /load 加载满血 FP16 管线", "model_not_loaded"))
            return

        prompt = (inputs.prompt or "").strip()
        if not prompt:
            stream.put_threadsafe(_make_error_frame(task_id, "prompt 不能为空", "bad_input"))
            return

        stream.put_threadsafe({
            "progress": 0.0, "stage": "generating", "task_id": task_id,
            "message": f"开始生成（{inputs.num_inference_steps} 步蒸馏）: {prompt[:60]}",
        })

        try:
            frames, audio_data, sampling_rate, meta = await asyncio.to_thread(
                self._run_diffusion, prompt=prompt,
                negative_prompt=inputs.negative_prompt,
                width=inputs.width, height=inputs.height,
                num_frames=inputs.num_frames,
                num_inference_steps=inputs.num_inference_steps,
                seed=inputs.seed,
            )
        except Exception as e:
            logger.exception("蒸馏采样失败")
            stream.put_threadsafe(_make_error_frame(task_id, f"推理失败: {e}", "inference_failed"))
            return

        # 编码 mp4 + 可选音频
        try:
            fps = 24
            output_dir = self.config.output_root / "video"
            video_result = await asyncio.to_thread(
                _save_video, frames, audio_data, output_dir, task_id, fps, sampling_rate)
        except Exception as e:
            logger.exception("保存视频失败")
            stream.put_threadsafe(_make_error_frame(task_id, f"保存视频失败: {e}", "save_failed"))
            return

        result = {
            "videos": [{
                "video_path": video_result["video_path"],
                "video_url": video_result["video_url"],
                "width": inputs.width,
                "height": inputs.height,
                "duration": video_result["duration"],
            }],
            "fps": fps,
            "seed": inputs.seed,
            "num_inference_steps": inputs.num_inference_steps,
            "variant": self._loaded_variant if hasattr(self, "_loaded_variant") else "MiniMaxH3Fast/fp16-full-8step",
            "device": self.config.resolve_device(),
            "is_placeholder": False,
            **meta,
        }
        logger.info("视频生成完成: task_id=%s, video_url=%s, duration=%.2fs",
                    task_id, result["videos"][0]["video_url"], result["videos"][0]["duration"])

        stream.put_threadsafe({
            "progress": 1.0, "stage": "completed", "task_id": task_id, "result": result,
        })

    def _run_diffusion(self, **kwargs) -> tuple:
        """同步执行的满血 FP16 蒸馏采样（供 asyncio.to_thread 包裹）。

        Returns:
            (frames: (T,H,W,C) uint8, audio_data: (C,S) float32 或 None,
             sampling_rate: int, meta: dict)
        """
        import numpy as np

        device = self.config.resolve_device()
        width = kwargs["width"]
        height = kwargs["height"]
        num_steps = kwargs["num_inference_steps"]
        seed = kwargs["seed"]

        # 帧数对齐到 17n+5（MiniMax-H3 时间压缩约束）
        num_frames = max(5, int(kwargs["num_frames"]))
        num_frames = ((num_frames + 12) // 17) * 17 + 5

        import torch
        if seed == -1:
            generator = torch.Generator("cpu")
        else:
            generator = torch.Generator("cpu").manual_seed(int(seed))

        # flow_shift：按分辨率选 video shift（12/6），audio 恒 3
        flow_shift = self.config.pick_flow_shift(width, height)
        audio_flow_shift = self.config.audio_flow_shift

        # sigma shift 生效：必须在 pipeline 内部 set_timesteps 前 set_shift
        try:
            for sched_name, sched, shift in (
                ("scheduler", getattr(self._pipe, "scheduler", None), flow_shift),
                ("audio_scheduler", getattr(self._pipe, "audio_scheduler", None), audio_flow_shift),
            ):
                if sched is None or shift is None:
                    continue
                if hasattr(sched, "set_shift"):
                    sched.set_shift(float(shift))
                    logger.info("[shift] %s set_shift=%.2f", sched_name, float(shift))
        except Exception as e:
            logger.warning("[shift] 设置 sigma shift 失败: %s", e)

        pipe_kwargs = {
            "prompt": kwargs["prompt"],
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_steps,
            "generator": generator,
            "output_type": "np",   # 返回 (1,T,H,W,C) float [0,1]
            # 注意：MiniMax-H3 是 guidance-distilled，guidance_scale/negative_prompt
            # 被官方 pipeline 忽略（与参考引擎一致的 t2v 传参，避免不支持的 kwarg）
        }

        logger.info("启动满血 FP16 蒸馏采样: %dx%d, frames=%d, steps=%d, shift=%.1f/%.1f, device=%s",
                    width, height, num_frames, num_steps, flow_shift, audio_flow_shift, device)

        state = self._pipe(**pipe_kwargs)

        # ----- 提取视频帧 -----
        videos = state.get("videos")
        if videos is None or len(videos) == 0:
            raise RuntimeError("推理未生成视频（videos 为空）")
        video_array = videos[0]
        if video_array.ndim == 5:
            video_array = video_array[0]
        frames = (video_array * 255.0).clip(0, 255).astype(np.uint8)  # (T,H,W,C)
        if frames.ndim != 4:
            raise RuntimeError(f"视频数组维度异常: ndim={frames.ndim}, shape={frames.shape}")

        # ----- 提取音频（可选）-----
        audio_data = None
        sampling_rate = int(state.get("sampling_rate", 32000))
        audio_state = state.get("audio")
        if audio_state is not None and len(audio_state) > 0:
            audio_array = audio_state[0]
            if hasattr(audio_array, "cpu"):
                audio_array = audio_array.cpu().float().numpy()
            else:
                audio_array = np.asarray(audio_array, dtype=np.float32)
            while audio_array.ndim > 2:
                audio_array = audio_array[0]   # (C,S)
            audio_data = audio_array

        meta = {
            "flow_shift": float(flow_shift),
            "audio_flow_shift": float(audio_flow_shift),
        }
        logger.info("蒸馏采样完成: frames=%s, audio=%s, total_frames=%d",
                    frames.shape, "yes" if audio_data is not None else "no", num_frames)
        return frames, audio_data, sampling_rate, meta

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=(self.default_variant if self._pipe is not None else None),
            vram_used="0MB" if self._pipe is None else "未测量（满血 FP16 管线驻留 GPU）",
            loaded_loras=[self.config.lora_fl2v_file] if self._lora_ctx is not None else [],
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": self.config.resolve_device(),
                "is_loaded": self._pipe is not None,
                "backend": "diffusers-fp16",
                "task_type": self.task_type,
                "capability": CAPABILITY,
                "model_dir": str(self.config.model_dir),
                "default_variant": self.default_variant,
                "lora_injected": self._lora_ctx is not None,
                "note": "满血 FP16 + LightX2V Turbo 蒸馏 LoRA（S3 实装）",
                "available_variants": [
                    {
                        "name": v.name,
                        "available": v.available,
                        "backend": v.backend,
                        "vram_required": v.vram_required,
                    }
                    for v in self._available_variants
                ],
            },
        )

    # ============ LoRA（固定蒸馏 LoRA 由 load() 注入，运行时接口保留占位）============

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：本引擎的蒸馏 LoRA 由 load() 固定注入，不支持运行时热插拔。"""
        raise RuntimeError(
            "MiniMaxH3Fast 的 LightX2V Turbo 蒸馏 LoRA 在 load() 时自动注入，"
            "不支持运行时 /lora/load（如需换档请先 unload 重载）"
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：同 _load_lora，蒸馏 LoRA 与模型同生命周期。"""
        raise RuntimeError(
            "MiniMaxH3Fast 的蒸馏 LoRA 与模型同生命周期，不支持单独卸载（通过 /unload 释放）"
        )


def _make_error_frame(task_id: str, message: str, etype: str = "inference_failed") -> dict:
    """构造 SSE 错误帧（stage=error）。"""
    return {
        "progress": 0.0,
        "stage": "error",
        "task_id": task_id,
        "error": {"type": etype, "message": message},
    }


def _save_video(
    frames,
    audio_data,
    output_dir: Path,
    task_id: str,
    fps: int = 24,
    sample_rate: int = 32000,
) -> dict:
    """把帧序列编码为 mp4（可选音频 mux），返回产物契约。

    Returns:
        {"video_path": 绝对路径, "video_url": "/output/video/{task_id}.mp4",
         "duration": 秒}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = (output_dir / f"{task_id}.mp4").resolve()

    video_only_path = (output_dir / f"{task_id}.video_only.mp4").resolve()

    writer = None
    try:
        import imageio
        writer = imageio.get_writer(str(video_only_path), fps=fps, macro_block_size=None)
        for f in frames:
            writer.append_data(f)
    finally:
        if writer is not None:
            writer.close()

    # ---- 音频 mux（可选）：优先 imageio_ffmpeg 完整版，失败降级为仅视频 ----
    final_path.write_bytes(video_only_path.read_bytes())
    if audio_data is not None:
        try:
            import subprocess
            local_tmp = output_dir / f"{task_id}.mux.mp4"
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            # 用 wav 中间件承载音频，再 ffmpeg mux 进 mp4
            wav_path = output_dir / f"{task_id}.audio.wav"
            import scipy.io.wavfile as wavfile
            wavfile.write(str(wav_path), sample_rate, audio_data.T.astype("float32"))
            cmd = [
                ffmpeg, "-y",
                "-i", str(video_only_path),
                "-i", str(wav_path),
                "-c:v", "copy",
                "-c:a", "aac", "-shortest",
                str(local_tmp),
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode == 0 and local_tmp.is_file():
                final_path.write_bytes(local_tmp.read_bytes())
                local_tmp.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("音频 mux 失败，降级为无音轨视频: %s", e)

    duration = frames.shape[0] / fps if fps > 0 else 0.0
    return {
        "video_path": str(final_path),
        "video_url": f"/output/video/{task_id}.mp4",
        "duration": duration,
    }