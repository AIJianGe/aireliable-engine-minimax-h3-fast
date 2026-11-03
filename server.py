"""MiniMaxH3Fast 满血 FP16 视频生成引擎 HTTP 服务（FastAPI）。

8 个标准端点 / CORS / 日志端点 / 静态挂载由 SDK 骨架（``build_engine_app``）
提供；本文件保留 ``/info`` 与根路径（依赖 WebUI 是否挂载）。

启动（与全部既有引擎一致，走 -m 而非直接跑 server.py）：
    envs/video-minimax-h3/.venv/Scripts/python.exe \
        -m engines.minimax_h3_fast_a7c9d3e2.server --port 7014

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（video_generation + 1 档位）
    POST /load           满血 FP16 管线加载 + LightX2V Turbo 蒸馏 LoRA 注入
    POST /unload         释放模型 + 恢复 LoRA 旁路
    POST /infer          视频生成推理（蒸馏 8 步采样 → 编码 → completed 帧）
    GET  /status         状态查询
    POST /lora/load      LoRA 加载（蒸馏 LoRA 由 load 注入，运行时接口预留）
    POST /lora/unload    LoRA 卸载（同上）
    GET  /output/*       静态产物文件
    GET  /log_level      查询/调整日志级别（SDK 注册）
    GET  /ui/            原生 WebUI（webui/index.html）
    GET  /info           引擎信息（恒可用）
    GET  /               根路径 → 307 跳 /ui/（保留 method）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ===== sys.path 注入（保证任意 cwd 下均可启动）=====
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
_PROJECT_ROOT = _ENGINE_ROOT.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from aireliable_engine import build_engine_app, run_engine_server

from .config import EngineConfig, parse_args
from .engine import MiniMaxH3FastEngine

logger = logging.getLogger(__name__)

_WEBUI_DIR = _ENGINE_ROOT / "webui"
_HAS_WEBUI = (_WEBUI_DIR / "index.html").is_file()


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[MiniMaxH3FastEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> MiniMaxH3FastEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: MiniMaxH3FastEngine) -> None:
    """启动钩子：绑定全局实例 + 异步刷新 variant 扫描。"""
    global _engine

    _engine = engine
    # 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        engine.state_machine.state.value,
        len(engine._available_variants),
        engine.default_variant,
    )


async def _on_shutdown(engine: MiniMaxH3FastEngine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== MiniMaxH3Fast 视频生成引擎关闭 =====")
    try:
        if engine._pipe is not None:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由：``/info`` + 根路径 =====

router = APIRouter()


def _engine_info() -> dict:
    cfg = get_config()
    eng = get_engine()
    note = (
        "WebUI: GET /ui/；引擎信息: GET /info"
        if _HAS_WEBUI
        else "本引擎未提供 WebUI（webui/ 目录缺失）"
    )
    return {
        "engine_id": cfg.engine_id,
        "engine_name": cfg.engine_name,
        "capability": cfg.capability,
        "type": cfg.engine_type,
        "version": "0.2.0",
        "backend": "diffusers-fp16",
        "variants": [v.name for v in eng._available_variants],
        "endpoints": [
            "GET /health", "GET /capabilities", "POST /load", "POST /unload",
            "POST /infer", "GET /status", "POST /lora/load", "POST /lora/unload",
            "GET /output/*", "GET|POST /log_level",
        ]
        + (["GET /ui/", "GET /info"] if _HAS_WEBUI else ["GET /"]),
        "webui": ("/ui/" if _HAS_WEBUI else None),
        "note": note,
    }


@router.get("/info", include_in_schema=False)
async def _info() -> dict:
    """引擎信息（恒可用）。"""
    return _engine_info()


@router.get("/", include_in_schema=False)
async def _root():
    """根路径：有 WebUI 则 307 跳转（保留 method），否则返回引擎信息。"""
    if _HAS_WEBUI:
        # 307 保留 method（脚本探测根路径时不被改写；与 Qwen-Image 一致）
        return RedirectResponse(url="/ui/", status_code=307)
    return _engine_info()


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""
    global _config

    if config is None:
        config = parse_args()
    _config = config

    static_dirs = {"/output": config.output_root}
    if _HAS_WEBUI:
        static_dirs["/ui"] = _WEBUI_DIR

    return build_engine_app(
        MiniMaxH3FastEngine,
        config=config,
        title="MiniMaxH3Fast Engine",
        description=(
            "AIReliable 平台的 MiniMaxH3Fast 满血 FP16 视频生成引擎（LightX2V Turbo 蒸馏加速）"
        ),
        version="0.2.0",
        static_dirs=static_dirs,
        extra_routers=[router],
        redirect_root=None,  # 根路径由本文件的 _root 处理
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

def main() -> None:
    """命令行启动入口。"""
    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
