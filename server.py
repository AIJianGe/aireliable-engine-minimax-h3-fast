"""MiniMaxH3Fast 满血 FP16 视频生成引擎 HTTP 服务（FastAPI）。

实现 8 个标准 HTTP 端点 + 挂载原生 WebUI。

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# ===== sys.path 注入（保证任意 cwd 下均可启动）=====
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
_PROJECT_ROOT = _ENGINE_ROOT.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import MiniMaxH3FastEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
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


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动初始化引擎，关闭清理资源。"""
    global _engine, _config

    if _config is None:
        _config = parse_args()

    # 配置日志（v0.2.15+ 统一到 logs/engines/，依赖平台注入 AIRELIABLE_LOG_DIR）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== MiniMaxH3Fast 满血 FP16 视频生成引擎启动（S3 实装）=====")
    logger.info("配置: %s", _config.to_dict())

    _engine = MiniMaxH3FastEngine(_config)
    # 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
        _engine.default_variant,
    )

    yield

    # 关闭清理
    logger.info("===== MiniMaxH3Fast 视频生成引擎关闭 =====")
    if _engine is not None and _engine._pipe is not None:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。"""
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="MiniMaxH3Fast Engine",
        description=(
            "AIReliable 平台的 MiniMaxH3Fast 满血 FP16 视频生成引擎（LightX2V Turbo 蒸馏加速）"
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS（允许 iframe 嵌入 + 跨源调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供产物下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    # ===== 挂载原生 WebUI（webui/index.html 存在时）=====
    webui_mounted = _mount_webui(app)

    # ===== 根路径（依赖是否挂了 WebUI）=====
    _register_root(app, webui_mounted)

    return app


def _mount_webui(app: FastAPI) -> bool:
    """挂载引擎原生 WebUI（``webui/index.html`` 存在时挂到 ``/ui``）。"""
    webui_dir = Path(__file__).resolve().parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").is_file():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
        logger.info("WebUI 挂载: /ui -> %s", webui_dir)
        return True
    logger.info("WebUI 未挂载（webui/index.html 不存在），保持 headless")
    return False


def _register_root(app: FastAPI, has_webui: bool) -> None:
    """注册根路径 + ``/info``。

    平台「引擎操作」页 iframe `http://127.0.0.1:<端口>/`，故根路径跳到 /ui/。
    ``/info`` 恒返回 JSON（发现端点/脚本/排障）。
    """

    def _engine_info() -> dict:
        cfg = get_config()
        eng = get_engine()
        note = (
            "WebUI: GET /ui/；引擎信息: GET /info"
            if has_webui
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
            + (["GET /ui/", "GET /info"] if has_webui else ["GET /"]),
            "webui": ("/ui/" if has_webui else None),
            "note": note,
        }

    @app.get("/info", include_in_schema=False)
    async def _info() -> dict:
        """引擎信息（恒可用）。"""
        return _engine_info()

    @app.get("/", include_in_schema=False)
    async def _root():
        """根路径：有 WebUI 则 307 跳转（保留 method），否则返回引擎信息。"""
        if has_webui:
            # 307 保留 method（脚本探测根路径时不被改写；与 Qwen-Image 一致）
            return RedirectResponse(url="/ui/", status_code=307)
        return _engine_info()


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（video_generation + 档位）。"""
        resp = await get_engine().capabilities()
        # 手动序列化 variants，确保 MiniMaxH3VariantInfo 扩展字段被保留
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        return result

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """满血 FP16 管线加载 + LightX2V Turbo 蒸馏 LoRA 注入。"""
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """释放模型 + 恢复 LoRA 旁路。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """视频生成推理（SSE 流式；蒸馏 8 步采样 → 编码 → completed 帧）。"""
        engine = get_engine()
        sse_stream = engine.infer(request)
        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（蒸馏 LoRA 由 load 注入，运行时接口预留）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（蒸馏 LoRA 由 load 注入，运行时接口不支持热插拔）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（骨架不支持，预留）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（本引擎骨架不支持，预留接口）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 通用 =====

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)


# ===== 入口 =====
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 MiniMaxH3Fast 引擎: host=%s, port=%d",
        _config.host, _config.port,
    )

    app = create_app(_config)

    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()