"""MiniMaxH3Fast 满血 FP16 视频生成引擎（骨架占位）。

定位：
    - MiniMax-H3 生态视频生成引擎，对标社区 4-8 步高低清（蒸馏加速 ×2 采样）
    - paradigm: integrated（inherit CustomEngine）
    - capability: video_generation（网关 POST /v1/videos/generation 硬性校验）
    - task_type : video_generation（任务中心分类）

⚠️ 骨架阶段：
    - 8 个标准端点 + 原生 WebUI 路由可跑通（**空载运行**）
    - 模型加载 / 蒸馏推理为清晰 TODO（S3 阶段实装），/infer 返回 501 占位
    - 评测未定档：variant、模型仓库 repo_id、key_files 均未定（占位）

进级契约（S3 实装时必须遵守，先写死防忘）：
    1. 本机 GPU 为 Turing（SM 7.5，Quadro RTX 8000 / 48GB）：**无 BF16 / FP8 /
       FlashAttention-2**，必须固定 FP16 + PyTorch SDPA，禁用 FP8 量化声明
    2. 产物契约：completed 帧 result 必须为
       {"videos": [{"video_path","video_url","width","height","duration"}]}，
       供网关 video_endpoints.py 解析
    3. 音色管理红线（project-rules §7.4/第六部分 6 继承）：**不得**读写
       data/voice_library.db 与 data/voices/
"""

__version__ = "0.1.0"