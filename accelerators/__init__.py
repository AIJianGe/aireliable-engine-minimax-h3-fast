"""MiniMaxH3Fast 加速器模块。

S3 实装：提供 LightX2V Turbo 蒸馏 LoRA 的旁路注入（`lora_loader.py`）。

本引擎定位"满血 FP16 + 社区蒸馏加速"：不引入 GGUF / BlockSwap / VAE tiling
（Turing 48G 充裕），蒸馏加速只靠步数蒸馏 LoRA（旁路 adapter）这一条。
"""