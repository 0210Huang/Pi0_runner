# 来源说明

- I2RT硬件流程来自团队提供的ACT推理脚本，本仓库添加π0模型适配、部署与环境检查工具。
- LeRobot与Transformers依赖保留上游许可，安装脚本使用固定的官方提交，不包含模型权重。
- `models/pi0_tokenizer/` 为已有PaliGemma/Gemma分词器资源，其使用和再分发适用上游条款：[PaliGemma模型页面](https://huggingface.co/google/paligemma-3b-pt-224)、[Gemma使用条款](https://ai.google.dev/gemma/terms)。
- 原项目的设备驱动、现场配置和各自训练的模型不在本仓库中。
