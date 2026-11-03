# 模型放置位置

部署到原项目后，把每个人完整的 `pretrained_model` 文件夹放到：

```text
SuperInference-feat-i2rt_NEDF3/checkpoints/pi0/自己的模型名/pretrained_model/
```

例如 `checkpoints/pi0/model_a/pretrained_model/model.safetensors`。

配置、归一化参数和处理器文件也要放入同一文件夹。模型权重不包含在本仓库中，也不会被部署脚本下载或上传。
