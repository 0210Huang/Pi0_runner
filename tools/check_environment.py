#!/usr/bin/env python3
"""只读检查π0环境和模型目录，不连接相机、不启动设备。"""

import argparse
import importlib.metadata
import json
from pathlib import Path


def validate_checkpoint(path):
    path = Path(path)
    for name in ['config.json', 'model.safetensors', 'policy_preprocessor.json',
                 'policy_postprocessor.json']:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'模型目录缺少有效文件：{file}')
    cfg = json.loads((path / 'config.json').read_text())
    if cfg.get('type') != 'pi0':
        raise ValueError('检查点必须为LeRobot π0完整模型')
    inputs, outputs = cfg['input_features'], cfg['output_features']
    if inputs['observation.state']['shape'] != [14] or outputs['action']['shape'] != [14]:
        raise ValueError('状态和动作必须均为14维')
    cameras = {key for key in inputs if key.startswith('observation.images.')}
    if cameras != {'observation.images.left_wrist', 'observation.images.right_wrist'}:
        raise ValueError('相机必须为原项目的左右腕两路输入')
    for name in ['policy_preprocessor.json', 'policy_postprocessor.json']:
        pipeline = json.loads((path / name).read_text())
        for step in pipeline['steps']:
            if 'state_file' in step and not (path / step['state_file']).is_file():
                raise ValueError(f"处理器缺少参数文件：{step['state_file']}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--teleop-config', type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    for file in ['devices/robots/i2rt_teleop/worker.py', '.venv/bin/python',
                 '.superinference/local/i2rt_hardware_bindings.json']:
        if not (root / file).is_file():
            raise SystemExit(f'缺少原项目文件：{root / file}')
    if args.teleop_config:
        config = args.teleop_config if args.teleop_config.is_absolute() else root / args.teleop_config
        if not config.is_file():
            raise SystemExit(f'找不到设备配置：{config}')
    import torch
    import portal
    import pyrealsense2
    from transformers import AutoTokenizer
    from transformers.models.siglip import check
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    del portal, pyrealsense2, PI0Policy
    if importlib.metadata.version('lerobot') != '0.4.4':
        raise SystemExit('该工具适配LeRobot 0.4.4，请使用安装脚本配置的独立环境')
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise SystemExit('π0专用Transformers兼容检查失败')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA不可用，请核对NVIDIA驱动与PyTorch版本')
    AutoTokenizer.from_pretrained(str(root / 'models/pi0_tokenizer'), local_files_only=True)
    print('π0组件、相机通信依赖、CUDA及本地分词器检查通过')
    print('显卡：', torch.cuda.get_device_name(0))
    if args.checkpoint:
        path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
        cfg = validate_checkpoint(path.expanduser())
        if cfg.get('dtype') == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise SystemExit('当前模型使用bfloat16，但显卡不支持；不要直接执行')
        print('模型目录检查通过：', path)
        print('动作块长度：', cfg['chunk_size'])
    print('该检查没有加载完整权重或发送动作；下一步运行离线模型自检。')


if __name__ == '__main__':
    main()
