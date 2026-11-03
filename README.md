# Pi0_runner

供同一套 I2RT 双臂设备共用的 LeRobot π0 推理工具。保留原 ACT 脚本的双腕相机、14维动作接口、限幅、停止文件和结束后保持姿态方式，通过参数选择各自的模型。

**本仓库包含完整工具、分词器、部署脚本、独立环境安装脚本、检查工具和测试；不包含、不会下载任何模型权重。**

## 模型必须放在哪里

在你们现有的机械臂项目中，统一使用：

```text
/home/group7/SuperInference-feat-i2rt_NEDF3/checkpoints/pi0/自己的模型名/pretrained_model/
```

例如两位同学分别放在：

```text
checkpoints/pi0/model_a/pretrained_model/
checkpoints/pi0/model_b/pretrained_model/
```

每个文件夹必须包含自己模型的整套推理文件：

```text
pretrained_model/
├── model.safetensors
├── config.json
├── policy_preprocessor.json
├── policy_postprocessor.json
└── 上述处理器引用的归一化参数文件（通常也是.safetensors）
```

不要只复制权重后沿用别人的配置或统计量。完整训练状态、优化器文件不需要复制。`model_a` 是示例名，每个人可使用自己的模型名或训练步数。

## 主机已有环境，哪些复用、哪些新建

| 部分 | 处理方式 |
|---|---|
| 原机械臂项目、设备进程、CAN配置、电机驱动 | 直接复用原来能运行ACT的配置 |
| 原项目的 `.venv/bin/python` | 继续用于设备进程，不重新安装 |
| 原 ACT 模型环境 | 保留，不修改 |
| π0 模型环境 | 首次创建一个独立环境，安装π0专用依赖；全组共用一次 |
| 分词器 | 已放在仓库 `models/pi0_tokenizer/`，部署时复制，无需再下载 |
| 各自训练的π0模型 | 各自复制到上面的固定目录 |

新环境位于原项目的 `.superinference/local/i2rt-pi0-policy/`。首次安装需要联网下载Python依赖，不是包含所有CUDA二进制依赖的离线系统镜像。之后实际推理使用本地文件，启动器默认关闭模型联网下载。

## 1. 在主机下载并部署工具

以下命令均在你们的 **Ubuntu机械臂主机终端** 执行。

```bash
cd /home/group7
git clone https://github.com/0210Huang/Pi0_runner.git
cd Pi0_runner
bash deploy.sh /home/group7/SuperInference-feat-i2rt_NEDF3
```

部署脚本只复制本工具文件，创建模型目录，不会启动机械臂、安装环境、下载权重或改动原ACT脚本。

部署后的结构：

```text
SuperInference-feat-i2rt_NEDF3/
├── devices/robots/i2rt_teleop/worker.py     # 原项目已有
├── .venv/                                # 原设备环境，复用
├── .superinference/local/
│   ├── i2rt_hardware_bindings.json         # 原相机身份配置，复用
│   ├── i2rt_launches/                     # 原设备配置，复用
│   ├── pi0_runner/                        # 本工具的安装和检查工具
│   └── i2rt-pi0-policy/                    # 下一步创建的π0独立环境
├── scripts/inference/
│   ├── i2rt_act_runner.py                 # 原脚本保留
│   ├── i2rt_pi0_runner.py
│   ├── i2rt_pi0_safety.py
│   └── run_pi0.sh
├── models/pi0_tokenizer/
└── checkpoints/pi0/模型名/pretrained_model/
```

## 2. 首次安装π0独立环境

```bash
cd /home/group7/SuperInference-feat-i2rt_NEDF3
bash .superinference/local/pi0_runner/setup_env.sh --project-root "$PWD"
```

需要 Linux x86_64、NVIDIA驱动、Git、Python 3.10—3.12及对应venv支持。原主机已经具备的系统组件无需重装。若默认 `python3` 不合适，可以显式选择已安装的解释器：

```bash
bash .superinference/local/pi0_runner/setup_env.sh \
  --project-root "$PWD" --python python3.12
```

`requirements.lock` 是Linux Python 3.12的完整依赖解析快照，用于复核和排查；安装脚本使用 `requirements.txt`，以兼容所选择的CUDA来源和Python版本。

安装脚本默认使用匹配的 `torch==2.7.1`、`torchvision==0.22.1` CUDA 12.6版本，并固定 LeRobot 0.4.4 和官方 π0 专用 Transformers 提交。不会把普通的最新版Transformers误当成兼容实现。

如果主机驱动需要 CUDA 11.8 对应的PyTorch包，可指定官方来源：

```bash
bash .superinference/local/pi0_runner/setup_env.sh \
  --project-root "$PWD" \
  --torch-index-url https://download.pytorch.org/whl/cu118
```

无需每位同学、每个模型重复创建环境。安装最后会运行只读环境检查；显示检查通过后再继续。

## 3. 放入模型并检查

以 `model_a` 为例，将你自己的整个 `pretrained_model` 文件夹复制到：

```text
/home/group7/SuperInference-feat-i2rt_NEDF3/checkpoints/pi0/model_a/pretrained_model/
```

然后运行：

```bash
cd /home/group7/SuperInference-feat-i2rt_NEDF3
.superinference/local/i2rt-pi0-policy/bin/python \
  .superinference/local/pi0_runner/check_environment.py \
  --project-root "$PWD" \
  --checkpoint checkpoints/pi0/model_a/pretrained_model \
  --teleop-config .superinference/local/i2rt_launches/yam_d405_2cam.can.act_infer_20261101.yaml
```

这个检查只核对依赖、CUDA、分词器、设备文件和模型文件，不启动设备或完整加载模型权重。

## 4. 运行真实模型的离线自检

使用虚拟状态和黑色图像，不连接机械臂或相机：

```bash
cd /home/group7/SuperInference-feat-i2rt_NEDF3
rm -f /tmp/i2rt_act_stop
bash scripts/inference/run_pi0.sh \
  --checkpoint checkpoints/pi0/model_a/pretrained_model \
  --task "自己训练时的任务文字" \
  --offline-smoke-test
```

预期出现 `PI0_WEIGHTS_STRICTLY_VERIFIED`，随后是 `OFFLINE PASS shape=(50, 14) finite=True`。50是示例动作块长度，实际由模型配置决定。

`--task` 必须使用训练时的文字。如果训练数据里就是“测试”，此处填写 `--task "测试"`，不要擅自换成另一段描述。

## 5. 现场读取相机和状态，但不发送策略动作

```bash
cd /home/group7/SuperInference-feat-i2rt_NEDF3
bash scripts/inference/run_pi0.sh \
  --checkpoint checkpoints/pi0/model_a/pretrained_model \
  --task "自己训练时的任务文字" \
  --teleop-config .superinference/local/i2rt_launches/yam_d405_2cam.can.act_infer_20261101.yaml \
  --device-python .venv/bin/python
```

省略运动标志时不打开策略运动会话，但会启动原项目的设备进程，设备初始化行为与原ACT脚本相同。结果保存在 `output/pi0_i2rt/`，检查左右腕图片、状态和预测动作是否对应。

## 6. 按原ACT方式运行完整任务

前面的检查通过后执行：

```bash
cd /home/group7/SuperInference-feat-i2rt_NEDF3
rm -f /tmp/i2rt_act_stop
bash scripts/inference/run_pi0.sh \
  --checkpoint checkpoints/pi0/model_a/pretrained_model \
  --task "自己训练时的任务文字" \
  --teleop-config .superinference/local/i2rt_launches/yam_d405_2cam.can.act_infer_20261101.yaml \
  --device-python .venv/bin/python \
  --execute --confirm-motion \
  --steps-per-burst 40 --bursts 240 \
  --control-hz 30 \
  --max-joint-delta 0.05 --max-gripper-delta 0.08 \
  --hold-after-run
```

每位同学主要更换 `--checkpoint`、`--task`。模型的归一化参数从各自检查点读取，不固定为某个人的数据统计量。

若想保持之前直接调用Python的形式，上述命令开头也可以换为：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.superinference/local/i2rt-pi0-policy/bin/python scripts/inference/i2rt_pi0_runner.py
```

后面接相同参数。`--tokenizer-path` 默认是 `models/pi0_tokenizer/`；有不同分词器时可显式指定。已配置好其他兼容π0环境时，可通过 `PI0_PYTHON=/实际解释器路径 bash scripts/inference/run_pi0.sh ...` 一次性覆盖模型解释器。

停止方式与原ACT一致，在另一个终端执行：

```bash
touch /tmp/i2rt_act_stop
```

也可使用 `Ctrl+C`。单次GPU前向期间不会发送新的策略动作，但停止信号要等前向返回后才能处理；物理急停仍按原项目操作。运行结束保持姿态的操作也沿用原脚本。

## 兼容范围及运行行为

- 同一套 I2RT 双臂设备，状态与动作顺序均为左臂6关节、左夹爪、右臂6关节、右夹爪。
- 绝对关节位置使用弧度，夹爪反馈和动作采用原项目的归一化开度。
- 特征名是 `observation.images.left_wrist`、`observation.images.right_wrist` 和 `observation.state`。
- 使用 LeRobot 0.4.4 完整π0检查点，不适用于ACT、π0-FAST、π0.5或单独的适配器权重。
- 每轮动作长度由模型决定，`--steps-per-burst` 不得超过该长度。
- `--control-hz 30` 控制发送动作阶段的目标频率；图像读取、大模型预测和保存也耗时，整个循环不是保证持续30赫兹的异步控制。
- 本仓库依赖你们已经能运行ACT的原SuperInference硬件项目，不重复打包设备驱动、现场CAN编号、相机序列号或用户模型。

## 验证与来源

运行不涉及真实机械臂的测试：

```bash
# 在本仓库中，用已有的兼容测试环境执行
python -m pip install pytest
python -m pytest -q
```

测试包含模型参数传递、严格共享权重恢复、图像及任务输入、动作反归一化、停止入口、部署到含空格路径、保留原ACT文件和模型目录检查。测试替代了大模型运算，不能代替每份真实模型的离线自检和真机验证。

本次未在实验室主机安装这个新环境或执行机械臂动作；安装、检查和运行命令交由使用者执行。仓库提供可复现的安装步骤，不声称已经在目标主机验证完毕。

- 硬件控制部分基于同学提供的 `i2rt_act_runner.py`、`i2rt_act_safety.py`。
- [LeRobot v0.4.4官方依赖定义](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/pyproject.toml)。
- [官方指定的π0 Transformers兼容提交](https://github.com/huggingface/transformers/tree/dcddb970176382c0fcf4521b0c0e6fc15894dfe0)。
- [PyTorch历史版本安装命令](https://pytorch.org/get-started/previous-versions/)、[TorchCodec版本对应表](https://github.com/meta-pytorch/torchcodec)。

分词器为已有PaliGemma/Gemma分词器资源，不包含神经网络权重；使用时保留其上游适用条款。模型权重、现场配置、日志和虚拟环境均被Git忽略。
