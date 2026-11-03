#!/usr/bin/env bash
set -euo pipefail
tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=""
base_python=python3
torch_index_url=https://download.pytorch.org/whl/cu126
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) project_root=${2:?缺少项目目录}; shift 2 ;;
    --python) base_python=${2:?缺少Python命令}; shift 2 ;;
    --torch-index-url) torch_index_url=${2:?缺少PyTorch下载源}; shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$project_root" ]]; then
  echo "用法：bash setup_env.sh --project-root 项目目录 [--python python3.12]" >&2
  exit 2
fi
project_root=$(cd "$project_root" && pwd)
[[ -f "$project_root/devices/robots/i2rt_teleop/worker.py" ]] || {
  echo "目标不是原I2RT项目，未安装任何内容。" >&2; exit 2;
}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo "本安装脚本用于Linux x86_64 NVIDIA主机。" >&2; exit 2;
}
command -v git >/dev/null
"$base_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "请使用Python 3.10—3.12"'
environment="$project_root/.superinference/local/i2rt-pi0-policy"
if [[ ! -e "$environment" ]]; then
  "$base_python" -m venv "$environment"
elif [[ ! -x "$environment/bin/python" ]]; then
  echo "环境目录已存在但不完整，请检查：$environment" >&2; exit 2
fi
policy_python="$environment/bin/python"
requirements="$tools_dir/requirements.txt"
if [[ ! -f "$requirements" ]]; then
  requirements="$tools_dir/../requirements.txt"
fi
"$policy_python" -m pip install --upgrade pip
"$policy_python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$torch_index_url"
"$policy_python" -m pip install -r "$requirements"
"$policy_python" -m pip check
"$policy_python" "$tools_dir/check_environment.py" --project-root "$project_root"
echo "独立π0环境安装并检查通过，原ACT环境未修改。"
