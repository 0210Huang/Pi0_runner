#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -ne 1 ]]; then
  echo "用法：bash deploy.sh 现有SuperInference项目根目录" >&2
  exit 2
fi
project_root=$(cd "$1" && pwd)
if [[ ! -f "$project_root/devices/robots/i2rt_teleop/worker.py" ]]; then
  echo "目标目录缺少I2RT设备进程，请指定原SuperInference项目根目录。" >&2
  exit 2
fi
destination="$project_root/.superinference/local/pi0_runner"
mkdir -p "$destination" "$project_root/scripts/inference" \
  "$project_root/models/pi0_tokenizer" "$project_root/checkpoints/pi0"
for name in i2rt_pi0_runner.py i2rt_pi0_safety.py run_pi0.sh; do
  cp "$source_root/scripts/inference/$name" "$project_root/scripts/inference/$name"
done
cp "$source_root/tools/setup_env.sh" "$source_root/tools/check_environment.py" \
  "$source_root/requirements.txt" "$source_root/README.md" "$destination/"
cp "$source_root/models/pi0_tokenizer/"* "$project_root/models/pi0_tokenizer/"
echo "工具已部署到：$project_root"
echo "模型统一放在：$project_root/checkpoints/pi0/模型名/pretrained_model/"
echo "下一步创建独立环境："
printf 'bash %q --project-root %q\n' "$destination/setup_env.sh" "$project_root"
