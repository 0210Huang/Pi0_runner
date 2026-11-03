#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
policy_python=${PI0_PYTHON:-"$project_root/.superinference/local/i2rt-pi0-policy/bin/python"}
if [[ ! -x "$policy_python" ]]; then
  echo "未找到π0环境，请先运行部署说明中的setup_env.sh：$policy_python" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$policy_python" scripts/inference/i2rt_pi0_runner.py "$@"
