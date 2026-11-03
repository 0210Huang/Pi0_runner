import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deploy_rejects_wrong_directory_without_writing(tmp_path):
    host = tmp_path / 'wrong'
    host.mkdir()
    result = subprocess.run(['bash', str(ROOT / 'deploy.sh'), str(host)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert list(host.iterdir()) == []


def test_deploy_installs_complete_tools_and_preserves_act(tmp_path):
    host = tmp_path / 'robot project'
    worker = host / 'devices/robots/i2rt_teleop/worker.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('# existing hardware interface\n')
    act = host / 'scripts/inference/i2rt_act_runner.py'
    act.parent.mkdir(parents=True)
    act.write_text('original ACT')
    result = subprocess.run(['bash', str(ROOT / 'deploy.sh'), str(host)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ['i2rt_pi0_runner.py', 'i2rt_pi0_safety.py', 'run_pi0.sh']:
        assert (host / 'scripts/inference' / name).is_file()
    for name in ['setup_env.sh', 'check_environment.py', 'requirements.txt']:
        assert (host / '.superinference/local/pi0_runner' / name).is_file()
    assert (host / 'models/pi0_tokenizer/tokenizer.json').is_file()
    assert (host / 'checkpoints/pi0').is_dir()
    assert act.read_text() == 'original ACT'
    assert worker.read_text() == '# existing hardware interface\n'


def test_launcher_forwards_arguments_from_any_working_directory(tmp_path):
    host = tmp_path / 'robot project'
    runner_dir = host / 'scripts/inference'
    runner_dir.mkdir(parents=True)
    launcher = runner_dir / 'run_pi0.sh'
    launcher.write_bytes((ROOT / 'scripts/inference/run_pi0.sh').read_bytes())
    interpreter = tmp_path / 'python spy'
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    env = dict(os.environ, PI0_PYTHON=str(interpreter))
    result = subprocess.run(['bash', str(launcher), '--checkpoint',
                             'checkpoints/pi0/model a/pretrained_model', '--task', '放置 杯子'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['scripts/inference/i2rt_pi0_runner.py',
        '--checkpoint', 'checkpoints/pi0/model a/pretrained_model', '--task', '放置 杯子']
