import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).with_name('i2rt_pi0_runner.py')
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('i2rt_pi0_runner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_original_launch_parameters_accept_pi0_task_and_tokenizer(runner, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['i2rt_pi0_runner.py',
        '--checkpoint', 'checkpoints/pi0/model_a/pretrained_model',
        '--teleop-config', 'same.yaml', '--device-python', '.venv/bin/python',
        '--execute', '--confirm-motion', '--steps-per-burst', '40', '--bursts', '240',
        '--control-hz', '30', '--max-joint-delta', '0.05', '--max-gripper-delta', '0.08',
        '--hold-after-run', '--task', '测试', '--tokenizer-path', 'models/pi0_tokenizer'])
    args = runner.parse_args()
    runner.ensure_safe_args(args)
    assert args.task == '测试'
    assert args.steps_per_burst == 40 and args.bursts == 240
    assert args.stop_file == Path('/tmp/i2rt_act_stop')
    assert args.tokenizer_path == Path('models/pi0_tokenizer')


def test_motion_still_requires_both_flags(runner, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['runner', '--checkpoint', 'checkpoints/pi0/model_a/pretrained_model',
                                    '--task', '测试', '--execute'])
    with pytest.raises(SystemExit, match='Motion is gated'):
        runner.ensure_safe_args(runner.parse_args())


def test_restore_shared_weights_for_strict_loading(runner):
    model = torch.nn.Module()
    model.register_parameter('source', torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter('alias', model.source)
    state = runner.restore_aliases({'source': torch.tensor([2., 3.])},
                                   {'alias': 'source'}, model.state_dict())
    model.load_state_dict(state, strict=True)
    assert torch.equal(model.alias, torch.tensor([2., 3.]))
    with pytest.raises(ValueError):
        runner.restore_aliases({'source': torch.ones(3)}, {'alias': 'source'}, model.state_dict())


def test_predict_preserves_camera_order_task_and_absolute_action_units(runner):
    # 只替代大模型运算；观测转换和本脚本的数据路径使用真实实现。
    class Policy:
        def predict_action_chunk(self, batch):
            assert batch['task'] == '测试'
            assert batch['observation.state'].shape == (1, 14)
            assert batch['observation.images.left_wrist'].shape == (1, 3, 480, 640)
            assert torch.all(batch['observation.images.left_wrist'] == 1)
            assert torch.all(batch['observation.images.right_wrist'] == 0)
            return torch.full((1, 50, 14), 0.25)

    state = np.zeros(14, dtype=np.float32)
    chunk = runner.predict_chunk(state, np.full((480, 640, 3), 255, dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8), torch.device('cpu'), Policy(),
        lambda batch: batch, lambda action: action * 2 + 0.1, task='测试')
    assert chunk.shape == (50, 14)
    np.testing.assert_allclose(chunk, 0.6)


def test_nonfinite_prediction_stops(runner):
    class Policy:
        def predict_action_chunk(self, batch):
            return torch.full((1, 50, 14), float('nan'))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(runner.SafetyStop):
        runner.predict_chunk(np.zeros(14, dtype=np.float32), image, image,
            torch.device('cpu'), Policy(), lambda x: x, lambda x: x, task='测试')


def test_chunk_shorter_than_requested_execution_is_rejected(runner):
    with pytest.raises(ValueError):
        runner.validate_chunk_length(51, 50)
    runner.validate_chunk_length(40, 50)


@pytest.mark.parametrize('checkpoint,task', [
    ('checkpoints/pi0/model_a/pretrained_model', '拿起积木'),
    ('/tmp/another_model/pretrained_model', '放置杯子'),
])
def test_offline_mode_uses_each_users_checkpoint_and_task(
    runner, monkeypatch, capsys, tmp_path, checkpoint, task
):
    class Policy:
        def predict_action_chunk(self, batch):
            assert batch['task'] == task
            return torch.zeros((1, 50, 14))
    loaded = []
    def load(path, device, tokenizer):
        loaded.append(path)
        return torch.device('cpu'), Policy(), lambda x: x, lambda x: x
    monkeypatch.setattr(sys, 'argv', ['runner', '--checkpoint', checkpoint, '--task', task,
        '--offline-smoke-test', '--stop-file', str(tmp_path / 'stop')])
    monkeypatch.setattr(runner, 'load_policy', load)
    assert runner.main() == 0
    assert loaded == [Path(checkpoint)]
    assert 'OFFLINE PASS shape=(50, 14) finite=True' in capsys.readouterr().out


def test_stop_file_prevents_model_or_hardware_start(runner, monkeypatch, tmp_path):
    stop = tmp_path / 'stop'
    stop.touch()
    monkeypatch.setattr(sys, 'argv', ['runner', '--checkpoint', 'checkpoints/pi0/model_a/pretrained_model',
                                    '--task', '测试', '--stop-file', str(stop)])
    with pytest.raises(SystemExit, match='Refusing to start while stop file exists'):
        runner.main()


def test_checkpoint_must_be_explicit_to_avoid_loading_someone_elses_model(runner, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['runner', '--task', '测试'])
    with pytest.raises(SystemExit):
        runner.parse_args()
    assert '--checkpoint' in capsys.readouterr().err
