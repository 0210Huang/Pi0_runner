import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('check_environment',
    Path(__file__).resolve().parents[1] / 'tools/check_environment.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def checkpoint(path):
    cfg = {'type': 'pi0', 'input_features': {
        'observation.state': {'shape': [14]},
        'observation.images.left_wrist': {'shape': [3, 480, 640]},
        'observation.images.right_wrist': {'shape': [3, 480, 640]}},
        'output_features': {'action': {'shape': [14]}}, 'chunk_size': 50}
    (path / 'config.json').write_text(json.dumps(cfg))
    (path / 'model.safetensors').write_bytes(b'placeholder: weights are not loaded by this check')
    for name in ['policy_preprocessor.json', 'policy_postprocessor.json']:
        (path / name).write_text(json.dumps({'steps': [{'state_file': 'stats.safetensors'}]}))
    (path / 'stats.safetensors').write_bytes(b'stats')
    return cfg


def test_complete_model_directory_passes(tmp_path):
    checkpoint(tmp_path)
    assert checker.validate_checkpoint(tmp_path)['chunk_size'] == 50


def test_missing_normalization_stats_is_rejected(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / 'stats.safetensors').unlink()
    with pytest.raises(ValueError, match='参数文件'):
        checker.validate_checkpoint(tmp_path)


def test_act_checkpoint_is_rejected(tmp_path):
    cfg = checkpoint(tmp_path)
    cfg['type'] = 'act'
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='π0'):
        checker.validate_checkpoint(tmp_path)
