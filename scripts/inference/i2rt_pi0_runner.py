#!/usr/bin/env python3
"""供同一套I2RT双腕相机、14维双臂设备共用的LeRobot π0推理脚本。

The default mode is a hardware dry run: read follower state and both cameras,
run one PI0 forward pass, save the camera frames, and never open a control
session.  Motion requires both ``--execute`` and ``--confirm-motion``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Running the file directly places scripts/inference, not the repository root,
# on sys.path.  Add the root so the in-repository hardware client is importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from lerobot.policies.utils import prepare_observation_for_inference
from i2rt_pi0_safety import bounded_trial_action, smooth_action_target

DEFAULT_TELEOP_CONFIG = Path(
    ".superinference/local/i2rt_launches/yam_d405_2cam.can.teleop.yaml"
)
DEFAULT_BINDINGS = Path(".superinference/local/i2rt_hardware_bindings.json")
DEFAULT_DEVICE_PYTHON = Path(".venv/bin/python")
DEFAULT_STOP_FILE = Path("/tmp/i2rt_act_stop")


class SafetyStop(RuntimeError):
    """A deliberate fail-closed stop, not an error to recover through."""


class RpcEndpointClient:
    """Minimal client for the follower worker RPC contract.

    This is intentionally local to the runner: importing the collection-side
    coordinator would pull unrelated device dependencies into the policy
    environment.
    """

    def __init__(self, port: int, timeout_s: float, label: str) -> None:
        import portal

        self.timeout_s = timeout_s
        self.label = label
        self._client = portal.Client(f"127.0.0.1:{port}")

    def _start(self, method: str, *args: Any) -> Any:
        try:
            return getattr(self._client, method)(*args)
        except Exception as exc:
            raise RuntimeError(f"{self.label} {method} failed: {exc}") from exc

    def _finish(self, method: str, future: Any) -> Any:
        try:
            return future.result(timeout=self.timeout_s)
        except Exception as exc:
            raise RuntimeError(f"{self.label} {method} failed: {exc}") from exc

    def _call(self, method: str, *args: Any) -> Any:
        return self._finish(method, self._start(method, *args))

    def ping(self) -> Any:
        return self._call("ping")

    def start_get_sample(self) -> Any:
        return self._start("get_sample")

    def finish_get_sample(self, future: Any) -> Any:
        return self._finish("get_sample", future)

    def get_status(self) -> Any:
        return self._call("get_status")

    def begin_control_session(self, session_id: str) -> Any:
        return self._call("begin_control_session", session_id)

    def set_phase(self, session_id: str, phase: str) -> Any:
        return self._call("set_phase", session_id, phase)

    def submit_control_frame(self, value: dict[str, Any]) -> Any:
        return self._call("submit_control_frame", value)

    def end_control_session(self, session_id: str) -> Any:
        return self._call("end_control_session", session_id)

    def close(self) -> None:
        self._client.close(timeout=self.timeout_s)


@dataclass
class CameraPair:
    left: rs.pipeline
    right: rs.pipeline

    def stop(self) -> None:
        for pipeline in (self.left, self.right):
            try:
                pipeline.stop()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="自己的π0完整检查点目录，建议放在checkpoints/pi0/模型名/pretrained_model")
    parser.add_argument("--task", required=True, help="与训练数据一致的任务文字")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/pi0_tokenizer"),
                        help="本地PaliGemma分词器目录")
    parser.add_argument("--teleop-config", type=Path, default=DEFAULT_TELEOP_CONFIG)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--device-python", type=Path, default=DEFAULT_DEVICE_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pi0_i2rt"))
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_FILE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--left-port", type=int, default=1234)
    parser.add_argument("--right-port", type=int, default=1235)
    parser.add_argument("--rpc-timeout", type=float, default=2.0)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--offline-smoke-test",
        action="store_true",
        help="Load the checkpoint and run synthetic inputs without touching hardware.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--bounded-single-step", action="store_true",
        help="Explicit single-step trial: clip joint/gripper increments to at most 0.01.",
    )
    parser.add_argument(
        "--bounded-rollout", action="store_true",
        help="One bounded action chunk, at most 60 steps / 3 seconds; increments <=0.01.",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="Second explicit gate required together with --execute.",
    )
    parser.add_argument("--steps-per-burst", type=int, default=5)
    parser.add_argument(
        "--bursts",
        type=int,
        default=1,
        help="Number of observe/predict/execute bursts; keep 1 for first trial.",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-joint-delta", type=float, default=0.05)
    parser.add_argument("--max-gripper-delta", type=float, default=0.08)
    parser.add_argument("--temporal-ensemble", action="store_true",
                        help="Blend actions for the current step from overlapping PI0 chunks.")
    parser.add_argument("--ensemble-decay", type=float, default=0.5,
                        help="Weight older applicable chunks by exp(-decay * age rank).")
    parser.add_argument("--action-smoothing-alpha", type=float, default=1.0,
                        help="Low-pass PI0 targets against the last acknowledged command; 1 disables smoothing.")
    parser.add_argument("--hold-after-run", action="store_true",
                        help="Keep follower workers in safe hold after successful execution until stopped.")
    parser.add_argument("--verbose-steps", action="store_true",
                        help="Print the full per-step command trace to the terminal.")
    parser.add_argument("--save-images-every", type=int, default=10,
                        help="Save before/after camera PNGs every N bursts (default: 10).")
    parser.add_argument("--no-start-workers", action="store_true")
    parser.add_argument(
        "--align-to-state",
        type=Path,
        help="Smoothly align to a reviewed 14D follower state before dry-run.",
    )
    parser.add_argument(
        "--confirm-alignment",
        action="store_true",
        help="Explicit motion gate required with --align-to-state.",
    )
    parser.add_argument("--alignment-duration", type=float, default=5.0)
    parser.add_argument("--max-alignment-joint-delta", type=float, default=0.20)
    parser.add_argument("--max-alignment-gripper-delta", type=float, default=0.10)
    return parser.parse_args()


def ensure_safe_args(args: argparse.Namespace) -> None:
    if not args.task.strip():
        raise SystemExit("任务文字不能为空")
    if args.bounded_rollout and (
        args.bounded_single_step or not args.execute or args.bursts != 1
        or not 1 <= args.steps_per_burst <= 60
        or not 0 < args.max_joint_delta <= 0.01
        or not 0 < args.max_gripper_delta <= 0.01
    ):
        raise SystemExit("Bounded rollout requires execute, one burst, <=60 steps, and increments <=0.01")
    if args.bounded_single_step and (
        not args.execute or args.steps_per_burst != 1 or args.bursts != 1
        or args.max_joint_delta > 0.01 or args.max_gripper_delta > 0.01
    ):
        raise SystemExit("Bounded trial requires execute, one step, one burst, and delta limits <=0.01")
    if args.offline_smoke_test and (args.execute or args.confirm_motion):
        raise SystemExit("--offline-smoke-test cannot be combined with motion flags")
    if args.execute != args.confirm_motion:
        raise SystemExit(
            "Motion is gated: pass both --execute and --confirm-motion, or neither."
        )
    if (args.align_to_state is None) != (not args.confirm_alignment):
        raise SystemExit(
            "Alignment is gated: pass both --align-to-state PATH and "
            "--confirm-alignment, or neither."
        )
    if args.align_to_state is not None and args.execute:
        raise SystemExit(
            "Alignment and PI0 execution must be separate runs; remove --execute."
        )
    if args.steps_per_burst < 1 or args.steps_per_burst > 60:
        raise SystemExit("steps exceed the selected execution mode")
    if args.bursts < 1:
        raise SystemExit("--bursts must be positive")
    if args.control_hz <= 0 or args.control_hz > 30:
        raise SystemExit("--control-hz must be in (0, 30]")
    if args.max_joint_delta <= 0 or args.max_gripper_delta <= 0:
        raise SystemExit("delta limits must be positive")
    if not np.isfinite(args.ensemble_decay) or not 0 <= args.ensemble_decay <= 2:
        raise SystemExit("--ensemble-decay must be finite and in [0, 2]")
    if args.temporal_ensemble and (not args.execute or args.bounded_single_step or args.bounded_rollout):
        raise SystemExit("--temporal-ensemble requires normal PI0 execution")
    if not np.isfinite(args.action_smoothing_alpha) or not 0 < args.action_smoothing_alpha <= 1:
        raise SystemExit("--action-smoothing-alpha must be finite and in (0, 1]")
    if args.action_smoothing_alpha != 1 and (
        not args.execute or args.bounded_single_step or args.bounded_rollout
    ):
        raise SystemExit("action smoothing requires normal PI0 execution")
    if args.hold_after_run and not args.execute:
        raise SystemExit("--hold-after-run requires --execute")
    if args.save_images_every < 1:
        raise SystemExit("--save-images-every must be positive")
    if args.alignment_duration < 3.0:
        raise SystemExit("--alignment-duration must be at least 3 seconds")
    if (
        args.max_alignment_joint_delta <= 0
        or args.max_alignment_gripper_delta <= 0
    ):
        raise SystemExit("alignment delta limits must be positive")


def check_stop(stop_file: Path, interrupted: list[bool]) -> None:
    if interrupted[0]:
        raise SafetyStop("SIGINT/SIGTERM received")
    if stop_file.exists():
        raise SafetyStop(f"stop file exists: {stop_file}")


def read_camera_serials(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = payload["profiles"]["yam_d405_2cam"]["cameras"]
    return str(cameras["left"]["identity"]), str(cameras["right"]["identity"])


def start_one_camera(serial: str, fps: int) -> rs.pipeline:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    actual = profile.get_device().get_info(rs.camera_info.serial_number)
    if actual != serial:
        pipeline.stop()
        raise RuntimeError(f"camera identity mismatch: requested {serial}, got {actual}")
    for _ in range(5):
        pipeline.wait_for_frames(3000)
    return pipeline


def start_cameras(bindings: Path, fps: int) -> CameraPair:
    left_serial, right_serial = read_camera_serials(bindings)
    left = start_one_camera(left_serial, fps)
    try:
        right = start_one_camera(right_serial, fps)
    except Exception:
        left.stop()
        raise
    return CameraPair(left=left, right=right)


def get_rgb(pipeline: rs.pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames(3000)
    frame = frames.get_color_frame()
    if not frame:
        raise RuntimeError("RealSense returned no color frame")
    image = np.asanyarray(frame.get_data()).copy()
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected image contract: {image.shape} {image.dtype}")
    return image


def spawn_follower_workers(args: argparse.Namespace) -> list[subprocess.Popen[Any]]:
    processes: list[subprocess.Popen[Any]] = []
    for side in ("left", "right"):
        command = [
            str(args.device_python),
            "-m",
            "devices.robots.i2rt_teleop.worker",
            "--config",
            str(args.teleop_config),
            "--side",
            side,
            "--role",
            "follower",
        ]
        processes.append(subprocess.Popen(command, start_new_session=True))
    return processes


def stop_workers(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 8.0
    for process in processes:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()


def connect_workers(args: argparse.Namespace) -> tuple[RpcEndpointClient, RpcEndpointClient]:
    left = RpcEndpointClient(args.left_port, args.rpc_timeout, "left follower")
    right = RpcEndpointClient(args.right_port, args.rpc_timeout, "right follower")
    deadline = time.monotonic() + 15.0
    while True:
        try:
            left.ping()
            right.ping()
            return left, right
        except RuntimeError:
            if time.monotonic() >= deadline:
                left.close()
                right.close()
                raise
            time.sleep(0.2)


def read_state(
    left: RpcEndpointClient, right: RpcEndpointClient
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    left_future = left.start_get_sample()
    right_future = right.start_get_sample()
    left_sample = left.finish_get_sample(left_future)
    right_sample = right.finish_get_sample(right_future)
    left_state = np.asarray(left_sample["state"], dtype=np.float32).reshape(-1)
    right_state = np.asarray(right_sample["state"], dtype=np.float32).reshape(-1)
    if left_state.shape != (7,) or right_state.shape != (7,):
        raise RuntimeError(
            f"expected two 7D follower states, got {left_state.shape}/{right_state.shape}"
        )
    state = np.concatenate([left_state, right_state])
    if not np.all(np.isfinite(state)):
        raise SafetyStop("non-finite follower state")
    return state, left_sample, right_sample


def restore_aliases(state, metadata, expected):
    """恢复safetensors省略的共享权重，随后仍严格检查全部参数。"""
    state = state.copy()
    for alias, source in (metadata or {}).items():
        if alias not in state and alias in expected and source in state:
            if state[source].shape != expected[alias].shape:
                raise ValueError("共享权重尺寸不匹配：" + alias)
            state[alias] = state[source]
    return state


def validate_chunk_length(requested: int, available: int) -> None:
    if not 1 <= requested <= available:
        raise ValueError(f"每轮执行步数{requested}超过模型动作块长度{available}")


def load_policy(checkpoint: Path, device_name: str, tokenizer_path: Path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用CUDA，但当前环境无法使用显卡")
    checkpoint = checkpoint.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"未找到本地分词器：{tokenizer_path}")
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if not isinstance(cfg, PI0Config):
        raise ValueError("该目录不是π0模型，不能传入ACT检查点")
    if list(cfg.input_features["observation.state"].shape) != [14]:
        raise ValueError("该脚本要求14维双臂状态")
    if list(cfg.output_features["action"].shape) != [14]:
        raise ValueError("该脚本要求14维双臂动作")
    if set(cfg.image_features) != {
        "observation.images.left_wrist", "observation.images.right_wrist"
    }:
        raise ValueError("该脚本要求左腕、右腕两路相机输入")
    cfg.device = str(device)
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    # 与训练环境保持相同的π0专用Transformers实现，不绕过依赖检查。
    try:
        from transformers.models.siglip import check
        if not check.check_whether_transformers_replace_is_installed_correctly():
            raise ImportError("π0专用Transformers实现未通过检查")
    except ImportError as exc:
        raise RuntimeError("当前环境缺少训练时使用的π0专用Transformers依赖；请先按使用说明检查环境") from exc
    policy = PI0Policy(cfg)
    weights = checkpoint / "model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    state = restore_aliases(load_file(str(weights)), metadata, policy.state_dict())
    policy.load_state_dict(state, strict=True)
    del state
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print(f"PI0_WEIGHTS_STRICTLY_VERIFIED checkpoint={checkpoint}", flush=True)
    return device, policy, preprocessor, postprocessor


def predict_chunk(
    state: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    device: torch.device,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: str,
) -> np.ndarray:
    observation = {
        "observation.state": state.astype(np.float32, copy=False),
        "observation.images.left_wrist": left_image,
        "observation.images.right_wrist": right_image,
    }
    with torch.inference_mode():
        batch = prepare_observation_for_inference(observation, device, task=task)
        batch = preprocessor(batch)
        raw_chunk = policy.predict_action_chunk(batch)
        if raw_chunk.ndim == 2:
            raw_chunk = raw_chunk.unsqueeze(0)
        processed = [postprocessor(raw_chunk[:, index, :]) for index in range(raw_chunk.shape[1])]
        chunk = torch.stack(processed, dim=1).squeeze(0).float().cpu().numpy()
    if chunk.ndim != 2 or chunk.shape[1] != 14:
        raise RuntimeError(f"unexpected action chunk shape: {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise SafetyStop("policy produced a non-finite action")
    return chunk


def temporal_ensemble_actions(
    history: list[tuple[int, np.ndarray]], start_step: int,
    count: int, decay: float,
) -> np.ndarray:
    """Select actions by their global execution time, not chunk-local prefix."""
    selected = []
    for step in range(start_step, start_step + count):
        candidates = [chunk[step - origin] for origin, chunk in history
                      if 0 <= step - origin < len(chunk)]
        if not candidates:
            raise ValueError(f"no PI0 prediction covers global step {step}")
        weights = np.exp(-decay * np.arange(len(candidates), dtype=np.float64))
        selected.append(np.average(np.stack(candidates), axis=0, weights=weights))
    return np.asarray(selected, dtype=np.float32)


def validate_action(action: np.ndarray, reference: np.ndarray, args: argparse.Namespace) -> None:
    delta = np.abs(action - reference)
    joint_delta = np.concatenate((delta[:6], delta[7:13]))
    gripper_delta = delta[[6, 13]]
    if float(joint_delta.max()) > args.max_joint_delta:
        raise SafetyStop(
            f"joint delta {joint_delta.max():.4f} exceeds {args.max_joint_delta:.4f} rad"
        )
    if float(gripper_delta.max()) > args.max_gripper_delta:
        raise SafetyStop(
            f"gripper delta {gripper_delta.max():.4f} exceeds {args.max_gripper_delta:.4f}"
        )


def frame(session_id: str, sequence: int, phase: str, command: np.ndarray) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "sequence": sequence,
        "issued_at_ns": time.monotonic_ns(),
        "phase": phase,
        "joint_position": command.astype(np.float64),
    }


def wait_applied(client: RpcEndpointClient, sequence: int, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get_status()
        if int(status["last_applied_sequence"]) >= sequence:
            return
        time.sleep(0.005)
    raise SafetyStop(f"worker did not apply sequence {sequence} within {timeout}s")


def execute_burst(
    chunk: np.ndarray,
    current_state: np.ndarray,
    left: RpcEndpointClient,
    right: RpcEndpointClient,
    args: argparse.Namespace,
    interrupted: list[bool],
    previous_command: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    session_id = f"pi0-{uuid.uuid4().hex}"
    trace: list[dict[str, Any]] = []
    opened: list[RpcEndpointClient] = []
    try:
        left.begin_control_session(session_id)
        opened.append(left)
        right.begin_control_session(session_id)
        opened.append(right)
        left.submit_control_frame(frame(session_id, 0, "ALIGNING", current_state[:7]))
        right.submit_control_frame(frame(session_id, 0, "ALIGNING", current_state[7:]))
        wait_applied(left, 0)
        wait_applied(right, 0)
        left.set_phase(session_id, "ACTIVE")
        right.set_phase(session_id, "ACTIVE")

        period = 1.0 / args.control_hz
        rollout_start = time.monotonic()
        for index, action in enumerate(chunk[: args.steps_per_burst], start=1):
            started = time.monotonic()
            check_stop(args.stop_file, interrupted)
            measured_state, left_sample, right_sample = read_state(left, right)
            if left_sample["fault_reason"] or right_sample["fault_reason"]:
                raise SafetyStop("worker fault before PI0 command")
            requested = action.copy()
            if args.action_smoothing_alpha != 1:
                reference = measured_state if previous_command is None else previous_command
                action = smooth_action_target(action, reference, args.action_smoothing_alpha)
            if args.bounded_rollout and time.monotonic() - rollout_start >= 3:
                raise SafetyStop("bounded rollout reached its 3-second time limit")
            if args.bounded_single_step or args.bounded_rollout:
                action = bounded_trial_action(
                    action, measured_state, args.max_joint_delta, args.max_gripper_delta
                )
            else:
                # Move toward the policy target one bounded step at a time.
                limits = np.full(14, args.max_joint_delta * (1 - 1e-6))
                limits[[6, 13]] = args.max_gripper_delta * (1 - 1e-6)
                action = measured_state + np.clip(
                    action - measured_state, -limits, limits
                )
                action[[6, 13]] = np.clip(action[[6, 13]], 0.0, 1.0)
            if args.bounded_rollout:
                total_delta = np.abs(action - current_state)
                if np.max(np.r_[total_delta[:6], total_delta[7:13]]) > 0.15:
                    raise SafetyStop("bounded rollout reached its 0.15-rad start-position envelope")
                if np.max(total_delta[[6, 13]]) > 0.10:
                    raise SafetyStop("bounded rollout reached its 0.10 gripper-opening envelope")
            validate_action(action, measured_state, args)
            left.submit_control_frame(frame(session_id, index, "ACTIVE", action[:7]))
            right.submit_control_frame(frame(session_id, index, "ACTIVE", action[7:]))
            wait_applied(left, index)
            wait_applied(right, index)
            previous_command = action.copy()
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            entry = {
                "sequence": index, "requested_action": requested.tolist(),
                "command": action.tolist(), "measured_before": measured_state.tolist(),
                "command_delta": (action - measured_state).tolist(),
            }
            trace.append(entry)
            if args.verbose_steps:
                print("PI0_STEP_APPLIED " + json.dumps(entry), flush=True)
    finally:
        for client in opened:
            try:
                client.end_control_session(session_id)
            except Exception as exc:
                print(f"WARNING: failed to end one control session: {exc}", file=sys.stderr)
    return trace, previous_command


def align_to_reviewed_state(
    target: np.ndarray,
    left: RpcEndpointClient,
    right: RpcEndpointClient,
    args: argparse.Namespace,
    interrupted: list[bool],
) -> None:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if target.shape != (14,) or not np.all(np.isfinite(target)):
        raise SafetyStop("alignment target must be a finite 14D state")
    start, _, _ = read_state(left, right)
    delta = np.abs(target - start)
    joint_delta = np.r_[delta[:6], delta[7:13]]
    gripper_delta = delta[[6, 13]]
    if float(joint_delta.max()) > args.max_alignment_joint_delta:
        raise SafetyStop(
            f"alignment joint delta {joint_delta.max():.4f} exceeds reviewed "
            f"limit {args.max_alignment_joint_delta:.4f} rad"
        )
    if float(gripper_delta.max()) > args.max_alignment_gripper_delta:
        raise SafetyStop(
            f"alignment gripper delta {gripper_delta.max():.4f} exceeds reviewed "
            f"limit {args.max_alignment_gripper_delta:.4f}"
        )

    session_id = f"align-{uuid.uuid4().hex}"
    opened: list[RpcEndpointClient] = []
    try:
        left.begin_control_session(session_id)
        opened.append(left)
        right.begin_control_session(session_id)
        opened.append(right)
        steps = max(1, int(round(args.alignment_duration * args.control_hz)))
        period = args.alignment_duration / steps
        for sequence in range(steps + 1):
            started = time.monotonic()
            check_stop(args.stop_file, interrupted)
            linear = sequence / steps
            alpha = linear * linear * (3.0 - 2.0 * linear)
            command = start * (1.0 - alpha) + target * alpha
            left.submit_control_frame(
                frame(session_id, sequence, "ALIGNING", command[:7])
            )
            right.submit_control_frame(
                frame(session_id, sequence, "ALIGNING", command[7:])
            )
            wait_applied(left, sequence)
            wait_applied(right, sequence)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        # Keep renewing the 150 ms command lease while the mechanism settles.
        # A passive sleep here would correctly trip the worker watchdog.
        settle_deadline = time.monotonic() + 1.0
        sequence = steps
        while True:
            measured, _, _ = read_state(left, right)
            final_error = np.abs(measured - target)
            final_joint_error = float(
                np.max(np.r_[final_error[:6], final_error[7:13]])
            )
            if final_joint_error <= 0.04:
                break
            if time.monotonic() >= settle_deadline:
                raise SafetyStop(
                    "alignment did not settle: max joint error "
                    f"{final_joint_error:.4f} rad"
                )
            sequence += 1
            left.submit_control_frame(
                frame(session_id, sequence, "ALIGNING", target[:7])
            )
            right.submit_control_frame(
                frame(session_id, sequence, "ALIGNING", target[7:])
            )
            wait_applied(left, sequence)
            wait_applied(right, sequence)
            time.sleep(period)
        print(
            f"ALIGNMENT PASS start_max_joint_delta={joint_delta.max():.4f} "
            f"final_max_joint_error={final_joint_error:.4f}"
        )
    finally:
        for client in opened:
            try:
                client.end_control_session(session_id)
            except Exception as exc:
                print(f"WARNING: failed to end alignment session: {exc}", file=sys.stderr)


def save_dry_run(
    output_dir: Path,
    state: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    chunk: np.ndarray,
    save_images: bool = True,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    destination = output_dir / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    if save_images:
        Image.fromarray(left_image).save(destination / "left_wrist.png")
        Image.fromarray(right_image).save(destination / "right_wrist.png")
    np.save(destination / "state.npy", state)
    np.save(destination / "action_chunk.npy", chunk)
    summary = {
        "state": state.tolist(),
        "first_action": chunk[0].tolist(),
        "first_action_delta": (chunk[0] - state).tolist(),
        "action_shape": list(chunk.shape),
        "images_saved": save_images,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return destination


def main() -> int:
    args = parse_args()
    ensure_safe_args(args)
    if args.stop_file.exists():
        raise SystemExit(
            f"Refusing to start while stop file exists: {args.stop_file}. "
            "Remove it explicitly after checking the workspace."
        )
    interrupted = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.offline_smoke_test:
        device, policy, preprocessor, postprocessor = load_policy(
            args.checkpoint, args.device, args.tokenizer_path
        )
        state = np.zeros(14, dtype=np.float32)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        chunk = predict_chunk(
            state,
            image,
            image.copy(),
            device,
            policy,
            preprocessor,
            postprocessor,
            task=args.task,
        )
        print(
            f"OFFLINE PASS shape={chunk.shape} finite={np.all(np.isfinite(chunk))} "
            f"inference={time.perf_counter() - started:.3f}s"
        )
        return 0

    workers: list[subprocess.Popen[Any]] = []
    cameras: CameraPair | None = None
    clients: tuple[RpcEndpointClient, RpcEndpointClient] | None = None
    try:
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"task={args.task!r}", flush=True)
        device, policy, preprocessor, postprocessor = load_policy(
            args.checkpoint, args.device, args.tokenizer_path
        )
        validate_chunk_length(args.steps_per_burst, policy.config.chunk_size)
        if not args.no_start_workers:
            workers = spawn_follower_workers(args)
        clients = connect_workers(args)
        left, right = clients
        if args.align_to_state is not None:
            align_to_reviewed_state(
                np.load(args.align_to_state), left, right, args, interrupted
            )
        cameras = start_cameras(args.bindings, args.camera_fps)

        history: list[tuple[int, np.ndarray]] = []
        previous_command: np.ndarray | None = None
        global_step = 0
        for burst_index in range(args.bursts):
            check_stop(args.stop_file, interrupted)
            state, left_sample, right_sample = read_state(left, right)
            if left_sample["fault_reason"] or right_sample["fault_reason"]:
                raise SafetyStop(
                    f"worker fault: left={left_sample['fault_reason']!r}, "
                    f"right={right_sample['fault_reason']!r}"
                )
            left_image = get_rgb(cameras.left)
            right_image = get_rgb(cameras.right)
            started = time.perf_counter()
            chunk = predict_chunk(
                state,
                left_image,
                right_image,
                device,
                policy,
                preprocessor,
                postprocessor,
                task=args.task,
            )
            validate_chunk_length(args.steps_per_burst, len(chunk))
            inference_s = time.perf_counter() - started
            save_images = not args.execute or burst_index % args.save_images_every == 0
            destination = save_dry_run(
                args.output_dir, state, left_image, right_image, chunk, save_images
            )
            first_delta = np.abs(chunk[0] - state)
            print(
                f"burst={burst_index + 1}/{args.bursts} inference={inference_s:.3f}s "
                f"first_joint_delta={np.r_[first_delta[:6], first_delta[7:13]].max():.4f} "
                f"first_gripper_delta={first_delta[[6, 13]].max():.4f} "
                f"saved={destination}"
            )
            if not args.execute:
                break
            if args.temporal_ensemble:
                history.append((global_step, chunk))
                history = [(origin, prior) for origin, prior in history
                           if global_step - origin < len(prior)]
                execution_chunk = temporal_ensemble_actions(
                    history, global_step, args.steps_per_burst, args.ensemble_decay
                )
            else:
                execution_chunk = chunk[:args.steps_per_burst]
            np.save(destination / "selected_action_chunk.npy", execution_chunk)
            trace, previous_command = execute_burst(
                execution_chunk, state, left, right, args, interrupted, previous_command
            )
            (destination / "execution_trace.json").write_text(json.dumps(trace), encoding="utf-8")
            global_step += len(execution_chunk)
            final_state, _, _ = read_state(left, right)
            np.save(destination / "state_after.npy", final_state)
            if save_images:
                Image.fromarray(get_rgb(cameras.left)).save(destination / "left_wrist_after.png")
                Image.fromarray(get_rgb(cameras.right)).save(destination / "right_wrist_after.png")
        if args.execute and args.hold_after_run:
            print("PI0 COMPLETE: followers holding current pose. Support both arms before Ctrl-C or stop-file release.", flush=True)
            while not interrupted[0] and not args.stop_file.exists():
                time.sleep(0.1)
        return 0
    except SafetyStop as exc:
        print(f"SAFETY STOP: {exc}", file=sys.stderr)
        return 2
    finally:
        if clients is not None:
            for client in clients:
                try:
                    client.close()
                except Exception:
                    pass
        if cameras is not None:
            cameras.stop()
        stop_workers(workers)


if __name__ == "__main__":
    raise SystemExit(main())
