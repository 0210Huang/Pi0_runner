"""Bound one explicitly authorized ACT trial command; no hardware or model imports."""

import numpy as np


def smooth_action_target(action, previous_command, alpha):
    """Low-pass a policy target against the last acknowledged command.

    This only shapes the target. The runner still applies its per-step limits
    against fresh follower feedback before dispatching either arm.
    """
    target = np.asarray(action, dtype=np.float64)
    previous = np.asarray(previous_command, dtype=np.float64)
    if target.shape != (14,) or previous.shape != (14,):
        raise ValueError("ACT smoothing requires two 14D vectors")
    if not np.isfinite(target).all() or not np.isfinite(previous).all():
        raise ValueError("ACT smoothing requires finite action vectors")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("ACT smoothing alpha must be in (0, 1]")
    return previous + alpha * (target - previous)


def bounded_trial_action(action, measured_state, max_joint_delta, max_gripper_delta):
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(measured_state, dtype=np.float64)
    if action.shape != (14,) or state.shape != (14,):
        raise ValueError("Trial requires two 14D vectors")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Trial requires finite state and action")
    for limit in (max_joint_delta, max_gripper_delta):
        if not np.isfinite(limit) or not 0 < limit <= 0.01:
            raise ValueError("Trial increment limits must be in (0, 0.01]")
    if np.any((state[[6, 13]] < 0) | (state[[6, 13]] > 1)):
        raise ValueError("Trial requires normalized gripper feedback in [0, 1]")
    # A small inward margin avoids exceeding the inclusive safety gate through rounding.
    limits = np.full(14, max_joint_delta * (1 - 1e-6))
    limits[[6, 13]] = max_gripper_delta * (1 - 1e-6)
    result = state + np.clip(action - state, -limits, limits)
    result[[6, 13]] = np.clip(result[[6, 13]], 0, 1)
    return result
