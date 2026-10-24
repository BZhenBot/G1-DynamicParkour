"""Explicit current-observation supervisor for two genuine frozen PPO policies.

Effective actor120: trained jump114 + current world-axis foot/platform offsets6.
No contact surface identity, privileged FSM/verdict, critic or future trajectory.
"""
import math

import torch
from tensordict import TensorDict
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg


def current_relative_feet(env):
    robot, target = env.scene["robot"], env.scene["target"]
    feet_ids = robot.find_sites(("left_foot", "right_foot"))[0]
    center = target.data.root_link_pos_w.clone()
    center[:, 1] += target.data.joint_pos[:, 0]
    center[:, 2] += env.geometry.half_thickness
    # Current FK and measured platform pose expressed in known world axes.
    # Hardware estimation accuracy/latency has not yet been verified.
    return (robot.data.site_pos_w[:, feet_ids] - center[:, None, :]).flatten(1)


def attach_public_supervisor_observations(cfg):
    cfg.observations["supervisor_current"] = ObservationGroupCfg(
        terms={"current_relative_feet_world": ObservationTermCfg(func=current_relative_feet)},
        enable_corruption=False,
    )
    return cfg


class PublicObservationHandoff:
    """One-bit causal latch; all switch decisions use only supplied current actor tensors."""

    def __init__(self, jump_policy, actor, geometry, num_envs, device, thresholds=None, transition_seconds=0., control_dt=.04):
        self.thresholds = {"tilt_deg":15., "angular_speed":.5, "horizontal_speed":.15, "vertical_speed":.1}
        overrides = thresholds or {}
        if set(overrides) - set(self.thresholds):
            raise ValueError("Unknown supervisor threshold")
        self.thresholds.update(overrides)
        if not all(math.isfinite(v) and v > 0 for v in self.thresholds.values()):
            raise ValueError("Switch thresholds must be finite and positive")
        if self.thresholds["tilt_deg"] > 45 or self.thresholds["angular_speed"] > 5 or self.thresholds["horizontal_speed"] > 1 or self.thresholds["vertical_speed"] > .5:
            raise ValueError("Supervisor development thresholds exceed preregistered bounds")
        self.jump_policy = jump_policy
        self.actor = actor
        self.bounds = torch.tensor(
            [geometry.target_half_length - .02, geometry.target_half_width - .02], device=device
        )
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if not math.isfinite(transition_seconds) or not 0 <= transition_seconds <= 2:
            raise ValueError("Transition duration must be finite and within [0,2] seconds")
        self.transition_seconds = transition_seconds
        self.control_dt = control_dt
        self.alpha = torch.zeros(num_envs, device=device)
        self.last_ready = torch.zeros_like(self.active)
        self.last_new_handoff = torch.zeros_like(self.active)
        self.last_public_observation = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.active[ids] = False
        self.alpha[ids] = 0.
        self.last_ready[ids] = False
        self.last_new_handoff[ids] = False

    def ready(self, actor114, relative_feet6):
        if actor114.ndim != 2 or actor114.shape[1] != 114:
            raise ValueError("Public handoff requires the documented current actor114 layout")
        if relative_feet6.shape != (actor114.shape[0], 6):
            raise ValueError("Current two-foot relative positions must have six components")
        feet = relative_feet6.reshape(-1, 2, 3)
        finite = torch.isfinite(actor114).all(-1) & torch.isfinite(relative_feet6).all(-1)
        contact = actor114[:, 106:108].gt(.5).all(-1)
        inside = (feet[:, :, :2].abs() <= self.bounds).all(-1).all(-1)
        on_top = ((feet[:, :, 2] >= -.01) & (feet[:, :, 2] <= .04)).all(-1)
        upright = -actor114[:, 8] >= math.cos(math.radians(self.thresholds["tilt_deg"]))
        angular_slow = torch.linalg.vector_norm(actor114[:, 3:6], dim=-1) <= self.thresholds["angular_speed"]
        horizontal_slow = torch.linalg.vector_norm(actor114[:, 102:104], dim=-1) <= self.thresholds["horizontal_speed"]
        vertical_slow = actor114[:, 104].abs() <= self.thresholds["vertical_speed"]
        above_platform = -actor114[:, 101] >= .5
        clock_elapsed = actor114[:, 105] >= 1
        return finite & contact & inside & on_top & upright & angular_slow & horizontal_slow & vertical_slow & above_platform & clock_elapsed

    def __call__(self, obs):
        actor114, feet6 = obs["actor"], obs["supervisor_current"]
        self.last_public_observation = torch.cat((actor114, feet6), -1).detach().clone()
        self.last_ready = self.ready(actor114, feet6)
        self.last_new_handoff = self.last_ready & ~self.active
        self.active |= self.last_ready
        increment = self.control_dt / self.transition_seconds if self.transition_seconds else 1.
        self.alpha = torch.where(self.active, (self.alpha + increment).clamp(max=1.), self.alpha)
        # Explicitly exclude critic and all other observation groups from both policies.
        only_actor = TensorDict({"actor": actor114}, batch_size=obs.batch_size)
        action = self.jump_policy(only_actor).clone()
        if self.active.any():
            proprio = TensorDict({"actor": actor114[:, :99]}, batch_size=obs.batch_size)
            standing = self.actor.jump_prior.standing_prior(proprio).clamp(-1, 1) / self.actor.jump_prior.physical_units
            # Blend actual applied joint commands of two state-feedback PPO policies.
            # Latent30 clipping and current direction match the real action adapter.
            jumping = (action[:, :29].clamp(-1, 1) + action[:, 29:30].clamp(-1, 1) * self.actor.last_direction).clamp(-1, 1)
            blended = ((1 - self.alpha[:, None]) * jumping + self.alpha[:, None] * standing).clamp(-1, 1)
            action[self.active, :29] = blended[self.active]
            action[self.active, 29] = 0.
        return action
