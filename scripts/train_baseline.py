"""复用 mjlab/rsl_rl PPO 的基础站立/步行入口；尚不包含跳跃任务。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import importlib.metadata

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.utils.os import dump_yaml


def fallen(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2] < 0.5


def illegal_ground_contact(env):
    force = env.scene["nonfoot_ground"].data.force
    return (torch.linalg.vector_norm(force, dim=-1) > 10.0).any(dim=-1)


class BaselineEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, directory, **kwargs):
        self.directory = directory
        self.episode_index = [0] * cfg.scene.num_envs
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else env_ids.detach().cpu().tolist()
        lengths = self.episode_length_buf.detach().cpu().tolist()
        if self.common_step_counter > 0:
            sums = {name: value.detach().cpu().tolist() for name, value in self.reward_manager._episode_sums.items()}
            heights = self.scene["robot"].data.root_link_pos_w[:, 2].detach().cpu().tolist()
            terminators = {name: self.termination_manager.get_term(name).detach().cpu().tolist()
                           for name in self.termination_manager.active_terms}
            with (self.directory / "episodes.jsonl").open("a", encoding="utf-8") as file:
                for env_id in ids:
                    if lengths[env_id] == 0:
                        continue
                    terms = {name: float(value[env_id]) for name, value in sums.items()}
                    reasons = [name for name, value in terminators.items() if value[env_id]]
                    row = {"env_id": env_id, "episode_id": self.episode_index[env_id], "seed": getattr(self,"evaluation_seed",self.cfg.seed),
                           "initialization_seed": self.cfg.seed,
                           "control_steps": lengths[env_id], "duration_s": lengths[env_id] * self.step_dt,
                           "reward_terms": terms, "episode_reward": sum(terms.values()),
                           "terminal_height_m": heights[env_id], "termination_reasons": reasons,
                           "task": "baseline_stand_or_walk", "jump_events": "不适用：基础任务"}
                    row.update(self.episode_details(env_id))
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.episode_index[env_id] += 1
        super()._reset_idx(env_ids)

    def episode_details(self, env_id):
        return {}


def alive(env):
    return (~env.reset_terminated).float()


def root_angular_velocity_l2(env):
    return torch.sum(env.scene["robot"].data.root_link_ang_vel_b.square(), dim=-1)


def build_configs(settings):
    cfg = unitree_g1_flat_env_cfg()
    cfg.scene.num_envs = settings["num_envs"]
    cfg.seed = settings["seed"]
    cfg.episode_length_s = settings["episode_length_s"]
    cfg.events = {"reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset")}
    cfg.curriculum = {}
    command = cfg.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = tuple(settings["velocity_x"])
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 1.0 if settings["task"] == "standing" else settings.get("standing_fraction", 0.0)
    if settings.get("world_forward_heading_hold"):
        command.heading_command = True
        command.ranges.heading = (0.0, 0.0)
        command.rel_heading_envs = 1.0
        command.rel_world_envs = 1.0
        command.rel_forward_envs = 0.0
        command.ranges.ang_vel_z = (-0.5, 0.5)
    nonfoot = ContactSensorCfg(name="nonfoot_ground",
        primary=ContactMatch(mode="geom", pattern=".*_collision", entity="robot", exclude=(".*_foot[1-7]_collision",)),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1)
    cfg.scene.sensors += (nonfoot,)
    cfg.terminations["low_height"] = TerminationTermCfg(func=fallen)
    cfg.terminations["illegal_ground_contact"] = TerminationTermCfg(func=illegal_ground_contact)
    if settings.get("standing_alive_rate", 0.0):
        cfg.rewards["alive"] = RewardTermCfg(func=alive, weight=settings["standing_alive_rate"])
    cfg.rewards["action_rate_l2"].weight *= settings.get("action_smoothing_scale", 1.0)
    if settings.get("standing_pose_std"):
        cfg.rewards["pose"].params["std_standing"] = {".*": settings["standing_pose_std"]}
    if settings.get("linear_tracking_std"):
        cfg.rewards["track_linear_velocity"].params["std"] = settings["linear_tracking_std"]
    if settings.get("angular_tracking_std"):
        cfg.rewards["track_angular_velocity"].params["std"] = settings["angular_tracking_std"]
    if settings.get("root_angular_velocity_cost"):
        cfg.rewards["root_angular_velocity_l2"] = RewardTermCfg(func=root_angular_velocity_l2,
            weight=-settings["root_angular_velocity_cost"])
    cfg.rewards["track_linear_velocity"].weight *= settings.get("linear_tracking_weight_scale", 1.0)
    cfg.rewards["foot_slip"].weight *= settings.get("foot_slip_weight_scale", 1.0)
    cfg.rewards["foot_clearance"].weight *= settings.get("foot_clearance_weight_scale", 1.0)
    if settings.get("air_time_rate"):
        # mjlab 1.6.0 rewards current airborne intervals continuously, not landing.
        # RewardManager applies dt once. Do not divide this rate by dt.
        cfg.rewards["air_time"].weight = settings["air_time_rate"]
        cfg.rewards["air_time"].params["command_threshold"] = 0.05
    agent = unitree_g1_ppo_runner_cfg()
    agent.seed = settings["seed"]
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.actor.hidden_dims = tuple(settings["hidden_dims"])
    agent.critic.hidden_dims = tuple(settings["hidden_dims"])
    agent.actor.distribution_cfg["init_std"] = settings["initial_std"]
    agent.num_steps_per_env = settings["rollout_steps"]
    agent.max_iterations = settings["iterations"]
    agent.save_interval = settings["save_interval"]
    agent.clip_actions = 1.0
    agent.algorithm.entropy_coef = settings.get("entropy_coef", agent.algorithm.entropy_coef)
    return cfg, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    directory = Path(os.environ["G1_RUN_DIR"])
    cfg, agent = build_configs(settings)
    torch.set_num_threads(4)
    env = BaselineEnv(cfg, directory=directory, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent_dict = asdict(agent)
    dump_yaml(directory / "params/env.yaml", asdict(cfg))
    dump_yaml(directory / "params/agent.yaml", agent_dict)
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, str(directory / "ppo"), device="cuda:0")
    if args.resume:
        runner.load(str(args.resume))
    # WSL wall clock has jumped during training; patch only the runner module timer.
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.time = SimpleNamespace(time=time.monotonic)
    start_monotonic = time.monotonic()
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
    checkpoints = sorted((directory / "ppo").glob("*.pt"))
    manifest = {"task": settings["task"], "seed": settings["seed"], "iterations": agent.max_iterations,
        "transitions_this_run": settings["num_envs"] * settings["rollout_steps"] * agent.max_iterations,
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        "checkpoint_sha256": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(), "jump_task_complete": False,
        "elapsed_seconds_monotonic": time.monotonic()-start_monotonic,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")}}
    (directory / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    wrapped.close()
    print(json.dumps(manifest), flush=True)

if __name__ == "__main__":
    main()
