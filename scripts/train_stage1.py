"""Bounded PPO training on a genuine static gap with current-state waypoint command."""
import argparse
from dataclasses import asdict
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from train_baseline import BaselineEnv,build_configs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from g1_dynamicparkour.scene import Geometry
from g1_dynamicparkour.platform_control import attach_platform_scene
from g1_dynamicparkour.waypoint import attach_waypoint_command
from g1_dynamicparkour import stage1

class Stage1Env(BaselineEnv):
    def __init__(self,cfg,directory,geometry,**kwargs):
        self.geometry = geometry
        super().__init__(cfg,directory,**kwargs)

    def episode_details(self, env_id):
        s = stage1.state(self)
        return {"task":"stage1_static_gap_stepping_not_jump",
            "success":bool(s.success[env_id]),"target_contact":bool(s.reached_target[env_id]),
            "stable_seconds":float(s.stable_time[env_id]),"distance_to_goal_m":float(s.distance[env_id]),
            "peak_total_target_force_n":float(s.peak_target_force[env_id]),
            "jump_events":"不适用：Stage1允许跨步，不能称跳跃"}

def build_stage1_configs(settings):
    cfg,agent = build_configs(settings)
    geometry = Geometry(gap=settings.get("gap",.02),amplitude=0)
    cfg = attach_waypoint_command(attach_platform_scene(cfg,geometry),settings.get("max_speed",.25),settings.get("stop_on_arrival",False))
    cfg.events["reset_task_state"] = EventTermCfg(func=stage1.reset_state,mode="reset")
    # Update signals once before all termination/reward reads, at fixed 20ms.
    cfg.terminations = {"update_task_state":TerminationTermCfg(func=stage1.update_state),**cfg.terminations}
    if settings.get("terminate_on_success", False):
        cfg.terminations["task_success"] = TerminationTermCfg(func=stage1.task_success)
    for name,func in (("foot_air_time",stage1.feet_air_time),("foot_contact",stage1.feet_contact),
                      ("foot_contact_forces",stage1.feet_forces)):
        cfg.observations["critic"].terms[name].func = func
        cfg.observations["critic"].terms[name].params = {}
    cfg.rewards["air_time"] = RewardTermCfg(func=stage1.air_time_reward,weight=.5)
    cfg.rewards["foot_slip"] = RewardTermCfg(func=stage1.foot_slip_cost,weight=-10)
    # The original single-ground landing/air-time rewards cannot describe two platforms.
    cfg.rewards.pop("foot_swing_height")
    cfg.rewards["soft_landing"] = RewardTermCfg(func=stage1.impact_cost,weight=-.05)
    cfg.rewards["foot_clearance"].params["target_height"] = .05
    dt = cfg.sim.mujoco.timestep*cfg.decimation
    cfg.rewards.update({
        "task_progress":RewardTermCfg(func=stage1.progress,weight=2),
        "goal_proximity":RewardTermCfg(func=stage1.goal_proximity,weight=2),
        "stop_velocity":RewardTermCfg(func=stage1.stop_velocity_cost,weight=-5),
        "target_stable":RewardTermCfg(func=stage1.stable_reward,weight=3),
        # Contact and success have separate lifetime latches; each pulse occurs once.
        "target_contact_event":RewardTermCfg(func=stage1.target_contact_event,weight=1/dt),
        "success_event":RewardTermCfg(func=stage1.success_event,weight=5/dt)})
    return cfg,agent,geometry

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--resume",type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    directory=Path(os.environ["G1_RUN_DIR"])
    cfg,agent,geometry=build_stage1_configs(settings)
    torch.set_num_threads(4)
    env=Stage1Env(cfg,directory,geometry,device="cuda:0")
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    dump_yaml(directory/"params/env.yaml",asdict(cfg))
    dump_yaml(directory/"params/agent.yaml",asdict(agent))
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),str(directory/"ppo"),device="cuda:0")
    runner.load(str(args.resume))
    import rsl_rl.runners.on_policy_runner as module
    module.time=SimpleNamespace(time=time.monotonic)
    begin=time.monotonic()
    runner.learn(num_learning_iterations=agent.max_iterations,init_at_random_ep_len=False)
    s = stage1.state(env)
    active = [{"env_id":i,"partial_episode":True,"control_steps":int(env.episode_length_buf[i]),
        "current_reward_terms":{name:float(value[i]) for name,value in env.reward_manager._episode_sums.items()},
        "target_contact":bool(s.reached_target[i]),"current_stable_seconds":float(s.stable_time[i]),
        "success_event_claimed":bool(s.success_claimed[i]),"current_goal_distance_m":float(s.distance[i])}
        for i in range(env.num_envs)]
    (directory/"active_episode_snapshot.json").write_text(json.dumps(active,indent=2))
    result={"task":"stage1_static_gap_stepping_not_jump","seed":settings["seed"],
        "geometry":asdict(geometry),"iterations":agent.max_iterations,
        "transitions_this_run":settings["num_envs"]*settings["rollout_steps"]*agent.max_iterations,
        "resume_checkpoint":str(args.resume),"resume_sha256":hashlib.sha256(args.resume.read_bytes()).hexdigest(),
        "checkpoint_sha256":{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/"ppo").glob("*.pt")},
        "elapsed_seconds_monotonic":time.monotonic()-begin,"jump_task_complete":False,
        "peak_torch_reserved_bytes":torch.cuda.max_memory_reserved(),
        "runtime_versions":{name:importlib.metadata.version(name) for name in ("torch","mjlab","mujoco","mujoco-warp","warp-lang","rsl-rl-lib")}}
    (directory/"training_manifest.json").write_text(json.dumps(result,indent=2))
    wrapped.close()
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()