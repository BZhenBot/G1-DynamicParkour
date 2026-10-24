"""PPO entry for physical single-gap jumps; inspect/validate config before any training."""
import argparse
from dataclasses import asdict,fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from train_baseline import build_configs
from train_stage1 import Stage1Env
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from g1_dynamicparkour.scene import Geometry
from g1_dynamicparkour.platform_control import attach_platform_scene
from g1_dynamicparkour import stage1,jump,action_basis
from g1_dynamicparkour.residual_policy import ResidualStandingModelCfg,configure_runner_model
from g1_dynamicparkour.jump_prior_policy import ResidualJumpModelCfg
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from g1_dynamicparkour.batched_events import PHASES,FAILURES
from g1_dynamicparkour.checkpoint_transfer import expand_model_state

class JumpEnv(Stage1Env):
    def episode_details(self,env_id):
        details=super().episode_details(env_id)
        s=jump.state(self)
        details.update({"task":"single_gap_physical_jump","success":bool(s.detector.phase[env_id]==5),
            "jump_events":{"takeoff":bool(s.event_counts[env_id,2]>0),
                "airborne":bool(s.event_counts[env_id,3]>0),
                "target_contact_after_airborne":bool(s.event_counts[env_id,4]>0),
                "stable_success":bool(s.event_counts[env_id,5]>0)},
            "jump_phase":PHASES[int(s.detector.phase[env_id])],
            "jump_failure":FAILURES[int(s.detector.failure[env_id])]})
        return details

def build_jump_configs(settings):
    base=dict(settings,task="standing",velocity_x=[0,0],world_forward_heading_hold=False)
    cfg,agent=build_configs(base)
    cfg.decimation=settings.get("control_decimation",cfg.decimation)
    if cfg.decimation not in (4,8):
        raise ValueError("Allowed candidate control decimation4/8; original physics dt retained; validate before training")
    geometry=Geometry(gap=settings.get("gap",.02),robot_start_offset_x=settings.get("robot_start_offset",.2),
        amplitude=settings.get("amplitude",0),frequency=settings.get("frequency",.1),
        platform_servo_kp=settings.get("platform_servo_kp",100000.),
        platform_servo_kv=settings.get("platform_servo_kv",4000.))
    cfg=attach_platform_scene(cfg,geometry)
    periodic=cfg.actions["periodic_platform"]
    periodic.random_phase=bool(settings.get("platform_random_phase",False))
    periodic.phase=float(settings.get("platform_phase_rad",periodic.phase))
    if not math.isfinite(periodic.phase):
        raise ValueError("Platform reset phase must be finite")
    multiplier=settings.get("leg_action_scale_multiplier",1.)
    if not 1<=multiplier<=2:
        raise ValueError("Validated leg target range multiplier must be in [1,2]; physical torque limits remain unchanged")
    if settings.get("action_basis","direct") not in ("direct","bilateral_sagittal","recovery_coordinate"):
        raise ValueError("Unsupported action basis")
    if settings.get("action_basis")=="bilateral_sagittal":
        original=cfg.actions["joint_pos"]
        cfg.actions["joint_pos"]=action_basis.BilateralJointPositionActionCfg(**{field.name:getattr(original,field.name) for field in fields(JointPositionActionCfg)})
    if settings.get("action_basis")=="recovery_coordinate":
        original=cfg.actions["joint_pos"]
        cfg.actions["joint_pos"]=action_basis.RecoveryCoordinateActionCfg(**{field.name:getattr(original,field.name) for field in fields(JointPositionActionCfg)})
    if multiplier!=1:
        scope=settings.get("leg_scale_scope","all")
        if scope not in ("all","sagittal"):
            raise ValueError("Action-range scope must be all or sagittal")
        words=("hip","knee","ankle") if scope=="all" else ("hip_pitch","knee","ankle_pitch")
        cfg.actions["joint_pos"].scale={pattern:value*(multiplier if any(word in pattern for word in words) else 1.) for pattern,value in cfg.actions["joint_pos"].scale.items()}
        for group in ("actor","critic"):
            cfg.observations[group].terms["actions"].func=jump.scaled_previous_actions
        cfg.rewards["action_rate_l2"].func=jump.scaled_action_rate
    if settings.get("action_basis")=="bilateral_sagittal":
        for group in ("actor","critic"):
            cfg.observations[group].terms["actions"].func=jump.scaled_previous_actions
        cfg.rewards["action_rate_l2"].func=jump.scaled_action_rate
    if settings.get("action_basis")=="recovery_coordinate":
        for group in ("actor","critic"):
            cfg.observations[group].terms["actions"].func=jump.recovery_previous_actions
        cfg.rewards["action_rate_l2"].func=jump.recovery_action_rate
    cfg.events["reset_platform_state"]=EventTermCfg(func=stage1.reset_state,mode="reset")
    cfg.events["reset_jump_state"]=EventTermCfg(func=jump.reset_state,mode="reset")
    cfg.terminations={"update_jump_state":TerminationTermCfg(func=jump.update_state),**cfg.terminations,
        "jump_failure":TerminationTermCfg(func=jump.failure),
        "jump_success":TerminationTermCfg(func=jump.success)}
    for name,func in (("foot_air_time",stage1.feet_air_time),("foot_contact",stage1.feet_contact),
                      ("foot_contact_forces",stage1.feet_forces)):
        cfg.observations["critic"].terms[name].func=func
        cfg.observations["critic"].terms[name].params={}
    for group in ("actor","critic"):
        cfg.observations[group].terms["platform_relative_state"]=ObservationTermCfg(func=jump.platform_relative_state)
        cfg.observations[group].terms["task_clock"]=ObservationTermCfg(func=jump.episode_clock,params={"duration":settings.get("clock_duration",.5)})
    if settings.get("actor_contact_observations",False):
        if settings.get("action_basis")!="recovery_coordinate":
            raise ValueError("Contact extension is gated for the106/30 recovery-coordinate architecture only")
        cfg.observations["actor"].terms["current_unlabelled_foot_contact"]=ObservationTermCfg(func=jump.actor_unlabelled_foot_contacts)
    # Contact identity and true FSM/servo state are explicit privileged critic inputs.
    cfg.observations["critic"].terms["jump_phase_privileged"]=ObservationTermCfg(func=jump.privileged_phase)
    cfg.observations["critic"].terms["motion_privileged"]=ObservationTermCfg(func=jump.privileged_geometry_motion)
    cfg.rewards["track_linear_velocity"].weight=0
    cfg.rewards["air_time"].weight=0
    cfg.rewards["foot_slip"]=RewardTermCfg(func=stage1.foot_slip_cost,weight=-2)
    cfg.rewards["soft_landing"]=RewardTermCfg(func=stage1.impact_cost,weight=-.05)
    cfg.rewards.pop("foot_swing_height")
    cfg.rewards["foot_clearance"].weight=0
    dt=cfg.sim.mujoco.timestep*cfg.decimation
    cfg.rewards.update({
        "prepare_height":RewardTermCfg(func=jump.preparation_height,weight=10),
        "propulsion":RewardTermCfg(func=jump.propulsion,weight=10),
        "flight_progress":RewardTermCfg(func=jump.flight_progress,weight=5),
        "land_position":RewardTermCfg(func=jump.landed_position,weight=2),
        "land_relative_velocity":RewardTermCfg(func=jump.landed_relative_velocity,weight=-2),
        "land_stable":RewardTermCfg(func=jump.target_stability,weight=5),
        "energy":RewardTermCfg(func=jump.energy,weight=-.00005),
        "airborne_event":RewardTermCfg(func=jump.airborne_event,weight=5/dt),
        "landing_event":RewardTermCfg(func=jump.target_event,weight=15/dt),
        "success_event":RewardTermCfg(func=jump.success_event,weight=50/dt),
        "failure_event":RewardTermCfg(func=jump.failure_event,weight=-10/dt)})
    if settings.get("progress_before_landing"):
        cfg.rewards["flight_progress"].func=jump.progress_before_landing
    for name,func in (("flight_foot_position_cost",jump.flight_foot_position_cost),("flight_tilt_cost",jump.flight_tilt_cost)):
        rate=settings.get(name,0)
        if rate<0:
            raise ValueError("Flight cost rates must be nonnegative magnitudes")
        if rate:
            params={"include_landing":settings.get("foot_cost_after_target_contact",False)} if name=="flight_foot_position_cost" else {"include_landing":settings.get("tilt_cost_after_target_contact",False)}
            if name=="flight_foot_position_cost" and "landing_foot_x_margin_m" in settings:
                margin=settings["landing_foot_x_margin_m"]
                if not math.isfinite(margin) or not .02 <= margin <= .10 or margin >= geometry.target_half_length:
                    raise ValueError("Invalid reward-only landing interior x margin")
                params["x_margin"]=margin
            cfg.rewards[name]=RewardTermCfg(func=func,weight=-rate,params=params)
    if settings.get("landing_height_cost",0):
        cfg.rewards["landing_height_deficit"]=RewardTermCfg(func=jump.landing_height_deficit_cost,
            params={"minimum_height":settings.get("landing_minimum_height",.68)},weight=-settings["landing_height_cost"])
    if settings.get("landed_dual_support_rate",0):
        cfg.rewards["landed_dual_support"]=RewardTermCfg(func=jump.landed_dual_support,weight=settings["landed_dual_support_rate"])
    if settings.get("propulsion_end_time") is not None:
        cfg.rewards["propulsion"].params["end_time"]=settings["propulsion_end_time"]
    if settings.get("source_wait_rate",0):
        cfg.rewards["source_wait"]=RewardTermCfg(func=jump.source_wait,weight=-settings["source_wait_rate"])
    for name,bonus in settings.get("event_bonuses",{}).items():
        if name not in ("airborne_event","landing_event","success_event","failure_event"):
            raise ValueError("Event overrides must name explicit physical pulses")
        cfg.rewards[name].weight=bonus/dt
    if settings.get("jump_prior_checkpoint"):
        if settings.get("action_basis") not in ("direct","recovery_coordinate") or settings.get("leg_scale_scope")!="sagittal" or multiplier!=2:
            raise ValueError("Exact jump prior transfer requires direct joint actions and unchanged sagittal units2")
        agent.actor=ResidualJumpModelCfg(**asdict(agent.actor),prior_checkpoint=settings["standing_prior_checkpoint"],
            prior_sha256=settings["standing_prior_sha256"],jump_checkpoint=settings["jump_prior_checkpoint"],
            jump_sha256=settings["jump_prior_sha256"],recovery_blend=settings.get("learned_recovery_blend",False))
        agent.actor.class_name="g1_dynamicparkour.jump_prior_policy:ResidualJumpMLP"
        if settings.get("action_basis")=="recovery_coordinate":
            from g1_dynamicparkour.recovery_coordinate import RecoveryCoordinateModelCfg
            agent.actor=RecoveryCoordinateModelCfg(**asdict(agent.actor),contact_extension=settings.get("actor_contact_observations",False),bounded_coordinate_mean=settings.get("bounded_coordinate_mean",False))
            agent.actor.class_name="g1_dynamicparkour.recovery_coordinate:RecoveryCoordinateMLP"
    elif settings.get("standing_prior_checkpoint"):
        if settings.get("action_basis")!="bilateral_sagittal" or settings.get("leg_scale_scope")!="sagittal" or multiplier!=2:
            raise ValueError("Exact bounded prior mapping requires bilateral basis and sagittal physical units2")
        original=agent.actor
        agent.actor=ResidualStandingModelCfg(**asdict(original),prior_checkpoint=settings["standing_prior_checkpoint"],prior_sha256=settings["standing_prior_sha256"])
        agent.actor.class_name="g1_dynamicparkour.residual_policy:ResidualStandingMLP"
    return cfg,agent,geometry

def load_warmstart(runner,path,output_multipliers=None,basis_matrix=None):
    old=torch.load(path,map_location="cpu",weights_only=False)
    report={}
    for role in ("actor","critic"):
        model=getattr(runner.alg,role)
        old_state=old[role+"_state_dict"]
        new_dim=model.state_dict()["mlp.0.weight"].shape[1]
        expanded=expand_model_state(old_state,new_dim)
        if role=="actor" and hasattr(model,"initialize_residual"):
            model.initialize_residual(old_state)
            report[role]={"old_input_dim":99,"new_input_dim":model.obs_dim,"mode":"frozen_clipped_standing_prior_plus_zero_residual"}
            continue
        if role=="actor" and output_multipliers is not None:
            units=output_multipliers.detach().cpu()
            expanded["mlp.4.weight"] /= units[:,None]
            expanded["mlp.4.bias"] /= units
            if basis_matrix is not None:
                inverse=torch.linalg.inv(basis_matrix.detach().cpu())
                expanded["mlp.4.weight"]=inverse@expanded["mlp.4.weight"]
                expanded["mlp.4.bias"]=inverse@expanded["mlp.4.bias"]
            # Keep parent learned std in normalized new actions: broader physical exploration, mean control preserved.
        model.load_state_dict(expanded,strict=True)
        report[role]={"old_input_dim":old_state["mlp.0.weight"].shape[1],"new_input_dim":new_dim}
    # Fresh optimizer and iteration count for a changed task/input shape.
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--warm-start",type=Path)
    group.add_argument("--resume",type=Path)
    group.add_argument("--fine-tune",type=Path)
    group.add_argument("--transfer-jump",type=Path)
    group.add_argument("--expand-contact",type=Path)
    group.add_argument("--bound-coordinate",type=Path)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    directory=Path(os.environ["G1_RUN_DIR"])
    cfg,agent,geometry=build_jump_configs(settings)
    torch.set_num_threads(4)
    env=JumpEnv(cfg,directory,geometry,device="cuda:0")
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    configure_runner_model(agent,env)
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),str(directory/"ppo"),device="cuda:0")
    from g1_dynamicparkour.recovery_coordinate import bind_recovery_coordinate
    bind_recovery_coordinate(env,runner.alg.actor)
    if args.warm_start:
        if settings.get("standing_prior_sha256") and hashlib.sha256(args.warm_start.read_bytes()).hexdigest()!=settings["standing_prior_sha256"]:
            raise ValueError("Residual warm-start must match the configured frozen standing source")
        transfer=load_warmstart(runner,args.warm_start,jump.action_units(env)[0],action_basis.matrix(env))
        parent=args.warm_start
    elif args.bound_coordinate:
        from g1_dynamicparkour.recovery_coordinate import load_bounded_coordinate_transfer
        transfer=load_bounded_coordinate_transfer(runner,args.bound_coordinate)
        parent=args.bound_coordinate
    elif args.expand_contact:
        from g1_dynamicparkour.recovery_coordinate import load_contact_expansion
        transfer=load_contact_expansion(runner,args.expand_contact)
        parent=args.expand_contact
    elif args.transfer_jump:
        if not settings.get("jump_prior_checkpoint") or hashlib.sha256(args.transfer_jump.read_bytes()).hexdigest()!=settings["jump_prior_sha256"]:
            raise ValueError("Jump transfer checkpoint must match the declared frozen jump prior")
        source=torch.load(args.transfer_jump,map_location=env.device,weights_only=False)
        runner.alg.critic.load_state_dict(source["critic_state_dict"],strict=True)
        transfer={"actor":"exact frozen previous PPO physical joint commands plus zero direct residual",
                  "critic":"previous130dim critic and normalization", "optimizer":"fresh for changed action distribution and new trainable residual", "iteration":"fresh0"}
        parent=args.transfer_jump
    elif args.fine_tune:
        if settings.get('finetune_parent_sha256') and hashlib.sha256(args.fine_tune.read_bytes()).hexdigest()!=settings['finetune_parent_sha256']:
            raise ValueError('Fine-tuning parent must match the predeclared checkpoint')
        source=torch.load(args.fine_tune,map_location=env.device,weights_only=False)
        loaded_iteration=runner.alg.load(source,{'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict=True)
        assert not loaded_iteration and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in source['actor_state_dict'].items())
        transfer={'actor':'exact shared pretrained actor and normalization','critic':'exact shared pretrained critic and normalization',
                  'optimizer':'fresh Adam state, not inherited','iteration':'fresh0','independence_scope':settings.get('finetune_scope','independent seeded dynamic recovery fine-tuning; shared standing/jump/static recovery pretraining and landing bank')}
        parent=args.fine_tune
    else:
        runner.load(str(args.resume))
        transfer=None
        parent=args.resume
    if (args.warm_start or args.transfer_jump) and settings.get("warmstart_std_by_joint"):
        import re
        std=runner.alg.actor.distribution.std_param
        with torch.no_grad():
            for i,name in enumerate(env.action_manager.get_term("joint_pos").target_names):
                matched=[value for pattern,value in settings["warmstart_std_by_joint"].items() if re.fullmatch(pattern,name)]
                if len(matched)>1:
                    raise ValueError("Exploration std patterns must be mutually exclusive")
                if matched:
                    std[i]=matched[0]
    if (args.warm_start or args.transfer_jump) and settings.get("action_basis")=="recovery_coordinate":
        recovery_std=settings["recovery_coordinate_std"]
        if not 0<recovery_std<=1:
            raise ValueError("Declared coordinated exploration std must be in (0,1]")
        with torch.no_grad():
            runner.alg.actor.distribution.std_param[29]=recovery_std
    std_scale=settings.get('finetune_std_scale',1.)
    if not 0<std_scale<=1:
        raise ValueError('Fine-tune initial std scale must be in (0,1]')
    if std_scale!=1:
        if not args.fine_tune or runner.alg.actor.distribution.std_type!='scalar':
            raise ValueError('Initial std scaling only supported for explicit fine-tune scalar Gaussian')
        before_std_scale={key:value.clone() for key,value in runner.alg.actor.state_dict().items()}
        distribution=runner.alg.actor.distribution
        with torch.no_grad():
            distribution.std_param.mul_(std_scale)
        changed=[key for key,value in before_std_scale.items() if not torch.equal(value,runner.alg.actor.state_dict()[key])]
        assert changed==['distribution.std_param'], ('Std-only candidate changed other actor state',changed)
        assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state
        transfer['initial_exploration']={'std_scale':std_scale,'only_changed_state_keys':changed,
            'parent_raw_std':before_std_scale['distribution.std_param'].detach().cpu().tolist(),
            'new_effective_std':distribution.std_param.clamp(*distribution.std_range).detach().cpu().tolist(),
            'deterministic_mean_preservation':'Std-only change; separately validated on actual recorded observations by CPU and physical gates'}
    actual_start_std=runner.alg.actor.state_dict()["distribution.std_param"].detach().cpu().tolist()
    dump_yaml(directory/"params/env.yaml",asdict(cfg))
    dump_yaml(directory/"params/agent.yaml",asdict(agent))
    import rsl_rl.runners.on_policy_runner as module
    module.time=SimpleNamespace(time=time.monotonic)
    frozen_prior=getattr(runner.alg.actor,"jump_prior",getattr(runner.alg.actor,"standing_prior",None))
    frozen_prior_before={key:value.clone() for key,value in frozen_prior.state_dict().items()} if frozen_prior is not None else None
    initial_trainable_actor={name:p.detach().clone() for name,p in runner.alg.actor.named_parameters() if p.requires_grad}
    start_iteration=runner.current_learning_iteration
    start=time.monotonic()
    runner.learn(num_learning_iterations=agent.max_iterations,init_at_random_ep_len=False)
    if frozen_prior_before is not None:
        assert all(torch.equal(value,frozen_prior.state_dict()[key]) for key,value in frozen_prior_before.items()),"Frozen real PPO prior changed during training"
    trainable_actor_delta_l2=sum(float((p.detach()-initial_trainable_actor[name]).square().sum()) for name,p in runner.alg.actor.named_parameters() if p.requires_grad)**.5
    if args.fine_tune:
        assert trainable_actor_delta_l2>0 and runner.alg.optimizer.state, 'Fine-tuning must perform actual PPO actor/optimizer updates'
    s=jump.state(env)
    partial=[{"env_id":i,"partial_episode":True,"control_steps":int(env.episode_length_buf[i]),
        "reward_terms":{name:float(value[i]) for name,value in env.reward_manager._episode_sums.items()},
        "phase":PHASES[int(s.detector.phase[i])],"event_counts":s.event_counts[i].tolist()}
        for i in range(env.num_envs)]
    (directory/"active_episode_snapshot.json").write_text(json.dumps(partial,indent=2))
    result={"task":settings.get("training_task_scope","real_landing_reset_recovery_only" if settings.get("recovery_only_training") else "physical_single_gap_jump"),"geometry":asdict(geometry),"seed":settings["seed"],
        "iterations":agent.max_iterations,"start_iteration":start_iteration,"end_iteration":runner.current_learning_iteration,"transitions_this_run":settings["num_envs"]*settings["rollout_steps"]*agent.max_iterations,
        "parent_checkpoint":str(parent),"parent_sha256":hashlib.sha256(parent.read_bytes()).hexdigest(),
        "warmstart_transfer":transfer,"actual_initial_action_std":actual_start_std,
        "trainable_actor_parameter_delta_l2":trainable_actor_delta_l2,
        "optimizer_state_entries_after_training":len(runner.alg.optimizer.state),
        "action_basis":settings.get("action_basis","direct"),"leg_scale_scope":settings.get("leg_scale_scope","all"),
        "policy_action_dimension":env.action_manager.total_action_dim,"physical_joint_dimension":29,
        "action_coordinate_joint_names":env.action_manager.get_term("joint_pos").target_names,
        "action_basis_matrix":action_basis.matrix(env).cpu().tolist(),
        "leg_action_scale_multiplier":settings.get("leg_action_scale_multiplier",1.),
        "physical_torque_limits":"unchanged original G1 actuator effort limits","elapsed_seconds_monotonic":time.monotonic()-start,
        "checkpoint_sha256":{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/"ppo").glob("*.pt")},
        "runtime_versions":{name:importlib.metadata.version(name) for name in ("torch","mjlab","mujoco","mujoco-warp","warp-lang","rsl-rl-lib")},
        "frozen_standing_prior_unchanged":True if frozen_prior_before is not None else None,
        "standing_prior_checkpoint":settings.get("standing_prior_checkpoint"),
        "standing_prior_sha256":settings.get("standing_prior_sha256"),
        "jump_prior_checkpoint":settings.get("jump_prior_checkpoint"),"jump_prior_sha256":settings.get("jump_prior_sha256"),
        "frozen_jump_prior_unchanged":True if settings.get("jump_prior_checkpoint") and frozen_prior_before is not None else None,
        "actor_contact_observations":settings.get("actor_contact_observations",False),
        "bounded_coordinate_mean":settings.get("bounded_coordinate_mean",False),
        "actor_observation_dimension":runner.alg.actor.obs_dim,
        "actor_contact_input_weight_norm":float(torch.linalg.vector_norm(runner.alg.actor.mlp[0].weight[:,106:])) if settings.get("actor_contact_observations",False) else None,
        "learned_recovery_blend":settings.get("learned_recovery_blend",False),
        "recovery_blend_parameter_norm":float(torch.linalg.vector_norm(torch.cat([value.detach().flatten() for value in runner.alg.actor.recovery_blend_head.parameters()]))) if getattr(runner.alg.actor,"recovery_blend_head",None) is not None else None,
        "task_success":"Only independent physical-event evaluation can establish success; no claim from training reward"}
    (directory/"training_manifest.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    wrapped.close()

if __name__=="__main__":
    main()
