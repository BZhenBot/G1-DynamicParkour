"""Frozen PPO jump evaluation using the scalar physical detector, independent of training FSM/reward."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import random
from pathlib import Path
import time

import torch
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_jump import JumpEnv,build_jump_configs,configure_runner_model
from g1_dynamicparkour.events import JumpEventDetector,Sample,Phase,TIME_COMPARISON_EPS_S


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=3)
    parser.add_argument("--seed",type=int,default=12000)
    parser.add_argument("--seconds",type=float,default=12)
    parser.add_argument("--video-episodes",type=int,default=0)
    parser.add_argument("--phase-bins",type=int,default=0,help="Explicit stratified fixed bin centers across episodes; reset only, no phase/future actor input")
    parser.add_argument("--phase-jitter",action="store_true",help="Uniform within each declared phase bin; separate seeded reset RNG, never an actor input")
    parser.add_argument("--time-budget-diagnostic",action="store_true",help="Explicitly separate a longer recovery horizon from the standard12s gate")
    parser.add_argument("--continue-after-success",action="store_true",help="Diagnostic genuine continuous replay after first success; requires --time-budget-diagnostic")
    parser.add_argument("--public-observation-handoff",action="store_true",help="Explicit current-observation supervisor plus two real PPO policies; actor120, no privileged switch")
    parser.add_argument("--recovery-config",type=Path)
    parser.add_argument("--recovery-checkpoint",type=Path)
    args=parser.parse_args()
    if bool(args.recovery_config)!=bool(args.recovery_checkpoint):
        parser.error("Composed controller requires both recovery config and checkpoint")
    if args.recovery_checkpoint:
        args.public_observation_handoff=True
    if args.episodes<1 or args.seconds<=0:
        parser.error("episodes and seconds must be positive")
    if args.continue_after_success and not args.time_budget_diagnostic:
        parser.error("Long replay is diagnostic only; retain the standard12s evaluation scope")
    directory=Path(os.environ["G1_RUN_DIR"])
    settings=json.loads(args.config.read_text())
    settings["num_envs"]=1
    cfg,agent,geometry=build_jump_configs(settings)
    if args.public_observation_handoff:
        if not settings.get("actor_contact_observations") or settings.get("action_basis")!="recovery_coordinate":
            parser.error("Public supervisor requires trained114/30 recovery-coordinate model")
        from g1_dynamicparkour.public_handoff import attach_public_supervisor_observations
        attach_public_supervisor_observations(cfg)
    if args.phase_bins<0 or (args.phase_bins and geometry.amplitude<=0) or (args.phase_jitter and not args.phase_bins):
        parser.error("Phase bins require a dynamic platform and nonnegative count")
    source_control_dt=cfg.sim.mujoco.timestep*cfg.decimation
    recovery_settings=None
    if args.recovery_checkpoint:
        recovery_settings=json.loads(args.recovery_config.read_text())
        cfg.decimation=recovery_settings["control_decimation"]
        for name,bonus in settings.get("event_bonuses",{}).items():
            cfg.rewards[name].weight=bonus/(cfg.sim.mujoco.timestep*cfg.decimation)
    cfg.seed,cfg.auto_reset,cfg.episode_length_s=args.seed,False,args.seconds+1
    # Terminate for physical falls/nonfoot contacts, but let the independent scalar
    # detector decide jump success/failure. Training FSM remains diagnostic only.
    cfg.terminations.pop("jump_success")
    cfg.terminations.pop("jump_failure")
    for group in cfg.observations.values():
        group.enable_corruption=False
    cfg.events["joint_noise"]=EventTermCfg(func=reset_joints_by_offset,mode="reset",
        params={"position_range":(-.01,.01),"velocity_range":(0,0),
                "asset_cfg":SceneEntityCfg("robot",joint_names=(".*",))})
    cfg.viewer.width,cfg.viewer.height=640,480
    cfg.viewer.distance,cfg.viewer.azimuth,cfg.viewer.elevation=3.5,90,-25
    manifest={"task":"extended_time_budget_physical_jump_diagnostic_not_standard12s" if args.time_budget_diagnostic else "independent_scalar_physical_jump_evaluation","geometry":asdict(geometry),
        "platform_reset_phase_mode":"stratified_uniform_within_bins" if args.phase_jitter else "stratified_bin_centers" if args.phase_bins else "uniform_random" if settings.get("platform_random_phase",False) else "fixed",
        "platform_phase_bins":args.phase_bins,
        "platform_configured_phase_rad":cfg.actions["periodic_platform"].phase,
        "platform_phase_scope":"reset/simulator/critic and diagnostic logs only; not an actor observation or future trajectory",
        "time_budget_diagnostic":args.time_budget_diagnostic,
        "continue_after_success":args.continue_after_success,
        "post_success_replay_scope":"Real continuous state-feedback control; later physical fall/illegal contact invalidates this diagnostic episode; original first-success event is retained",
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config":str(args.config),"episodes":args.episodes,"first_seed":args.seed,"maximum_seconds":args.seconds,
        "actor":"114: original106 + current unlabelled foot-contact2/netforce6; no future/FSM or surface identity" if settings.get("actor_contact_observations") else "106: original99 + current relative platform position/velocity6 + known clock1; no future/FSM",
        "controller":"frozen_real_jump_prior_plus_ppo_sampled_recovery_coordinate_and_joint_residual" if settings.get("action_basis")=="recovery_coordinate" else "frozen_real_jump_prior_plus_frozen_direct_ppo_residual" if settings.get("jump_prior_checkpoint") else "frozen_real_standing_prior_plus_frozen_ppo_residual" if settings.get("standing_prior_checkpoint") else "single_frozen_ppo_policy","action_basis":settings.get("action_basis","direct"),
        "criterion":"source dual support >=.1s, no feet support >=.08s with vz>.1 and rise>=.03; actual target contact without source support; both feet inside target2cm, foot heights[-.01,.04], tilt<=15deg, root relative vxy<=.15/vz<=.1 and omega<=.5 for continuous2s; no illegal contact/fall",
        "detector_time_comparison_epsilon_s":TIME_COMPARISON_EPS_S,
        "sensor_contact_lag_s":cfg.sim.mujoco.timestep,"physics_dt_s":cfg.sim.mujoco.timestep,"control_dt_s":cfg.sim.mujoco.timestep*cfg.decimation,
        "contact_note":"mjlab contact buffers refer to last physics evaluation before integration; pose is post-step, lag retained explicitly",
        "external_root_force":False,"robot_state_overwrite_during_episode":False,
        "detector":"scalar JumpEventDetector; no training reward or tensor FSM used for verdict",
        "policy_action_dimension":30 if settings.get("action_basis")=="recovery_coordinate" else 29,"physical_joint_dimension":29,
        "actor_contact_observations":settings.get("actor_contact_observations",False),
        "bounded_coordinate_mean":settings.get("bounded_coordinate_mean",False),
        "contact_observation_scope":"Exact current MuJoCo sensor sum; equivalent hardware sensor/estimator is unverified; no surface identity/FSM/future",
        "checkpoint_is_calibration":("calibration" in args.checkpoint.name or "zero_residual" in args.checkpoint.name),
        "learned_recovery_blend":settings.get("learned_recovery_blend",False),
        "blend_information":"Only configured current actor observations; no contact FSM or privileged state"}
    if args.public_observation_handoff:
        manifest.update({
            "task":"independent_public_observation_hierarchical_jump_evaluation" if not args.time_budget_diagnostic else "extended_public_observation_hierarchical_diagnostic_not_standard12s",
            "controller":"two_genuine_frozen_PPO_policies_with_explicit_causal_current_observation_handoff",
            "effective_actor_observation_dimension":120,"jump_network_observation_dimension":114,"standing_network_observation_dimension":99,
            "supervisor_observations":"Current world-axis FK foot offsets relative to current platform top6 plus original actor114; no surface ID, privileged FSM/verdict, critic or future",
            "supervisor_latch":"One bit resets per episode; current contact+inside/top+gravity+angular speed+relative velocity+height+elapsed clock",
            "switch_is_learned":False,"hardware_equivalent_measurements":"unverified",
            "handoff_thresholds":dict({"tilt_deg":15.,"angular_speed":.5,"horizontal_speed":.15,"vertical_speed":.1},**settings.get("public_handoff_thresholds",{})),
            "switch_thresholds_are_success_criteria":False,
            "handoff_transition_seconds":settings.get("public_handoff_transition_seconds",0.),
            "blend_semantics":"Linear mix of current actual bounded29 joint commands of two real state-feedback PPO policies, not an open-loop joint trajectory",
        })
    if args.recovery_checkpoint:
        manifest.update({
            "task":"independent_dual_rate_composed_jump_recovery_evaluation" if not args.time_budget_diagnostic else "extended_dual_rate_composed_recovery_diagnostic_not_standard12s",
            "controller":"Two real PPO policies, causal current geometry handoff; original jump25Hz held actual29 commands, dedicated recovery50Hz",
            "recovery_config":str(args.recovery_config),"recovery_checkpoint":str(args.recovery_checkpoint),
            "recovery_checkpoint_sha256":hashlib.sha256(args.recovery_checkpoint.read_bytes()).hexdigest(),
            "jump_policy_period_s":source_control_dt,"recovery_policy_period_s":cfg.sim.mujoco.timestep*cfg.decimation,
            "handoff_rule":"Current both any-contact, feet inside2cm and on-top[-.01,.04], tilt<=45deg, relative height>=.5, known normalized clock>=.8; no speed gate, no source/target identity, no detector",
            "recovery_network_observation_dimension":114,
            "handoff_thresholds":{"tilt_deg":45.,"minimum_clock":.8,"speed_gate":False},
            "supervisor_latch":"One bit resets per episode; current contact, world-axis FK bounds, gravity, relative height and clock; no speed gate",
            "blend_semantics":"Hard switch of current feedback commands; jump actual29 commands held for two manager steps; no open-loop joint trajectory",
            "handoff_transition_seconds":0.,
            "handoff_thresholds_are_success_criteria":False,"switch_is_learned":False,"subpolicy_probability_dimensions":[30,29],
            "composed_actions_have_single_gaussian_density":False,"robot_state_writes_at_episode_reset":"original source only; no landing resets",
        })
    (directory/"evaluation_config.json").write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4)
    env=JumpEnv(cfg,directory,geometry,device="cuda:0",render_mode="rgb_array" if args.video_episodes else None)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    configure_runner_model(agent,env)
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),log_dir=None,device="cuda:0")
    from g1_dynamicparkour.recovery_coordinate import bind_recovery_coordinate
    bind_recovery_coordinate(env,runner.alg.actor)
    runner.load(str(args.checkpoint))
    policy=runner.get_inference_policy(device="cuda:0")
    supervisor=None
    if args.public_observation_handoff:
        from g1_dynamicparkour.public_handoff import PublicObservationHandoff
        if args.recovery_checkpoint:
            from train_recovery_standing import build_recovery_standing_configs
            from g1_dynamicparkour.recovery_standing_policy import RecoveryStandingMLP
            from g1_dynamicparkour.composed_controller import DualRateJumpRecovery
            recovery_cfg,recovery_agent,recovery_geometry=build_recovery_standing_configs(recovery_settings)
            if asdict(recovery_geometry)!=asdict(geometry):
                raise ValueError("Composed policies require identical declared scene geometry and drive gains")
            configure_runner_model(recovery_agent,env)
            recovery_params=asdict(recovery_agent.actor)
            recovery_params.pop("class_name")
            current_obs=wrapped.get_observations()
            recovery_actor=RecoveryStandingMLP(current_obs,{"actor":["actor"]},"actor",29,**recovery_params).to(env.device)
            recovery_actor.load_state_dict(torch.load(args.recovery_checkpoint,map_location=env.device,weights_only=False)["actor_state_dict"],strict=True)
            recovery_actor.eval()
            if not torch.equal(recovery_actor.physical_units,runner.alg.actor.jump_prior.physical_units):
                raise ValueError("Subpolicies must agree on physical joint units")
            supervisor=DualRateJumpRecovery(policy,runner.alg.actor,recovery_actor,geometry,env.num_envs,env.device,env.step_dt,source_control_dt)
        else:
            supervisor=PublicObservationHandoff(policy,runner.alg.actor,geometry,env.num_envs,env.device,settings.get("public_handoff_thresholds"),settings.get("public_handoff_transition_seconds",0.),env.step_dt)
    robot,target=env.scene["robot"],env.scene["target"]
    foot_ids=robot.find_sites(("left_foot","right_foot"))[0]
    bounds=torch.tensor([geometry.target_half_length-.02,geometry.target_half_width-.02],device=env.device)
    def contact(name):
        data=env.scene[name].data
        force=torch.linalg.vector_norm(data.force[0],dim=-1)
        return ((data.found[0]>0)&(force>5)).cpu().tolist(),force.cpu().tolist()
    def vec(x):
        return x.detach().cpu().tolist()
    manifest["robot_joint_names"]=list(robot.joint_names)
    manifest["robot_actuator_names"]=list(robot.actuator_names)
    (directory/"evaluation_config.json").write_text(json.dumps(manifest,indent=2))
    rows=[]
    render_stride=max(1,round(1/(25*env.step_dt)))
    begin=time.monotonic()
    with torch.no_grad(),(directory/"samples.jsonl").open("a") as samples:
        for episode in range(args.episodes):
            wrapped.seed(args.seed+episode)
            periodic=env.action_manager.get_term("periodic_platform")
            phase_bin=episode%args.phase_bins if args.phase_bins else None
            if phase_bin is not None:
                periodic.cfg.random_phase=False
                offset=random.Random(args.seed+episode).random() if args.phase_jitter else .5
                periodic.cfg.phase=2*math.pi*(phase_bin+offset)/args.phase_bins
            obs,_=wrapped.reset()
            # Logger flushes the previous episode during reset, so label afterward.
            env.evaluation_episode_index=episode
            env.evaluation_seed=args.seed+episode
            initial_phase_rad=float(periodic.phase[0])
            detector=JumpEventDetector()
            if supervisor is not None:
                supervisor.reset()
            frames,reasons=[],[]
            first_success=None
            post_success_samples=post_success_stable_samples=0
            landing_velocity=None
            peak_force,force_integral,max_tilt=0.,0.,0.
            for step in range(round(args.seconds/env.step_dt)):
                contact_input=obs["actor"][0,106:].clone() if settings.get("actor_contact_observations") else None
                action=supervisor(obs) if supervisor is not None else policy(obs)
                obs,reward,done,extras=wrapped.step(action)
                t=(step+1)*env.step_dt
                start_support,start_force=contact("feet_ground_contact")
                target_support,target_force=contact("feet_target_contact")
                target_xy=target.data.root_link_pos_w[0,:2].clone()
                target_xy[1]+=target.data.joint_pos[0,0]
                target_v=torch.tensor([0.,float(target.data.joint_vel[0,0]),0.],device=env.device)
                relative_v=robot.data.root_link_lin_vel_w[0]-target_v
                foot_pos=robot.data.site_pos_w[0,foot_ids]
                inside=bool(((foot_pos[:,:2]-target_xy).abs()<=bounds).all())
                foot_height=foot_pos[:,2]-(target.data.root_link_pos_w[0,2]+geometry.half_thickness)
                on_top=bool(((foot_height>=-.01)&(foot_height<=.04)).all())
                tilt=math.degrees(math.acos(float((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
                omega=float(torch.linalg.vector_norm(robot.data.root_link_ang_vel_w[0]))
                height=float(robot.data.root_link_pos_w[0,2])
                illegal=any(bool((torch.linalg.vector_norm(env.scene[name].data.force[0],dim=-1)>10).any()) for name in ("nonfoot_ground","nonfoot_target"))
                fallen=height<.5 or tilt>70
                stable=all(target_support) and not any(start_support) and inside and on_top and tilt<=15 and float(torch.linalg.vector_norm(relative_v[:2]))<=.15 and abs(float(relative_v[2]))<=.1 and omega<=.5 and not illegal and not fallen
                sample=Sample(t,tuple(start_support),tuple(target_support),height,float(robot.data.root_link_lin_vel_w[0,2]),stable,illegal,fallen)
                changes=detector.update(sample)
                if detector.phase==Phase.STABLE_SUCCESS:
                    if first_success is None:
                        first_success=t
                    post_success_samples+=1
                    post_success_stable_samples+=int(stable)
                if Phase.TARGET_CONTACT.value in changes:
                    landing_velocity=vec(relative_v)
                peak_force=max(peak_force,sum(target_force))
                force_integral+=sum(target_force)*env.step_dt
                max_tilt=max(max_tilt,tilt)
                samples.write(json.dumps({"episode":episode,"seed":args.seed+episode,"time_s":t,
                    "platform_clock_s":float(periodic.t[0]),"platform_phase_rad":float(periodic.phase[0]),
                    "platform_phase_bin":phase_bin,
                    "contact_buffer_time_s":t-cfg.sim.mujoco.timestep,"action":vec(action[0]),
                    "applied_actor_coordinates":vec(env.action_manager.action[0]),
                    "learned_recovery_blend_weight":float(runner.alg.actor.last_recovery_blend[0,0]) if settings.get("learned_recovery_blend") else None,
                    "sampled_recovery_coordinate":float(env.action_manager.get_term("joint_pos").last_recovery_coordinate[0,0]) if settings.get("action_basis")=="recovery_coordinate" else None,
                    "actor_contact_input_before_step":vec(contact_input) if contact_input is not None else None,
                    "public_actor120_before_step":vec(supervisor.last_public_observation[0]) if supervisor is not None else None,
                    "public_handoff_ready_before_step":bool(supervisor.last_ready[0]) if supervisor is not None else None,
                    "public_handoff_applied_before_step":bool(supervisor.active[0]) if supervisor is not None else None,
                    "public_handoff_new_before_step":bool(supervisor.last_new_handoff[0]) if supervisor is not None else None,
                    "public_handoff_alpha_before_step":float(supervisor.alpha[0]) if supervisor is not None else None,
                    "source_policy_updated_before_step":bool(supervisor.last_source_updated[0]) if args.recovery_checkpoint else None,
                    "joint_normalized_action":vec(env.action_manager.get_term("joint_pos").raw_action[0]),
                    "joint_position_target":vec(env.action_manager.get_term("joint_pos")._processed_actions[0]),
                    "root_position":vec(robot.data.root_link_pos_w[0]),"root_velocity":vec(robot.data.root_link_lin_vel_w[0]),
                    "root_quaternion_wxyz":vec(robot.data.root_link_quat_w[0]),"root_heading_rad":float(robot.data.heading_w[0]),
                    "measured_joint_position":vec(robot.data.joint_pos[0]),"measured_joint_velocity":vec(robot.data.joint_vel[0]),
                    "measured_actuator_force":vec(robot.data.actuator_force[0]),
                    "target_xy":vec(target_xy),"relative_velocity":vec(relative_v),
                    "feet_position":vec(foot_pos),"foot_height_m":vec(foot_height),
                    "start_support":start_support,"target_support":target_support,"start_force_n":start_force,"target_force_n":target_force,
                    "tilt_deg":tilt,"angular_speed_radps":omega,"stable":stable,"illegal_contact":illegal,"fallen":fallen,
                    "phase":detector.phase.value,"events":changes,"failure_reason":detector.failure_reason})+"\n")
                if episode<args.video_episodes and step%render_stride==0:
                    frames.append(env.render())
                if bool(done[0]):
                    reasons=[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])]
                if reasons or fallen or illegal or detector.phase==Phase.FAILED or (detector.phase==Phase.STABLE_SUCCESS and not args.continue_after_success):
                    break
            success=detector.phase==Phase.STABLE_SUCCESS and not reasons and not fallen and not illegal
            physical_failed=detector.phase==Phase.FAILED or bool(reasons) or fallen or illegal
            row={"episode":episode,"seed":args.seed+episode,"success":success,
                "outcome":"success" if success else "physical_failure" if physical_failed else "time_limit_without_success",
                "duration_s":t,"phase":detector.phase.value,"events":detector.events,"failure_reason":detector.failure_reason,
                "termination_reasons":reasons,"fell":fallen,"illegal_contact":illegal,"maximum_tilt_deg":max_tilt,
                "landing_relative_velocity_mps":landing_velocity,"peak_target_force_n":peak_force,
                "sampled_target_force_integral_ns":force_integral,"force_integral_note":f"netforce magnitudes integrated at actual control dt={env.step_dt:g}s; diagnostic proxy, not full contact impulse",
                "final_root_position":vec(robot.data.root_link_pos_w[0]),
                "reward_terms":{name:float(value[0]) for name,value in env.reward_manager._episode_sums.items()}}
            row['platform_initial_phase_rad']=initial_phase_rad
            row['platform_phase_bin']=phase_bin
            row['first_success_time_s']=first_success
            row['continued_after_success']=args.continue_after_success
            if args.continue_after_success:
                row['post_success_samples']=post_success_samples
                row['post_success_stable_samples']=post_success_stable_samples
                row['post_success_stable_sample_fraction']=post_success_stable_samples/post_success_samples if post_success_samples else None
                row['post_success_physical_failure']=first_success is not None and physical_failed
            if frames:
                import numpy as np
                import mediapy
                import imageio_ffmpeg
                mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
                filename=directory/f"jump_episode_{episode:03d}.mp4"
                mediapy.write_video(str(filename),np.stack(frames),fps=1/(render_stride*env.step_dt))
                row["video"],row["video_sha256"]=filename.name,hashlib.sha256(filename.read_bytes()).hexdigest()
            rows.append(row)
            print(json.dumps(row),flush=True)
    result={"task":manifest["task"],"episodes":rows,"success_rate":sum(row["success"] for row in rows)/len(rows),
        "fall_rate":sum(row["fell"] for row in rows)/len(rows),"elapsed_seconds_monotonic":time.monotonic()-begin}
    (directory/"evaluation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!="episodes"}),flush=True)
    wrapped.close()

if __name__=="__main__":
    main()
