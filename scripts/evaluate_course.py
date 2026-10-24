"""真实完整双跳评价；冻结两PPO，不用检测器结果控制技能。"""
import argparse,hashlib,json,math,os,random,time
from dataclasses import asdict
from pathlib import Path
import torch
from course_runtime import build_course,load_controllers
from course_contact_env import CourseContactEnv
from g1_dynamicparkour.course_scene import PlatformTargetProvider,contacts,SURFACES,SENSOR_SUFFIX
from g1_dynamicparkour.course_events import CourseEventDetector,CourseSample
from g1_dynamicparkour.course_time_budget import time_failure

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--episodes',type=int,default=1)
p.add_argument('--seed',type=int,default=51000);p.add_argument('--video-episodes',type=int,default=0)
p.add_argument('--max-steps',type=int,default=0,help='只用于接口闸门，不能作为正式评价')
a=p.parse_args();settings=json.loads(a.config.read_text());out=Path(os.environ['G1_RUN_DIR'])
assert a.episodes>0
from formal_course_protocol import validate_formal
formal_receipt=validate_formal(settings,a.episodes,a.seed,a.max_steps)
for role in ('jump','recovery'):
    assert hashlib.sha256(Path(settings[role+'_checkpoint']).read_bytes()).hexdigest()==settings[role+'_sha256']
torch.set_num_threads(4);cfg,agent,course,g=build_course(settings,a.seed)
env=CourseContactEnv(cfg,course,g,device='cuda:0',render_mode='rgb_array' if a.video_episodes else None)
w,runner,coordinator=load_controllers(env,agent,settings);provider=PlatformTargetProvider(env)
robot=env.scene['robot'];foot_ids=robot.find_sites(('left_foot','right_foot'))[0]
manifest={'task':'two_real_jumps_course' if not a.max_steps else 'course_interface_gate_not_acceptance','course':asdict(course),
    'settings':settings,'config':str(a.config),'seed':a.seed,'episodes':a.episodes,'physics_dt_s':env.physics_dt,'control_dt_s':env.step_dt,
    'actor114':'原99本体/实际动作+当前目标相对状态6+公开局部技能clock1+所有表面无身份接触8',
    'supervisor_extra6':'当前两足FK相对当前平台；没有privileged/FSM/surfaceID/future输入',
    'critic130':'部署不使用；第二跳/恢复定向训练的当前phase及几何/平台sin-cos特权另见各训练curriculum与source，不输入Actor',
    'maximum_seconds':settings['maximum_seconds'],'per_leg_maximum_seconds':settings['per_leg_maximum_seconds'],
    'criterion':'每段原V1源双足>=.1s、无支持vz>.1、AIR>=.08s且升高>=.03m、合法目标接触及原2s稳定；所有三表面200Hz安全',
    'robot_joint_names':list(robot.joint_names),'physical_dimensions':[env.sim.mj_model.nq,env.sim.mj_model.nv,env.sim.mj_model.nu],
    'gravity_world_mps2':env.sim.mj_model.opt.gravity.tolist(),
    'initialization':'每回合仅原源台reset，无landing reset；中途禁止状态写入；动态平台t/phase不重置',
    'end_recovery_actor_dimension':99 if settings.get('coordinator',{}).get('end_recovery_kind')=='standing99' else 114,
    'alignment_internal_actor_dimension':103 if settings.get('alignment_kind')=='prepared103' else 114 if settings.get('alignment_kind')=='recovery114' else 99,'preparation_actor_current_measurements':'可选Actor103=本体/实际动作/当前body velocity command99+当前起跳点body XY/当前yaw sin-cos4；无future、表面ID、验收FSM',
    'max_steps_diagnostic':a.max_steps,'formal_protocol_receipt':formal_receipt,'switch_is_learned':False,'subpolicy_probability_dimensions':[30,29],'physical_joint_dimensions':29}
(out/'course_evaluation_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
rows=[];begin=time.monotonic();state_read_only_checks=0
with torch.no_grad(),(out/'course_samples.jsonl').open('a') as sample_file:
    for episode in range(a.episodes):
        seed=a.seed+episode;w.seed(seed);periodic=env.action_manager.get_term('periodic_platform')
        periodic.cfg.random_phase=False;phase_bin=episode%8;periodic.cfg.phase=2*math.pi*(phase_bin+random.Random(seed).random())/8
        previous_resets=env.actual_resets;obs,_=w.reset();assert env.actual_resets==previous_resets+1
        coordinator.reset();env.evaluation_episode_index=episode;env.evaluation_seed=seed;env.in_episode=True
        reset_count=env.actual_resets;detector=CourseEventDetector();frames=[];failure=None;last_platform_clock=0.;physics_failed=False
        samples_count=round(settings['maximum_seconds']/env.step_dt)
        if a.max_steps:samples_count=min(samples_count,a.max_steps)
        for step in range(samples_count):
            pre_q=env.sim.data.qpos.clone();pre_v=env.sim.data.qvel.clone();action=coordinator(obs)
            assert torch.equal(pre_q,env.sim.data.qpos) and torch.equal(pre_v,env.sim.data.qvel);state_read_only_checks+=1
            if coordinator.last_transition in ('jump_end','align_middle','recover_end') or (coordinator.last_transition=='final_settle_middle' and settings.get('coordinator',{}).get('prepared_capture_requires_stability',False)):
                term=env.action_manager.get_term('joint_pos')
                bank={'episode':episode,'seed':seed,'event':'actual_middle_stable_before_second_skill',
                    'time_s':float(env.episode_length_buf[0])*env.step_dt,
                    'root_state_world_link_13':torch.cat((robot.data.root_link_pos_w,robot.data.root_link_quat_w,robot.data.root_link_lin_vel_w,robot.data.root_link_ang_vel_w),-1)[0].cpu().tolist(),
                    'joint_names':list(robot.joint_names),'joint_position':robot.data.joint_pos[0].cpu().tolist(),
                    'joint_velocity':robot.data.joint_vel[0].cpu().tolist(),'platform_joint_position':env.scene['target'].data.joint_pos[0].cpu().tolist(),
                    'platform_joint_velocity':env.scene['target'].data.joint_vel[0].cpu().tolist(),'platform_clock_s':float(periodic.t[0]),
                    'platform_phase_rad':float(periodic.phase[0]),'raw_actual_joint_action':term.raw_action[0].cpu().tolist(),
                    'previous_actual_joint_action':term.previous_joint_actions[0].cpu().tolist(),
                    'robot_joint_position_target':term._processed_actions[0].cpu().tolist(),
                    'qacc_warmstart':env.sim.data.qacc_warmstart[0].cpu().tolist(),'course':asdict(course),
                    'scope':'真实前段跳跃/稳定后状态；只允许未来训练episode reset恢复，完整路线回放禁止中途恢复'}
                if coordinator.last_transition=='final_settle_middle':
                    bank.update(current_actor114=coordinator.last_actor[0].cpu().tolist(),target_id='middle',scope='真实当前稳定准备接管状态；仅未来Recovery训练reset恢复，不继承完整路线成功')
                if coordinator.last_transition=='recover_end':
                    support,_=contacts(env)
                    bank.update(event='actual_end_handoff_after_second_airborne',target_id='end',
                        skill_started_s=coordinator.started,current_actor114=coordinator.last_actor[0].cpu().tolist(),
                        current_feet6_world_relative=coordinator.last_feet[0].cpu().tolist(),
                        surface_supports={k:v[0].cpu().tolist() for k,v in support.items()},course_event_history=list(detector.events),
                        scope='真实第二跳后终点合格接管状态；仅未来Recovery训练episode reset恢复，完整路线禁止恢复')
                    filename=f'end_snapshot_{episode:03d}_recover_end.json'
                else:filename=f'middle_snapshot_{episode:03d}_{coordinator.last_transition}.json'
                (out/filename).write_text(json.dumps(bank,indent=2))
            expected=action[:,:29].clone();obs,reward,done,extras=w.step(action)
            assert env.actual_resets==reset_count
            applied=env.action_manager.get_term('joint_pos').raw_action
            assert torch.equal(expected,applied) and applied.shape==(1,29)
            assert obs['actor'].shape==(1,114) and obs['critic'].shape==(1,130)
            t=(step+1)*env.step_dt;clock=float(periodic.t[0]);assert clock>last_platform_clock;last_platform_clock=clock
            support,forces=contacts(env);supports={k:v[0].cpu().tolist() for k,v in support.items()}
            positions={};velocities={};stable={};feet=robot.data.site_pos_w[0,foot_ids]
            tilt=math.degrees(math.acos(float((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))))
            omega=float(torch.linalg.vector_norm(robot.data.root_link_ang_vel_w[0]));height=float(robot.data.root_link_pos_w[0,2])
            illegal=any(bool(torch.linalg.vector_norm(env.scene['nonfoot_'+suffix].data.force[0],dim=-1).gt(10).any()) for suffix in SENSOR_SUFFIX.values())
            fallen=height<.5 or tilt>70
            for name in SURFACES:
                center,velocity,half=provider.current(name);positions[name]=center[0].cpu().tolist();velocities[name]=velocity[0].cpu().tolist()
                relative=robot.data.root_link_lin_vel_w[0]-velocity[0];offset=feet-center[0];other=any(any(supports[k]) for k in SURFACES if k!=name)
                stable[name]=all(supports[name]) and not other and bool((offset[:,:2].abs()<=half-.02).all()) and bool(((offset[:,2]>=-.01)&(offset[:,2]<=.04)).all()) and tilt<=15 and float(torch.linalg.vector_norm(relative[:2]))<=.15 and abs(float(relative[2]))<=.1 and omega<=.5 and not illegal and not fallen
            detector.update(CourseSample(t,supports,height,float(robot.data.root_link_lin_vel_w[0,2]),stable,illegal,fallen))
            physics_failed=any(bool(v[-2]<.5 or v[-1]>70 or any(v[i*6+3]>10 for i in range(3))) for _,v in env.control_contacts)
            reasons=[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])] if bool(done[0]) else []
            record={'episode':episode,'seed':seed,'time_s':t,'physics_dt_s':.005,'platform_clock_s':clock,'platform_phase_rad':float(periodic.phase[0]),
                'phase_bin':phase_bin,'actual_reset_count':env.actual_resets,'coordinator_mode':coordinator.mode,'coordinator_leg':coordinator.leg,
                'coordinator_target':coordinator.target,'landing_goal_y_offset_m':getattr(env,'landing_goal_y_offset',0.),'coordinator_skill_start_s':coordinator.started,'coordinator_transition':coordinator.last_transition,
                'public_actor114_before_step':coordinator.last_actor[0].cpu().tolist(),'public_feet6_before_step':coordinator.last_feet[0].cpu().tolist(),
                'prepared_final_policy_blend_weight_before_step':coordinator.last_final_blend_weight,
                'preparation_goal_features_before_step':None if coordinator.last_preparation_features is None else coordinator.last_preparation_features[0].cpu().tolist(),
                'preparation_body_velocity_command_before_step':None if coordinator.last_preparation_command is None else coordinator.last_preparation_command[0].cpu().tolist(),
                'source_updated_before_step':coordinator.last_source_updated,'action':action[0].cpu().tolist(),'joint_normalized_action':applied[0].cpu().tolist(),
                'joint_position_target_rad':env.action_manager.get_term('joint_pos')._processed_actions[0].cpu().tolist(),
                'joint_position_rad':robot.data.joint_pos[0].cpu().tolist(),'joint_velocity_radps':robot.data.joint_vel[0].cpu().tolist(),
                'root_position':robot.data.root_link_pos_w[0].cpu().tolist(),'root_velocity':robot.data.root_link_lin_vel_w[0].cpu().tolist(),
                'root_quaternion_wxyz':robot.data.root_link_quat_w[0].cpu().tolist(),'tilt_deg':tilt,'angular_speed_radps':omega,
                'feet_position':feet.cpu().tolist(),'surfaces_position':positions,'surfaces_velocity':velocities,'supports':supports,
                'surface_forces_n':{k:v[0].cpu().tolist() for k,v in forces.items()},'stable':stable,'illegal_contact':illegal,'fallen':fallen,
                'detector_leg':detector.leg,'detector_phase':detector.detectors[detector.leg].phase.value,'detector_failed':detector.failed,
                'detector_success':detector.success,'termination_reasons':reasons,'reward':float(reward[0])}
            sample_file.write(json.dumps(record)+'\n')
            if episode<a.video_episodes and step%2==0:frames.append(env.render())
            if detector.failed or physics_failed or reasons:
                failure=detector.failure_reason or ('200hz_physical_failure' if physics_failed else str(reasons));break
            first_stable=next((e['time_s'] for e in detector.events if e['leg_id']==0 and e['phase']=='stable_success'),None)
            second_start=coordinator.started if coordinator.leg==1 else None
            timed_failure=time_failure(t,settings,first_stable,second_start)
            if timed_failure:failure=timed_failure;break
            if detector.success:break
        success=detector.success and not failure
        row={'episode':episode,'seed':seed,'success':success,'duration_s':t,'control_steps':step+1,'events':detector.events,
            'failure_reason':failure if failure else None if success else 'diagnostic_step_limit' if a.max_steps else 'route_timeout',
            'fell':fallen or any(v[-2]<.5 or v[-1]>70 for _,v in env.control_contacts),'illegal_contact':illegal or any(any(v[i*6+3]>10 for i in range(3)) for _,v in env.control_contacts),
            'phase_bin':phase_bin,'platform_initial_phase_rad':float(periodic.phase[0]),'reset_count':reset_count,'mid_episode_resets':env.actual_resets-reset_count,
            'coordinator_transitions':coordinator.trace,'reward_terms':{k:float(v[0]) for k,v in env.reward_manager._episode_sums.items()},
            'final_root_position':robot.data.root_link_pos_w[0].cpu().tolist()}
        if frames:
            import mediapy,imageio_ffmpeg,numpy as np
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe());filename=out/f'course_episode_{episode:03d}.mp4'
            mediapy.write_video(str(filename),np.stack(frames),fps=25)
            row['video']=filename.name;row['video_sha256']=hashlib.sha256(filename.read_bytes()).hexdigest()
        rows.append(row);print(json.dumps(row),flush=True);env.in_episode=False
result={'task':manifest['task'],'episodes':rows,'successes':sum(r['success'] for r in rows),'episodes_count':len(rows),
    'success_rate':sum(r['success'] for r in rows)/len(rows),'state_read_only_checks':state_read_only_checks,
    'elapsed_seconds_monotonic':time.monotonic()-begin,'training_seconds_this_run':0}
(out/'course_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');w.close()
print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)