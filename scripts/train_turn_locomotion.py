"""V1有限平台Locomotion PPO的中间台转向微调；当前命令，无未来状态。"""
import argparse,hashlib,json,math,os,time
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.os import dump_yaml
from train_baseline import build_configs
from train_course_turn import TurnEnv
from g1_dynamicparkour.platform_control import attach_platform_scene
from g1_dynamicparkour.scene import Geometry
from g1_dynamicparkour.course_scene import CourseGeometry,attach_course_scene,PlatformTargetProvider
from g1_dynamicparkour import course_turn as turn
from g1_dynamicparkour.checkpoint_transfer import expand_model_state
from v2_budget import enforce_budget

def current_turn_command(env):
    if getattr(env,'consistent_prepare_command',False):return env.command_manager.get_command('twist')
    r=env.scene['robot'];h=r.data.heading_w;center,_,_=PlatformTargetProvider(env).current('middle')
    desired=center[:,:2].clone();desired[:,0]+=getattr(env,'turn_launch_offset_m',.1)
    maximum=getattr(env,'turn_linear_speed_limit',.1);gain=getattr(env,'turn_linear_gain',.8)
    v=((desired-r.data.root_link_pos_w[:,:2])*gain).clamp(-maximum,maximum)
    return torch.stack((h.cos()*v[:,0]+h.sin()*v[:,1],-h.sin()*v[:,0]+h.cos()*v[:,1],turn.desired_turn_velocity(env)),-1)

def current_linear_tracking(env):
    r=env.scene['robot'];h=r.data.heading_w;command=current_turn_command(env)
    vx=h.cos()*command[:,0]-h.sin()*command[:,1];vy=h.sin()*command[:,0]+h.cos()*command[:,1]
    desired=torch.stack((vx,vy),-1)
    return torch.exp(-(r.data.root_link_lin_vel_w[:,:2]-desired).square().sum(-1)/.25**2)

def current_stepping_reward(env):
    air=turn.ensure(env)['air_time'];active=turn.goal_error(env).abs()>.1
    return ((air>.05)&(air<.5)).float().sum(-1)*active

def privileged_current_yaw(env):
    h=env.scene['robot'].data.heading_w;return torch.stack((h.sin(),h.cos()),-1)

def current_prepare_actor_features(env):
    from g1_dynamicparkour.preparation_observation import measured_launch_goal_features
    return measured_launch_goal_features(env,env.turn_launch_offset_m)

def current_prepare_critic_features(env):return current_prepare_actor_features(env)[:,:2]

class LocomotionTurnEnv(TurnEnv):
    def __init__(self,cfg,directory,settings,geometry,**kwargs):
        self.consistent_prepare_command=settings.get('consistent_feedback_command',False)
        self.turn_world_heading=True;self.turn_maximum_velocity=settings.get('turn_maximum_velocity_radps',.5)
        self.turn_linear_speed_limit=settings.get('turn_linear_speed_limit_mps',.1);self.turn_linear_gain=settings.get('turn_linear_gain',.8)
        self.turn_launch_offset_m=settings.get('turn_launch_offset_m',.1);self.turn_require_launch_pose=settings.get('turn_require_launch_pose',False)
        names=json.loads(Path(settings['middle_bank'][0]).read_text())['joint_names']
        self.bank_original_units=torch.tensor([2. if any(x in n for x in ('hip_pitch','knee','ankle_pitch')) else 1. for n in names],device=kwargs['device'])
        super().__init__(cfg,directory,settings,geometry,**kwargs)

def build(settings):
    base=json.load(open(settings['locomotion_config']));base.update(seed=settings['seed'],num_envs=settings['num_envs'],episode_length_s=8)
    cfg,agent=build_configs(base)
    cfg=attach_platform_scene(cfg,Geometry(gap=.02,robot_start_offset_x=.2,amplitude=.05,frequency=.1,platform_servo_kp=900000.,platform_servo_kv=36000.))
    cfg,g=attach_course_scene(cfg,CourseGeometry(**settings['course']))
    for group in ('actor','critic'):
        for name in ('platform_relative_state','task_clock','current_unlabelled_foot_contact'):cfg.observations[group].terms.pop(name,None)
        cfg.observations[group].terms['command']=ObservationTermCfg(func=current_turn_command)
    cfg.observations['critic'].terms['foot_air_time']=ObservationTermCfg(func=turn.feet_air_time)
    cfg.observations['critic'].terms['current_yaw_privileged']=ObservationTermCfg(func=privileged_current_yaw)
    if settings.get('explicit_current_launch_goal',False):
        cfg.observations['actor'].terms['current_launch_goal']=ObservationTermCfg(func=current_prepare_actor_features)
        cfg.observations['critic'].terms['current_launch_goal']=ObservationTermCfg(func=current_prepare_critic_features)
    if settings.get('consistent_feedback_command',False):
        from g1_dynamicparkour.preparation_command import PrepareVelocityCommandCfg
        cfg.commands['twist']=PrepareVelocityCommandCfg(entity_name='robot',resampling_time_range=(100.,100.),debug_vis=False,heading_command=False,init_velocity_prob=0.,ranges=PrepareVelocityCommandCfg.Ranges(lin_vel_x=(0.,0.),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))
    cfg.auto_reset=True;cfg.decimation=4;cfg.episode_length_s=8
    cfg.terminations={'update_turn':TerminationTermCfg(func=turn.update),**cfg.terminations,
        'turn_failed':TerminationTermCfg(func=turn.failed),'turn_success':TerminationTermCfg(func=turn.success)}
    for name in ('track_angular_velocity','body_ang_vel','root_angular_velocity_l2'):cfg.rewards.pop(name,None)
    cfg.rewards['track_linear_velocity']=RewardTermCfg(func=current_linear_tracking,weight=2)
    cfg.rewards['air_time']=RewardTermCfg(func=current_stepping_reward,weight=1)
    cfg.rewards.update({'turn_heading':RewardTermCfg(func=turn.heading_reward,weight=8),
        'turn_progress':RewardTermCfg(func=turn.progress,weight=10),'turn_velocity':RewardTermCfg(func=turn.angular_cost,weight=-5),
        'middle_center':RewardTermCfg(func=turn.center_cost,weight=-50),'middle_velocity':RewardTermCfg(func=turn.velocity_cost,weight=-2),
        'middle_foot_outside':RewardTermCfg(func=turn.outside_cost,weight=-300),'turn_height':RewardTermCfg(func=turn.height_cost,weight=-100),
        'turn_success_bonus':RewardTermCfg(func=turn.success_bonus,weight=50/.02),'turn_failure_bonus':RewardTermCfg(func=turn.failure_bonus,weight=-50/.02)})
    if 'turn_action_rate_weight' in settings:
        cfg.rewards['action_rate_l2'].weight=float(settings['turn_action_rate_weight'])
    if 'turn_center_cost_weight' in settings:
        cfg.rewards['middle_center'].weight=float(settings['turn_center_cost_weight'])
    if 'prepare_roll_pitch_rate_weight' in settings:
        cfg.rewards['prepare_roll_pitch_rate']=RewardTermCfg(func=turn.roll_pitch_rate_cost,weight=float(settings['prepare_roll_pitch_rate_weight']))
    if 'prepare_planar_tilt_weight' in settings:
        cfg.rewards['prepare_planar_tilt']=RewardTermCfg(func=turn.planar_tilt_cost,weight=float(settings['prepare_planar_tilt_weight']))
    if 'prepare_ankle_roll_velocity_weight' in settings:
        cfg.rewards['prepare_ankle_roll_velocity']=RewardTermCfg(func=turn.ankle_roll_velocity_cost,weight=float(settings['prepare_ankle_roll_velocity_weight']))
    # 此子技能原动作单位×1；部署用相同D换算到Jump环境×2，并非改变PD/力矩。
    agent.seed=settings['seed'];agent.num_steps_per_env=48;agent.max_iterations=settings['iterations'];agent.save_interval=settings['save_interval']
    agent.algorithm.learning_rate=settings['learning_rate'];agent.algorithm.schedule='fixed'
    agent.algorithm.gamma=math.sqrt(.99);agent.algorithm.lam=math.sqrt(.95);agent.algorithm.entropy_coef=.002
    return cfg,agent,g

def calibrate_yaw_command(model):
    n=model.obs_normalizer;col=98
    if not hasattr(n,'_std'):raise ValueError('需要已验证EmpiricalNormalization')
    with torch.no_grad():
        mean=n._mean[0,col].clone();std=n._std[0,col].clone()
        # 零命令时保留第一层preactivation；新命令±.5在合理归一范围，其他统计不动。
        model.mlp[0].bias.sub_(model.mlp[0].weight[:,col]*mean/(std+n.eps))
        n._mean[0,col]=0.;n._std[0,col]=.25;n._var[0,col]=.25**2
    return {'column':98,'old_mean':float(mean),'old_std':float(std),'new_mean':0.,'new_std':.25,'zero_command_preactivation_preserved':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    settings=json.loads(a.config.read_text());out=Path(os.environ['G1_RUN_DIR']);root=Path(__file__).resolve().parents[1]
    consumed=enforce_budget(root,out,settings['reserved_training_seconds']);cfg,agent,g=build(settings);torch.set_num_threads(4)
    env=LocomotionTurnEnv(cfg,out,settings,g,device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=MjlabOnPolicyRunner(w,asdict(agent),log_dir=str(out/'ppo') if not a.gate else None,device=env.device)
    parent=Path(settings['parent_checkpoint']);assert hashlib.sha256(parent.read_bytes()).hexdigest()==settings['parent_sha256']
    source=torch.load(parent,map_location=env.device,weights_only=False)
    actor_dim=103 if settings.get('explicit_current_launch_goal',False) else 99
    critic_dim=115 if settings.get('explicit_current_launch_goal',False) else 113
    runner.alg.actor.load_state_dict(expand_model_state(source['actor_state_dict'],actor_dim),strict=True)
    runner.alg.critic.load_state_dict(expand_model_state(source['critic_state_dict'],critic_dim),strict=True)
    assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state
    initial_obs=w.get_observations();assert initial_obs['actor'].shape==(settings['num_envs'],actor_dim) and initial_obs['critic'].shape==(settings['num_envs'],critic_dim)
    import copy
    obs=initial_obs.clone();obs['actor'][:,98]=0.;obs['critic'][:,98]=0.
    runner.alg.actor.eval();runner.alg.critic.eval()
    with torch.no_grad():beforea=runner.alg.actor(obs).clone();beforec=runner.alg.critic(obs).clone()
    calibration={role:calibrate_yaw_command(getattr(runner.alg,role)) for role in ('actor','critic')}
    with torch.no_grad():
        errora=float((beforea-runner.alg.actor(obs)).abs().max());errorc=float((beforec-runner.alg.critic(obs)).abs().max())
        assert errora<1e-4 and errorc<1e-3,(errora,errorc)
        runner.alg.actor.distribution.std_param.clamp_min_(.1)
    initial={k:v.detach().clone() for k,v in runner.alg.actor.named_parameters() if v.requires_grad}
    if a.gate:
        traces=[];density_error=0.
        if settings.get('explicit_current_launch_goal',False):
            if source['actor_state_dict']['mlp.0.weight'].shape[1]==99:assert not torch.count_nonzero(runner.alg.actor.mlp[0].weight[:,99:])
            if source['critic_state_dict']['mlp.0.weight'].shape[1]==113:assert not torch.count_nonzero(runner.alg.critic.mlp[0].weight[:,113:])
        with torch.no_grad():
            obs,_=w.reset()
            assert torch.allclose(obs['actor'][:,96:99],current_turn_command(env))
            if settings.get('explicit_current_launch_goal',False):
                assert torch.allclose(obs['actor'][:,99:103],current_prepare_actor_features(env))
                assert torch.allclose(obs['critic'][:,113:115],current_prepare_critic_features(env))
            assert torch.allclose(obs['critic'][:,96:99],current_turn_command(env))
            if settings.get('consistent_feedback_command',False):
                assert torch.allclose(obs['actor'][:,96:99],env.command_manager.get_command('twist'))
            term=env.action_manager.get_term('joint_pos')
            assert torch.allclose(term.raw_action*term.scale+term.offset,term._processed_actions,atol=1e-6)
            for i in range(100):
                q=env.sim.data.qpos.clone();v=env.sim.data.qvel.clone();action=runner.alg.actor(obs,stochastic_output=True)
                assert torch.equal(q,env.sim.data.qpos) and torch.equal(v,env.sim.data.qvel)
                mean,sigma=runner.alg.actor.output_mean,runner.alg.actor.output_std
                manual=(-.5*((action-mean)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
                error=float((manual-runner.alg.actor.get_output_log_prob(action)).abs().max());density_error=max(density_error,error);assert error<1e-4
                obs,reward,done,_=w.step(action);assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
                traces.append({'step':i,'mean_yaw_error':float(turn.goal_error(env).abs().mean()),'done':done.cpu().tolist()})
        report={'pass':True,'actor_dimension':actor_dim,'critic_dimension':critic_dim,'actor99':actor_dim==99,'critic113_current_yaw_privileged':critic_dim==113,'new_current_goal_columns_initially_zero':settings.get('explicit_current_launch_goal',False) and source['actor_state_dict']['mlp.0.weight'].shape[1]==99,'reward_observation_command_consistent':settings.get('consistent_feedback_command',False),'physical29_original_units':True,'density_error_max':density_error,
            'command_normalization_calibration':calibration,'zero_command_mean_error':[errora,errorc],'bank_history_units_checked':True,'bank_reset_count':env.reset_bank_count,
            'trace':traces,'scope':'4env100步，真实bank+PPO密度/父均值/当前命令/单位/只读，不是路线成功'}
        (out/'locomotion_turn_gate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='trace'},ensure_ascii=False));w.close();return
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(agent))
    import rsl_rl.runners.on_policy_runner as module
    module.time=SimpleNamespace(time=time.monotonic);start=time.monotonic();runner.learn(num_learning_iterations=settings['iterations'],init_at_random_ep_len=False)
    elapsed=time.monotonic()-start;delta=sum(float((p.detach()-initial[k]).square().sum()) for k,p in runner.alg.actor.named_parameters() if p.requires_grad)**.5
    assert delta>0 and runner.alg.optimizer.state
    episodes=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()] if (out/'episodes.jsonl').exists() else []
    report={'task':'actual_middle_reset_locomotion_turn_only','iterations':settings['iterations'],'transitions':settings['num_envs']*48*settings['iterations'],
        'elapsed_training_seconds':elapsed,'total_v2_training_seconds_after':consumed+elapsed,'budget_seconds':21600,'parent':str(parent),'parent_sha256':settings['parent_sha256'],
        'actor_dimension':actor_dim,'critic_dimension':critic_dim,'actor':'V1本体/动作历史+当前body velocity command3；启用时再加当前起跳点body XY及当前yaw sin/cos4，无future/支持面ID',
        'critic':'V1当前特权111+当前yaw sin/cos2；启用时另加当前起跳点body XY2，训练专用','command_normalization_calibration':calibration,
        'turn_successes':sum(x['turn_success'] for x in episodes),'completed_turn_episodes':len(episodes),'full_route_successes':'不适用：真实middle reset子任务',
        'trainable_actor_delta_l2':delta,'checkpoint_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'ppo').glob('*.pt')},
        'bank_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in env.bank.paths}}
    (out/'training_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True);w.close()
if __name__=='__main__':main()