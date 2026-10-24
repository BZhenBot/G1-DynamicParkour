"""沿用真实rsl-rl PPO，对中间平台再准备进行单模型微调。"""
import argparse,hashlib,json,math,os,time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.os import dump_yaml
from train_baseline import BaselineEnv
from train_jump import configure_runner_model
from train_recovery_standing import build_recovery_standing_configs
from g1_dynamicparkour.course_scene import CourseGeometry,attach_course_scene
from g1_dynamicparkour import course_turn as turn,jump

class TurnEnv(BaselineEnv):
    def __init__(self,cfg,directory,settings,geometry,**kwargs):
        self.course=CourseGeometry(**settings['course']);self.geometry=geometry;self.course_target_id='end';self.course_skill_start_s=-1.
        self.bank=turn.MiddleResetBank(settings['middle_bank']);self.bank_indices=None;self.reset_bank_count=0
        super().__init__(cfg,directory,**kwargs)
        self.bank_indices=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
    def _reset_idx(self,env_ids=None):
        super()._reset_idx(env_ids)
        if self.bank_indices is None:return
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        indices=torch.randint(len(self.bank.records),(len(ids),),device=self.device);self.bank.restore(self,ids,indices)
        self.bank_indices[ids]=indices;self.reset_bank_count+=len(ids)
        if getattr(self,'consistent_prepare_command',False):self.command_manager.get_term('twist')._update_command()
        jump.state(self).detector.phase[ids]=4;jump.state(self).event_counts[ids]=0
    def episode_details(self,env_id):
        s=turn.ensure(self)
        return {'task':'actual_middle_reset_turn_only','success':False,'full_route_success':False,
            'turn_success':bool(s['success'][env_id]),'turn_failed':bool(s['failed'][env_id]),
            'snapshot_index':int(self.bank_indices[env_id]) if self.bank_indices is not None else None,
            'goal_error_rad':float(turn.goal_error(self)[env_id]),
            'launch_x_error_m':None if 'diagnostic_launch_x_error_m' not in s else float(s['diagnostic_launch_x_error_m'][env_id]),
            'stance_x_difference_m':None if 'diagnostic_stance_x_difference_m' not in s else float(s['diagnostic_stance_x_difference_m'][env_id]),
            'strict_middle_safe':None if 'diagnostic_middle_safe' not in s else bool(s['diagnostic_middle_safe'][env_id]),
            'stable_time_s':float(s['stable_time'][env_id]),'jump_events':'不适用：转向子任务，无完整源台起跳'}

def build(settings):
    original=json.load(open(settings['recovery_config']));original.update(seed=settings['seed'],num_envs=settings['num_envs'],episode_length_s=8)
    cfg,agent,g=build_recovery_standing_configs(original);cfg,g=attach_course_scene(cfg,CourseGeometry(**settings['course']))
    cfg.auto_reset=True;cfg.decimation=4;cfg.episode_length_s=8
    cfg.terminations={'update_turn':TerminationTermCfg(func=turn.update),**cfg.terminations,
        'turn_failed':TerminationTermCfg(func=turn.failed),'turn_success':TerminationTermCfg(func=turn.success)}
    for name in ('track_angular_velocity','body_ang_vel','root_angular_velocity_l2'):cfg.rewards.pop(name,None)
    cfg.rewards.update({'turn_heading':RewardTermCfg(func=turn.heading_reward,weight=8),
        'turn_progress':RewardTermCfg(func=turn.progress,weight=10),
        'turn_velocity':RewardTermCfg(func=turn.angular_cost,weight=-5),
        'middle_center':RewardTermCfg(func=turn.center_cost,weight=-50),
        'middle_velocity':RewardTermCfg(func=turn.velocity_cost,weight=-2),
        'middle_foot_outside':RewardTermCfg(func=turn.outside_cost,weight=-300),
        'turn_height':RewardTermCfg(func=turn.height_cost,weight=-100),
        'turn_success_bonus':RewardTermCfg(func=turn.success_bonus,weight=50/.02),
        'turn_failure_bonus':RewardTermCfg(func=turn.failure_bonus,weight=-50/.02)})
    cfg.observations['critic'].terms['foot_air_time']=ObservationTermCfg(func=turn.feet_air_time)
    agent.seed=settings['seed'];agent.num_steps_per_env=48;agent.max_iterations=settings['iterations'];agent.save_interval=settings['save_interval']
    agent.algorithm.learning_rate=settings['learning_rate'];agent.algorithm.schedule='fixed'
    agent.algorithm.gamma=math.sqrt(.99);agent.algorithm.lam=math.sqrt(.95);agent.algorithm.entropy_coef=.0015
    return cfg,agent,g

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    settings=json.loads(a.config.read_text());out=Path(os.environ['G1_RUN_DIR']);root=Path(__file__).resolve().parents[1]
    from v2_budget import enforce_budget
    consumed=enforce_budget(root,out,settings['reserved_training_seconds'])
    cfg,agent,g=build(settings);torch.set_num_threads(4);env=TurnEnv(cfg,out,settings,g,device='cuda:0')
    w=RslRlVecEnvWrapper(env,clip_actions=1.);configure_runner_model(agent,env)
    runner=MjlabOnPolicyRunner(w,asdict(agent),log_dir=str(out/'ppo') if not a.gate else None,device=env.device)
    parent=Path(settings['parent_checkpoint']);assert hashlib.sha256(parent.read_bytes()).hexdigest()==settings['parent_sha256']
    source=torch.load(parent,map_location=env.device,weights_only=False)
    loaded=runner.alg.load(source,{'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict=True)
    assert not loaded and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in source['actor_state_dict'].items())
    scale=float(settings.get('finetune_std_scale',1.));assert 0<scale<=1
    before_std={k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()};mean_error=None
    with torch.no_grad():
        if scale!=1.:
            from tensordict import TensorDict
            current=torch.tensor([env.bank.records[i%len(env.bank.records)]['current_actor114'] for i in range(env.num_envs)],device=env.device)
            real_obs=TensorDict({'actor':current},batch_size=[env.num_envs]);old_mode=runner.alg.actor.training;runner.alg.actor.eval();mean_before=runner.alg.actor(real_obs).clone()
        runner.alg.actor.distribution.std_param.mul_(scale).clamp_min_(settings['minimum_std'])
        if scale!=1.:
            mean_error=float((mean_before-runner.alg.actor(real_obs)).abs().max());assert mean_error==0.;runner.alg.actor.train(old_mode)
    changed=[k for k,v in before_std.items() if not torch.equal(v,runner.alg.actor.state_dict()[k])];assert all(k=='distribution.std_param' for k in changed)
    (out/'initial_exploration_change.json').write_text(json.dumps({'std_scale':scale,'minimum_initial_std':settings['minimum_std'],'only_changed_actor_state_keys':changed,'deterministic_mean_error_on_actual_banks':mean_error,'parent_raw_std':before_std['distribution.std_param'].cpu().tolist(),'new_effective_std':runner.alg.actor.distribution.std_param.detach().clamp(*runner.alg.actor.distribution.std_range).cpu().tolist(),'scope':'只修改初始探索分布，不改确定性均值/物理/任务判据，PPO仍在真实高斯概率空间'},ensure_ascii=False,indent=2)+'\n')
    frozen={k:v.clone() for k,v in runner.alg.actor.standing_prior.state_dict().items()}
    initial={k:v.detach().clone() for k,v in runner.alg.actor.named_parameters() if v.requires_grad}
    if a.gate:
        with torch.no_grad():
            obs,_=w.reset();assert obs['actor'].shape==(settings['num_envs'],114) and obs['critic'].shape==(settings['num_envs'],130)
            assert env.action_manager.total_action_dim==29
            snapshots=[];density_error=0.
            if hasattr(env,'verify_initial_inputs'):env.verify_initial_inputs(obs)
            for step in range(100):
                beforeq=env.sim.data.qpos.clone();beforev=env.sim.data.qvel.clone();sample=runner.alg.actor(obs,stochastic_output=True)
                mu,sigma=runner.alg.actor.output_mean,runner.alg.actor.output_std
                expected=(-.5*((sample-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
                density_error=max(density_error,float((expected-runner.alg.actor.get_output_log_prob(sample)).abs().max()));assert density_error<1e-4
                action=sample.clamp(-1,1)
                assert torch.equal(beforeq,env.sim.data.qpos) and torch.equal(beforev,env.sim.data.qvel)
                obs,reward,done,_=w.step(action)
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
                snapshots.append({'step':step,'mean_error_rad':float(turn.goal_error(env).abs().mean()),'done':done.cpu().tolist()})
            report={'pass':True,'actor114':True,'critic130':True,'physical_action29':True,'actual_steps':100,'bank_resets':env.reset_bank_count,
                'source_bank_count':len(env.bank.records),'density_error_max':density_error,'snapshots':snapshots,'scope':'4env物理/加载/reset/只读闸门，不是完整路线成功'}
        (out/'turn_physical_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='snapshots'}));w.close();return
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(agent))
    import rsl_rl.runners.on_policy_runner as module
    module.time=SimpleNamespace(time=time.monotonic);started=time.monotonic()
    runner.learn(num_learning_iterations=settings['iterations'],init_at_random_ep_len=False)
    elapsed=time.monotonic()-started
    assert all(torch.equal(v,runner.alg.actor.standing_prior.state_dict()[k]) for k,v in frozen.items())
    delta=sum(float((p.detach()-initial[k]).square().sum()) for k,p in runner.alg.actor.named_parameters() if p.requires_grad)**.5
    assert delta>0 and runner.alg.optimizer.state
    records=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()] if (out/'episodes.jsonl').exists() else []
    manifest={'task':'actual_middle_reset_turn_only','seed':settings['seed'],'iterations':settings['iterations'],
        'transitions':settings['num_envs']*48*settings['iterations'],'elapsed_training_seconds':elapsed,
        'total_v2_training_seconds_after':consumed+elapsed,'budget_seconds':21600,'parent':str(parent),'parent_sha256':settings['parent_sha256'],
        'trainable_actor_delta_l2':delta,'frozen_standing_prior_unchanged':True,'completed_turn_episodes':len(records),
        'turn_successes':sum(v['turn_success'] for v in records),'full_route_successes':'不适用：所有训练episode从真实中间快照reset，不能计完整路线',
        'bank_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in env.bank.paths},
        'checkpoint_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'ppo').glob('*.pt')},
        'reward_dt':'持续项RewardManager乘dt一次；事件bonus/dt仅单步；独立路线验收门槛未修改'}
    (out/'training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False),flush=True);w.close()
if __name__=='__main__':main()