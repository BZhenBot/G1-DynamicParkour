"""V2终点Recovery114/29单模型微调；真实状态只在episode reset恢复。"""
import argparse,hashlib,json,math,os,time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.os import dump_yaml
from train_baseline import BaselineEnv
from train_jump import configure_runner_model
from train_recovery_standing import build_recovery_standing_configs
from g1_dynamicparkour.course_scene import CourseGeometry,attach_course_scene
from g1_dynamicparkour import end_recovery as task,jump
from v2_budget import enforce_budget

class EndRecoveryEnv(BaselineEnv):
    def __init__(self,cfg,directory,settings,g,**kwargs):
        self.settings=settings;self.course=CourseGeometry(**settings['course']);self.geometry=g
        self.course_target_id='end';self.course_skill_start_s=-1.;self.bank=task.EndResetBank(settings['end_bank_manifest'])
        self.bank_indices=None;self.bank_resets=0;self.initial_public_contact_measurement=None
        super().__init__(cfg,directory,**kwargs)
        self.bank_indices=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.course_skill_start_s=torch.zeros(self.num_envs,device=self.device)
    def _reset_idx(self,env_ids=None):
        super()._reset_idx(env_ids)
        if self.bank_indices is None:return
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        indices=torch.randint(len(self.bank.records),(len(ids),),device=self.device);self.bank.restore(self,ids,indices)
        self.bank_indices[ids]=indices;self.bank_resets+=len(ids)
        inherited=torch.tensor([self.bank.records[i]['time_s']-self.bank.records[i]['skill_started_s'] for i in indices.cpu().tolist()],device=self.device)
        self.course_skill_start_s[ids]=-inherited
        if self.settings.get('restore_initial_public_measurement',False):
            if self.initial_public_contact_measurement is None:self.initial_public_contact_measurement=torch.zeros(self.num_envs,8,device=self.device)
            self.initial_public_contact_measurement[ids]=torch.tensor([self.bank.records[i]['current_actor114'][106:114] for i in indices.cpu().tolist()],device=self.device)
        js=jump.state(self);js.detector.phase[ids]=4;js.event_counts[ids]=0;js.detector.events[ids]=0
        s=task.state(self)
        for name in ('stable_time','success','failed','success_pulse','failure_pulse'):s[name][ids]=0
        s['stable_since'][ids]=-1.
    def episode_details(self,i):
        s=task.state(self)
        return {'task':'actual_end_reset_recovery_only','success':False,'full_route_success':False,
            'recovery_success':bool(s['success'][i]),'recovery_failed':bool(s['failed'][i]),'stable_seconds':float(s['stable_time'][i]),
            'snapshot_index':int(self.bank_indices[i]) if self.bank_indices is not None else None,
            'jump_events':'不适用：只恢复任务，不继承两跳事件','root_velocity_world':self.scene['robot'].data.root_link_lin_vel_w[i].cpu().tolist()}
    def close(self):
        if self.bank_indices is not None:
            s=task.state(self);rows=[{'env_id':i,'partial_episode':True,'control_steps':int(self.episode_length_buf[i]),'snapshot_index':int(self.bank_indices[i]),
                'stable_seconds':float(s['stable_time'][i]),'reward_terms':{n:float(v[i]) for n,v in self.reward_manager._episode_sums.items()}} for i in range(self.num_envs)]
            (self.directory/'end_recovery_active_snapshot.json').write_text(json.dumps(rows,indent=2))
        super().close()

def build(settings):
    original=json.loads(Path(settings['recovery_config']).read_text())
    original.update(seed=settings['seed'],num_envs=settings['num_envs'],episode_length_s=settings['episode_length_s'])
    cfg,agent,g=build_recovery_standing_configs(original);rewards=dict(cfg.rewards)
    cfg,g=attach_course_scene(cfg,CourseGeometry(**settings['course']));cfg.rewards=rewards
    if settings.get('restore_initial_public_measurement',False):
        cfg.observations['actor'].terms['current_unlabelled_foot_contact'].func=task.initial_contact_measurement
    cfg.auto_reset=True;cfg.decimation=4;cfg.episode_length_s=settings['episode_length_s']
    cfg.terminations={'update_end_recovery':TerminationTermCfg(func=task.update),'end_recovery_success':TerminationTermCfg(func=task.completed),
        'end_recovery_failed':TerminationTermCfg(func=task.failed),'time_out':cfg.terminations['time_out']}
    cfg.rewards['success_event'].func=task.success_bonus;cfg.rewards['failure_event'].func=task.failure_bonus
    if settings.get('align_strict_stability_reward',False):
        term=cfg.rewards.pop('landed_dual_support');term.func=task.stable_reward;term.params={}
        cfg.rewards['end_strict_stability']=term
    agent.seed=settings['seed'];agent.num_steps_per_env=settings['rollout_steps'];agent.max_iterations=settings['iterations'];agent.save_interval=settings['save_interval']
    agent.algorithm.learning_rate=settings['learning_rate'];agent.algorithm.schedule='fixed'
    return cfg,agent,g

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    settings=json.loads(a.config.read_text());out=Path(os.environ['G1_RUN_DIR']);root=Path(__file__).resolve().parents[1]
    if a.gate:settings['num_envs']=4
    consumed=enforce_budget(root,out,settings['reserved_training_seconds']);cfg,agent,g=build(settings);torch.set_num_threads(4)
    env=EndRecoveryEnv(cfg,out,settings,g,device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=1.)
    configure_runner_model(agent,env);runner=MjlabOnPolicyRunner(w,asdict(agent),None if a.gate else str(out/'ppo'),device='cuda:0')
    cp=Path(settings['parent_checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==settings['parent_sha256']
    parent=torch.load(cp,map_location=env.device,weights_only=False);loaded=runner.alg.load(parent,{'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict=True)
    assert not loaded and runner.current_learning_iteration==0 and not runner.alg.optimizer.state
    for role in ('actor','critic'):assert all(torch.equal(v,getattr(runner.alg,role).state_dict()[k]) for k,v in parent[role+'_state_dict'].items())
    actor=runner.alg.actor;actor.eval();obs,_=w.reset()
    assert obs['actor'].shape==(env.num_envs,114) and obs['critic'].shape==(env.num_envs,130) and env.action_manager.total_action_dim==29
    with torch.no_grad():
        mean=actor(obs).clone();actor.distribution.std_param.clamp_min_(settings['minimum_std']);assert torch.equal(mean,actor(obs))
    frozen={k:v.clone() for k,v in actor.standing_prior.state_dict().items()};initial={n:v.detach().clone() for n,v in actor.named_parameters() if v.requires_grad}
    (out/'end_recovery_curriculum.json').write_text(json.dumps({'scope':'仅终点恢复，不计完整路线','bank_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in env.bank.paths},
        'actor114':'本体99/当前真实平台中心位姿速度6/已知技能clock1/三面无身份接触8；无future/FSM',
        'critic130':'当前接触phase4/当前gap-A-f-平台sin-cos明确特权；未继承事件奖金','reset_only_state_restore':True,
        'mean_preserved_before_PPO':True,'initial_std':actor.distribution.std_param.detach().cpu().tolist(),'strict_target_stable_seconds':2},ensure_ascii=False,indent=2))
    if a.gate:
        density=0.;readonly=0
        with torch.no_grad():
            for i in range(200):
                action=actor(obs,stochastic_output=True);mu,sigma=actor.output_mean,actor.output_std
                expected=(-.5*((action-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
                density=max(density,float((expected-actor.get_output_log_prob(action)).abs().max()));assert density<1e-4
                q,v=env.sim.data.qpos.clone(),env.sim.data.qvel.clone();task.update(env);assert torch.equal(q,env.sim.data.qpos) and torch.equal(v,env.sim.data.qvel);readonly+=1
                from g1_dynamicparkour.course_scene import contacts,unlabelled_contact
                if settings.get('restore_initial_public_measurement',False):
                    _,forces=contacts(env);before_force={k:v.clone() for k,v in forces.items()}
                    task.initial_contact_measurement(env)
                    _,after_force=contacts(env);assert all(torch.equal(v,after_force[k]) for k,v in before_force.items())
                obs,reward,done,_=w.step(action);assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
                active=env.episode_length_buf>0
                if active.any():assert torch.equal(obs['actor'][active,106:114],unlabelled_contact(env)[active])
                assert torch.allclose(obs['actor'][:,67:96],jump.scaled_previous_actions(env),atol=1e-6,rtol=1e-6)
        report={'pass':True,'num_envs':env.num_envs,'source_bank_count':len(env.bank.records),'actual_physics_dimensions':[env.sim.mj_model.nq,env.sim.mj_model.nv,env.sim.mj_model.nu],
            'actor114':True,'critic130':True,'direct_action29':True,'readonly_checks':readonly,'density_error':density,'scope':'4env接口/物理闸门，不是完整路线成功'}
        (out/'end_recovery_physics_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);w.close();return
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(agent))
    import rsl_rl.runners.on_policy_runner as module
    module.time=SimpleNamespace(time=time.monotonic);start=time.monotonic();runner.learn(num_learning_iterations=settings['iterations'],init_at_random_ep_len=False);elapsed=time.monotonic()-start
    assert all(torch.equal(v,actor.standing_prior.state_dict()[k]) for k,v in frozen.items())
    delta=sum(float((v.detach()-initial[n]).square().sum()) for n,v in actor.named_parameters() if v.requires_grad)**.5;assert delta>0 and runner.alg.optimizer.state
    rows=[json.loads(v) for v in (out/'episodes.jsonl').read_text().splitlines()]
    result={'task':'actual_end_reset_recovery_only','iterations':settings['iterations'],'transitions_this_run':env.num_envs*settings['rollout_steps']*settings['iterations'],
        'elapsed_training_seconds':elapsed,'total_v2_training_seconds_after':consumed+elapsed,'parent_checkpoint':str(cp),'parent_sha256':settings['parent_sha256'],
        'frozen_standing_prior_unchanged':True,'trainable_actor_delta_l2':delta,'completed_episodes':len(rows),'recovery_successes':sum(r['recovery_success'] for r in rows),
        'full_route_success':'不适用：仅终点恢复reset训练','checkpoint_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'ppo').glob('*.pt')}}
    (out/'training_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True);w.close()
if __name__=='__main__':main()
