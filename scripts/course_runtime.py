"""双跳冻结权重部署公用构造，V1 factory 不改写。"""
from dataclasses import asdict
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from train_jump import build_jump_configs,configure_runner_model
from train_recovery_standing import build_recovery_standing_configs
from g1_dynamicparkour.course_scene import CourseGeometry,attach_course_scene
from g1_dynamicparkour.recovery_coordinate import bind_recovery_coordinate
from g1_dynamicparkour.recovery_standing_policy import RecoveryStandingMLP
from g1_dynamicparkour.skill_coordinator import SkillCoordinator

class CourseEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,course,geometry,**kwargs):
        self.course=course; self.geometry=geometry; self.course_target_id='middle'; self.course_skill_start_s=0.
        self.actual_resets=0
        super().__init__(cfg,**kwargs)
    def _reset_idx(self,env_ids=None):
        self.actual_resets+=1; self.course_target_id='middle'; self.course_skill_start_s=0.
        return super()._reset_idx(env_ids)


def build_course(settings,seed,num_envs=1):
    original=json.load(open(settings['jump_config'])); original['seed']=seed; original['num_envs']=num_envs
    cfg,agent,_=build_jump_configs(original)
    course=CourseGeometry(**settings['course'])
    cfg,g=attach_course_scene(cfg,course)
    cfg.decimation=4; cfg.seed=seed; cfg.auto_reset=False; cfg.episode_length_s=settings['maximum_seconds']+1
    for group in cfg.observations.values(): group.enable_corruption=False
    from mjlab.envs.mdp.events import reset_joints_by_offset
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg.events['joint_noise']=EventTermCfg(func=reset_joints_by_offset,mode='reset',
        params={'position_range':(-.01,.01),'velocity_range':(0,0),'asset_cfg':SceneEntityCfg('robot',joint_names=('.*',))})
    return cfg,agent,course,g


def load_controllers(env,agent,settings):
    wrapper=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    configure_runner_model(agent,env)
    runner=MjlabOnPolicyRunner(wrapper,asdict(agent),log_dir=None,device=env.device)
    bind_recovery_coordinate(env,runner.alg.actor)
    runner.load(settings['jump_checkpoint'])
    policy=runner.get_inference_policy(device=env.device)
    rs=json.load(open(settings['recovery_config']))
    _,ra,_=build_recovery_standing_configs(rs)
    configure_runner_model(ra,env)
    kwargs=asdict(ra.actor); kwargs.pop('class_name')
    recovery=RecoveryStandingMLP(wrapper.get_observations(),{'actor':['actor']},'actor',29,**kwargs).to(env.device)
    recovery.load_state_dict(torch.load(settings['recovery_checkpoint'],map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
    recovery.eval()
    if not torch.equal(recovery.physical_units,runner.alg.actor.jump_prior.physical_units): raise ValueError('物理单位不一致')
    coordinator=SkillCoordinator(env,policy,runner.alg.actor,recovery,settings.get('coordinator'))
    if settings.get('settle_recovery_checkpoint'):
        import hashlib
        path=settings['settle_recovery_checkpoint'];assert hashlib.sha256(open(path,'rb').read()).hexdigest()==settings['settle_recovery_sha256']
        settle_actor=RecoveryStandingMLP(wrapper.get_observations(),{'actor':['actor']},'actor',29,**kwargs).to(env.device)
        settle_actor.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
        if not torch.equal(settle_actor.physical_units,recovery.physical_units):raise ValueError('准备接管Recovery物理单位不一致')
        settle_actor.eval();settle_actor.requires_grad_(False);coordinator.settle_recovery_actor=settle_actor
    if settings.get('end_recovery_checkpoint'):
        import hashlib
        path=settings['end_recovery_checkpoint']
        assert hashlib.sha256(open(path,'rb').read()).hexdigest()==settings['end_recovery_sha256']
        end_actor=RecoveryStandingMLP(wrapper.get_observations(),{'actor':['actor']},'actor',29,**kwargs).to(env.device)
        end_actor.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
        if not torch.equal(end_actor.physical_units,runner.alg.actor.jump_prior.physical_units):raise ValueError('终点Recovery物理单位不一致')
        end_actor.eval();end_actor.requires_grad_(False)
        assert torch.equal(end_actor.physical_units,recovery.physical_units)
        coordinator.end_recovery_actor=end_actor
    if settings.get('alignment_checkpoint'):
        import hashlib,inspect
        from rsl_rl.models import MLPModel
        from train_baseline import build_configs
        from tensordict import TensorDict
        path=settings['alignment_checkpoint']
        assert hashlib.sha256(open(path,'rb').read()).hexdigest()==settings['alignment_sha256']
        if settings.get('alignment_kind')=='recovery114':
            turn_actor=RecoveryStandingMLP(wrapper.get_observations(),{'actor':['actor']},'actor',29,**kwargs).to(env.device)
            turn_actor.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
            turn_actor.eval();turn_actor.requires_grad_(False)
            assert torch.equal(turn_actor.physical_units,recovery.physical_units)
            coordinator.alignment_actor=turn_actor;coordinator.alignment_kind='recovery114'
        else:
            _,walk_agent=build_configs(json.load(open(settings['alignment_config'])))
            params={k:v for k,v in asdict(walk_agent.actor).items() if k in inspect.signature(MLPModel.__init__).parameters}
            proprio_values=wrapper.get_observations()['actor'][:,:99]
            if settings.get('alignment_kind')=='prepared103':
                from g1_dynamicparkour.preparation_observation import measured_launch_goal_features
                features=measured_launch_goal_features(env,settings['coordinator'].get('launch_point_from_middle_center_m',.2))
                proprio_values=torch.cat((proprio_values,features),-1)
                coordinator.alignment_kind='prepared103'
            proprio=TensorDict({'actor':proprio_values},batch_size=[1])
            walker=MLPModel(proprio,{'actor':['actor']},'actor',29,**params).to(env.device)
            walker.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
            walker.eval();walker.requires_grad_(False);coordinator.alignment_actor=walker
    if settings.get('position_checkpoint'):
        import hashlib,inspect
        from rsl_rl.models import MLPModel
        from train_baseline import build_configs
        from tensordict import TensorDict
        path=settings['position_checkpoint']
        assert hashlib.sha256(open(path,'rb').read()).hexdigest()==settings['position_sha256']
        _,pa=build_configs(json.load(open(settings['position_config'])))
        params={k:v for k,v in asdict(pa.actor).items() if k in inspect.signature(MLPModel.__init__).parameters}
        proprio=TensorDict({'actor':wrapper.get_observations()['actor'][:,:99]},batch_size=[1])
        position_actor=MLPModel(proprio,{'actor':['actor']},'actor',29,**params).to(env.device)
        position_actor.load_state_dict(torch.load(path,map_location=env.device,weights_only=False)['actor_state_dict'],strict=True)
        position_actor.eval();position_actor.requires_grad_(False);coordinator.position_actor=position_actor
    if settings.get('second_jump_checkpoint'):
        import hashlib
        path=settings['second_jump_checkpoint']
        assert hashlib.sha256(open(path,'rb').read()).hexdigest()==settings['second_jump_sha256']
        second_runner=MjlabOnPolicyRunner(wrapper,asdict(agent),log_dir=None,device=env.device)
        second_runner.load(path)
        assert torch.equal(second_runner.alg.actor.jump_prior.physical_units,runner.alg.actor.jump_prior.physical_units)
        coordinator.second_jump_actor=second_runner.alg.actor
        coordinator.second_jump_policy=second_runner.get_inference_policy(device=env.device)
        # 保持原action adapter fresh第一Actor缓存；coord30=0，第二头的真实29映射由coordinator当前Actor完成。
    return wrapper,runner,coordinator