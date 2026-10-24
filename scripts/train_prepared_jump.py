"""只微调15cm既有Jump头适配真实middle准备态；完整路线评估仍禁止bank恢复。"""
import argparse,hashlib,json,os
from pathlib import Path
from dataclasses import replace,asdict
import torch
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactSensorCfg,ContactMatch
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from train_handoff_subtask import HandoffSubtaskEnv,build_handoff_subtask_configs
import train_jump
from g1_dynamicparkour.scene import Geometry,make_scene_cfg,start_spec
from g1_dynamicparkour.course_scene import CourseGeometry
from g1_dynamicparkour.course_turn import MiddleResetBank
from g1_dynamicparkour import prepared_jump as prepared
from v2_budget import enforce_budget

class PreparedJumpEnv(HandoffSubtaskEnv):
    settings=None
    def __init__(self,cfg,directory,geometry,**kwargs):
        self.course=CourseGeometry(**self.settings['course']);self.bank=MiddleResetBank(self.settings['prepared_bank'])
        for path,record in zip(self.bank.paths,self.bank.records):
            assert path.name.endswith('_jump_end.json') and record['time_s']>0
            assert record['course']==asdict(self.course)
        self.bank_indices=None;self.bank_resets=0;self.prepared_goal_y_offset=None
        super().__init__(cfg,directory,geometry,**kwargs)
        self.bank_indices=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
    def _reset_idx(self,env_ids=None):
        super()._reset_idx(env_ids)
        if self.bank_indices is None:return
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        indices=torch.randint(len(self.bank.records),(len(ids),),device=self.device)
        self.bank.restore(self,ids,indices);self.bank_indices[ids]=indices;self.bank_resets+=len(ids)
        if self.settings.get('landing_lane_waypoint',False):
            if self.prepared_goal_y_offset is None:self.prepared_goal_y_offset=torch.zeros(self.num_envs,device=self.device)
            target=self.scene['target'];robot=self.scene['robot']
            delta=robot.data.root_link_pos_w[ids,1]-target.data.root_link_pos_w[ids,1]-target.data.joint_pos[ids,0]
            maximum=self.settings['maximum_landing_lane_offset_m']
            assert 0<maximum<=self.course.half_width-.17
            self.prepared_goal_y_offset[ids]=delta.clamp(-maximum,maximum)
        # 新第二跳episode从phaseRESET开始，不能继承第一跳AIR或稳定奖励。
        from g1_dynamicparkour import jump
        assert not jump.state(self).event_counts[ids].any()
    def episode_details(self,env_id):
        d=super().episode_details(env_id)
        from g1_dynamicparkour import handoff_subtask
        teacher=handoff_subtask.state(self).actor
        if teacher is not None:
            d['handoff_teacher_speeds']={'horizontal_mps':float(torch.linalg.vector_norm(teacher[env_id,102:104])),
                'vertical_mps':abs(float(teacher[env_id,104])),'angular_radps':float(torch.linalg.vector_norm(teacher[env_id,3:6]))}
        d['handoff_velocity_contract']=self.settings.get('handoff_velocity_contract')
        if self.settings.get('handoff_maximum_backward_capture_offset_m') is not None:
            values=prepared.capture_inputs(self);d['handoff_capture_proxy']={k:float(v[env_id]) if torch.is_tensor(v) else v for k,v in values.items()}
            d['handoff_capture_proxy'].update(offset_m=float(prepared.capture_offset(self)[env_id]),maximum_offset_m=self.settings['handoff_maximum_backward_capture_offset_m'],root_link_is_com_proxy=True)
        d.update(task='prepared_middle_jump_to_handoff_only',full_route_success=False,prepared_middle_origin=True,
            snapshot_index=int(self.bank_indices[env_id]) if self.bank_indices is not None else None,
            subtask_requires='真实middle源支持→新AIR→end真实接触→公开ready；不继承第一跳事件，不计完整路线')
        return d

def build_prepared_configs(settings):
    cfg,agent,g=build_handoff_subtask_configs(settings);course=CourseGeometry(**settings['course'])
    source_geometry=replace(g,start_center_x=course.middle_x,start_half_length=course.middle_half_length,
        start_half_width=course.half_width,target_half_length=course.end_half_length,target_half_width=course.half_width,
        robot_start_offset_x=-.02,gap=course.gap2)
    assert abs(source_geometry.target_center_x-course.end_x)<1e-12
    scene=make_scene_cfg(source_geometry,num_envs=cfg.scene.num_envs)
    original=Geometry()
    scene.entities['past_start']=EntityCfg(spec_fn=lambda:start_spec(original),
        init_state=EntityCfg.InitialStateCfg(pos=(-.2,0.,-.1)))
    sensors=list(cfg.scene.sensors)
    for sensor in cfg.scene.sensors:
        if isinstance(sensor,ContactSensorCfg) and sensor.secondary is not None and sensor.secondary.entity=='start':
            sensors.append(replace(sensor,name=sensor.name.replace('ground','past_start'),
                secondary=ContactMatch(mode='body',pattern='start_surface',entity='past_start')))
    scene.sensors=tuple(sensors);scene.extent=4.;cfg.scene=scene
    for group in ('actor','handoff_teacher_current'):
        cfg.observations[group].terms['current_unlabelled_foot_contact'].func=prepared.current_contact
    if settings.get('landing_lane_waypoint',False):
        for group in ('actor','critic','handoff_teacher_current'):
            cfg.observations[group].terms['platform_relative_state'].func=prepared.lane_relative_state
            cfg.observations[group].terms['platform_relative_state'].params={}
    cfg.terminations['update_handoff_subtask'].func=prepared.update
    cfg.terminations['past_start_support_or_nonfoot']=TerminationTermCfg(func=prepared.other_failed)
    dt=cfg.decimation*cfg.sim.mujoco.timestep
    cfg.rewards['past_start_failure_event']=RewardTermCfg(func=prepared.failure_event,weight=-50/dt)
    return cfg,agent,source_geometry

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,required=True);a,_=p.parse_known_args()
    settings=json.loads(a.config.read_text());out=Path(os.environ['G1_RUN_DIR']);root=Path(__file__).resolve().parents[1]
    consumed=enforce_budget(root,out,settings['reserved_training_seconds']);PreparedJumpEnv.settings=settings
    (out/'prepared_curriculum_manifest.json').write_text(json.dumps({'scope':'只训练第二跳，bank只在episode reset；完整路线回放不恢复',
        'bank_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in settings['prepared_bank']},
        'original_source_retained_as_past_start':True,'actor114':'原布局+3表面无身份接触，当前end位姿/速度；无未来/支持面ID',
        'critic130':'新第二跳当前phase/实际gap/A/f/phase，当前特权字段保持明示','new_episode_clock':0,'platform_clock':'恢复真实记录的当前t/phase/q/qvel，只在reset',
        'inherited_airborne_reward':False},ensure_ascii=False,indent=2))
    train_jump.build_jump_configs=build_prepared_configs;train_jump.JumpEnv=PreparedJumpEnv;train_jump.main()
    mp=out/'training_manifest.json';m=json.loads(mp.read_text());m.update(elapsed_training_seconds=m['elapsed_seconds_monotonic'],
        total_v2_training_seconds_after=consumed+m['elapsed_seconds_monotonic'],budget_seconds=21600,
        full_route_successes='不适用：只训练真实prepared-middle第二跳子任务')
    mp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    summary={'episodes':len(rows),'qualified_subtask_successes':sum(r['subtask_success'] for r in rows),
        'full_route_successes':'不适用：第二跳训练reset课程，不是完整路线'}
    (out/'prepared_jump_training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)