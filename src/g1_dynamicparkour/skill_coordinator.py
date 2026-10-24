"""仅公开当前观测与历史的双跳协调器，物理状态不写入。"""
import math
import torch
from tensordict import TensorDict
from .course_scene import SkillObservationAdapter

class SkillCoordinator:
    def __init__(self,env,jump_policy,jump_actor,recovery_actor,settings=None):
        if env.num_envs!=1: raise ValueError('初版先单环境完整路线部署')
        self.env=env; self.jump_policy=jump_policy; self.jump_actor=jump_actor; self.recovery=recovery_actor
        self.end_recovery_actor=None; self.settle_recovery_actor=None; self.second_jump_policy=None; self.second_jump_actor=None; self.position_actor=None
        self.adapter=SkillObservationAdapter(env); self.settings=settings or {}; self.alignment_actor=None; self.alignment_kind='walking99'
        from .symmetry_projection import SagittalReflection
        self.reflection=SagittalReflection(env.action_manager.get_term('joint_pos').target_names,env.device)
        from .command_continuity import JointCommandContinuity
        self.command_continuity=JointCommandContinuity(env.step_dt,self.settings)
        self.final_policy_blend_seconds=float(self.settings.get('prepared_final_policy_blend_seconds',0.))
        if not math.isfinite(self.final_policy_blend_seconds) or not 0<=self.final_policy_blend_seconds<=.6:raise ValueError('当前策略融合时长须在0..600ms')
        self.reset()
    def reset(self):
        self.command_continuity.reset()
        self.final_settle_started=None
        self.leg=0; self.mode='jump'; self.target='middle'; self.started=0.; self.steps=0
        self.held=None; self.support_since=None; self.air_since=None; self.public_air=False
        self.support_height=None; self.stable_since=None; self.aligned_since=None; self.positioning_done=False; self.second_head_selected=None; self.trace=[]
        self.env.course_target_id='middle'; self.env.course_skill_start_s=0.;self.env.landing_goal_y_offset=0.
    def update_public_history(self,actor,feet,t):
        support=bool(actor[:,106:108].gt(.5).any())
        dual=bool(actor[:,106:108].gt(.5).all())
        r=self.env.scene['robot']; height=float(r.data.root_link_pos_w[0,2]); vz=float(r.data.root_link_lin_vel_w[0,2])
        if support:
            self.air_since=None
            if dual: self.support_height=height
        elif self.support_height is not None:
            if self.air_since is None and vz>.1: self.air_since=t
            if self.air_since is not None and t-self.air_since+1e-12>=.08 and height-self.support_height>=.03:
                self.public_air=True
    def ready(self,actor,feet,target):
        center,velocity,half=self.adapter.provider.current(target); f=feet.reshape(1,2,3)
        capture_limit=self.settings.get('handoff_maximum_backward_capture_offset_m')
        if self.leg==1 and capture_limit is not None:
            from .capture_proxy import backward_capture_offset
            if not 0<capture_limit<=.2:raise ValueError('近似捕获偏差候选阈值无效')
            r=self.env.scene['robot'];height=r.data.root_link_pos_w[:,2]-center[:,2]
            offset=backward_capture_offset(r.data.root_link_pos_w[:,0]-center[:,0],
                r.data.root_link_lin_vel_w[:,0]-velocity[:,0],height,f[:,:,0].mean(-1),
                abs(float(self.env.sim.mj_model.opt.gravity[2])))
            if not bool(torch.isfinite(offset).all() and offset.le(capture_limit).all()):return False
        limit=self.settings.get('handoff_velocity_contract')
        if limit and self.leg==1:
            if not bool(torch.linalg.vector_norm(actor[:,102:104],dim=-1).le(limit['horizontal_speed_mps']).all()
                and actor[:,104].abs().le(limit['vertical_speed_mps']).all()
                and torch.linalg.vector_norm(actor[:,3:6],dim=-1).le(limit['angular_speed_radps']).all()):return False
        return bool(torch.isfinite(actor).all() and torch.isfinite(f).all()
            and actor[:,106:108].gt(.5).all() and (f[:,:,:2].abs()<=half-.02).all()
            and ((f[:,:,2]>=-.01)&(f[:,:,2]<=self.settings.get('handoff_maximum_foot_height_m',.04))).all()
            and (-actor[:,8]>=math.cos(math.radians(45))).all() and (-actor[:,101]>=.5).all()
            and (actor[:,105]>=.8).all() and self.public_air)
    def stable(self,actor,feet,target):
        _,_,half=self.adapter.provider.current(target); f=feet.reshape(1,2,3)
        return bool(actor[:,106:108].gt(.5).all() and (f[:,:,:2].abs()<=half-.02).all()
            and ((f[:,:,2]>=-.01)&(f[:,:,2]<=.04)).all() and (-actor[:,8]>=math.cos(math.radians(15))).all()
            and torch.linalg.vector_norm(actor[:,3:6],dim=-1).le(.5).all()
            and torch.linalg.vector_norm(actor[:,102:104],dim=-1).le(.15).all()
            and actor[:,104].abs().le(.1).all() and (-actor[:,101]>=.5).all())
    def __call__(self,obs):
        t=float(self.env.episode_length_buf[0])*self.env.step_dt
        actor,feet=self.adapter.for_skill(obs,self.target,self.started)
        self.update_public_history(actor,feet,t)
        transition=None;self.last_preparation_features=None;self.last_preparation_command=None;self.last_final_blend_weight=None
        if self.mode=='jump' and self.ready(actor,feet,self.target):
            self.mode='recovery'; self.stable_since=None; transition='recover_'+self.target
        if self.mode=='recovery':
            stable=self.stable(actor,feet,self.target)
            self.stable_since=(t if self.stable_since is None else self.stable_since) if stable else None
            if self.leg==0 and self.stable_since is not None and t-self.stable_since>=self.settings.get('middle_stable_seconds',2.)+self.settings.get('middle_recovery_guard_seconds',.12):
                if self.alignment_actor is not None:
                    self.mode='align'; self.aligned_since=None; transition='align_middle'
                else:
                    self.begin_second_jump(t);actor,feet=self.adapter.for_skill(obs,self.target,self.started);transition='jump_end'
        if self.mode=='align' and self.settings.get('simultaneous_launch_preparation',False):
            # 当前反馈同时转向与定位；只决定控制技能，不读取独立验收FSM。
            r=self.env.scene['robot'];center,_,half=self.adapter.provider.current('middle')
            point=center[0,0]+self.settings.get('launch_point_from_middle_center_m',.2)
            position_error=abs(float(point-r.data.root_link_pos_w[0,0]))
            lane_safe=abs(float(center[0,1]-r.data.root_link_pos_w[0,1]))<=self.settings.get('maximum_launch_lateral_offset_m',.25)
            heading=float(r.data.heading_w[0]);f=feet.reshape(1,2,3)
            square=float((f[:,0,0]-f[:,1,0]).abs())<=self.settings.get('launch_stance_x_difference_m',.10)
            supported=bool(actor[:,106:108].gt(.5).all() and (f[:,:,:2].abs()<=half-.02).all())
            early_capture=self.settings.get('capture_alignment_before_position',False)
            direct_hold=self.settings.get('prepared_policy_launch_hold',False)
            if early_capture and not direct_hold and lane_safe and supported and abs(heading)<=self.settings.get('launch_heading_bound_rad',.35) and bool(((f[:,:,2]>=-.01)&(f[:,:,2]<=.06)).all()) and bool((-actor[:,8]>=math.cos(math.radians(45))).all()):
                self.mode='position';self.stable_since=None;transition='position_middle'
            capture_heading=self.settings.get('prepared_capture_heading_bound_rad',self.settings.get('launch_heading_bound_rad',.35))
            capture_stable=not self.settings.get('prepared_capture_requires_stability',False) or self.stable(actor,feet,'middle')
            if self.mode=='align' and direct_hold:
                if self.alignment_kind!='prepared103':raise ValueError('持续准备策略接管要求Prepared103')
                goal=center[0,:2].clone();goal[0]+=self.settings.get('launch_point_from_middle_center_m',.2)
                error_xy=float(torch.linalg.vector_norm(goal-r.data.root_link_pos_w[0,:2]))
                good=lane_safe and supported and square and error_xy<=self.settings.get('launch_position_tolerance_m',.04) and abs(heading)<=capture_heading and self.stable(actor,feet,'middle')
                good &= bool(torch.linalg.vector_norm(actor[:,102:104],dim=-1).le(self.settings.get('launch_horizontal_speed_limit_mps',.15)).all())
                good &= bool(torch.linalg.vector_norm(actor[:,3:6],dim=-1).le(self.settings.get('launch_angular_speed_limit_radps',.5)).all())
                if self.settings.get('launch_leg_joint_speed_limit_radps') is not None:
                    good &= bool(r.data.joint_vel[:,:12].abs().le(self.settings['launch_leg_joint_speed_limit_radps']).all())
                self.aligned_since=(t if self.aligned_since is None else self.aligned_since) if good else None
                if self.aligned_since is not None and t-self.aligned_since>=.20:
                    self.positioning_done=True;self.begin_second_jump(t)
                    actor,feet=self.adapter.for_skill(obs,self.target,self.started);transition='jump_end'
            elif self.mode=='align' and lane_safe and supported and square and position_error<=self.settings.get('launch_position_tolerance_m',.04) and abs(heading)<=capture_heading and capture_stable:
                self.mode='final_settle';self.final_settle_started=t;self.stable_since=None;transition='final_settle_middle'
        if self.mode=='align' and not self.settings.get('simultaneous_launch_preparation',False):
            heading=float(self.env.scene['robot'].data.heading_w[0])
            error=math.atan2(math.sin(-heading),math.cos(-heading))
            if self.alignment_kind=='recovery114':
                goal_actor,_=self.adapter.for_skill(obs,'end',-1.)
                error=math.atan2(float(goal_actor[0,100]),float(goal_actor[0,99]))
            if self.settings.get('brake_after_alignment',False):
                _,_,half=self.adapter.provider.current('middle');f=feet.reshape(1,2,3)
                capture=abs(error)<=self.settings.get('alignment_capture_rad',.15)
                capture &= bool(actor[:,106:108].gt(.5).all() and (f[:,:,:2].abs()<=half-.02).all()
                    and ((f[:,:,2]>=-.01)&(f[:,:,2]<=.06)).all()
                    and (-actor[:,8]>=math.cos(math.radians(45))).all() and (-actor[:,101]>=.5).all())
                if capture:self.mode='settle';self.stable_since=None;transition='settle_middle'
            else:
                if abs(error)<=self.settings.get('alignment_tolerance_rad',.10) and self.stable(actor,feet,'middle'):
                    self.aligned_since=t if self.aligned_since is None else self.aligned_since
                else:self.aligned_since=None
                if self.aligned_since is not None and t-self.aligned_since>=.20:
                    self.begin_second_jump(t);actor,feet=self.adapter.for_skill(obs,self.target,self.started);transition='jump_end'
        if self.mode=='settle':
            heading=float(self.env.scene['robot'].data.heading_w[0])
            error=math.atan2(math.sin(-heading),math.cos(-heading))
            stable=self.stable(actor,feet,'middle') and abs(error)<=self.settings.get('launch_heading_bound_rad',.35)
            self.stable_since=(t if self.stable_since is None else self.stable_since) if stable else None
            if self.stable_since is not None and t-self.stable_since>=.20:
                if self.settings.get('position_before_second_jump',False):
                    self.mode='position';self.stable_since=None;transition='position_middle'
                else:
                    self.begin_second_jump(t);actor,feet=self.adapter.for_skill(obs,self.target,self.started);transition='jump_end'
        if self.mode in ('position','final_settle'):
            r=self.env.scene['robot'];center,_,half=self.adapter.provider.current('middle')
            point=center[0,:2].clone();point[0]+=self.settings.get('launch_point_from_middle_center_m',.1)
            position_error=float(torch.linalg.vector_norm(point-r.data.root_link_pos_w[0,:2]))
            if self.settings.get('lane_preserving_preparation',False):
                position_error=abs(float(point[0]-r.data.root_link_pos_w[0,0]))
                lane_safe=abs(float(center[0,1]-r.data.root_link_pos_w[0,1]))<=self.settings.get('maximum_launch_lateral_offset_m',.25)
                if not lane_safe:position_error=float('inf')
            h=float(r.data.heading_w[0]);f=feet.reshape(1,2,3)
            square=float((f[:,0,0]-f[:,1,0]).abs())<=self.settings.get('launch_stance_x_difference_m',.10)
            if self.mode=='position' and position_error<=self.settings.get('launch_position_tolerance_m',.04) and abs(h)<=self.settings.get('position_heading_bound_rad',.20) and square:
                self.mode='final_settle';self.final_settle_started=t;self.stable_since=None;transition='final_settle_middle'
            if self.mode=='final_settle':
                stable=self.stable(actor,feet,'middle') and abs(h)<=self.settings.get('launch_heading_bound_rad',.35) and position_error<=.08 and square
                if self.settings.get('launch_horizontal_speed_limit_mps') is not None:
                    stable &= bool(torch.linalg.vector_norm(actor[:,102:104],dim=-1).le(self.settings['launch_horizontal_speed_limit_mps']).all())
                if self.settings.get('launch_angular_speed_limit_radps') is not None:
                    stable &= bool(torch.linalg.vector_norm(actor[:,3:6],dim=-1).le(self.settings['launch_angular_speed_limit_radps']).all())
                if self.settings.get('launch_leg_joint_speed_limit_radps') is not None:
                    stable &= bool(self.env.scene['robot'].data.joint_vel[:,:12].abs().le(self.settings['launch_leg_joint_speed_limit_radps']).all())
                self.stable_since=(t if self.stable_since is None else self.stable_since) if stable else None
                if self.stable_since is not None and t-self.stable_since>=.20:
                    self.positioning_done=True;self.begin_second_jump(t)
                    actor,feet=self.adapter.for_skill(obs,self.target,self.started);transition='jump_end'
        if self.mode=='recovery' and self.settings.get('recovery_uses_platform_center',False):
            # Recovery原114接口是当前真实平台中心，Jump的lane指令不能继续当其平台位姿。
            actor[:,99:105]=self.adapter.provider.relative(self.target)
        only=TensorDict({'actor':actor},batch_size=obs.batch_size)
        source=self.jump_policy(only)
        mapping_actor=self.jump_actor
        active_policy=self.jump_policy
        use_second=self.leg==1 and self.second_jump_policy is not None
        if use_second and self.settings.get('short_head_if_target_distance_below_m') is not None:
            if self.second_head_selected is None:
                self.second_head_selected=float(torch.linalg.vector_norm(actor[0,99:101]))>self.settings['short_head_if_target_distance_below_m']
            use_second=self.second_head_selected
        if use_second:
            active_policy=self.second_jump_policy;mapping_actor=self.second_jump_actor
            source=active_policy(only)
        mapped=(source[:,:29].clamp(-1,1)+source[:,29:30].clamp(-1,1)*mapping_actor.last_direction).clamp(-1,1)
        if self.settings.get('symmetry_projection',False):
            reflected=TensorDict({'actor':self.reflection.observation(actor)},batch_size=obs.batch_size)
            other=active_policy(reflected)
            other_mapped=(other[:,:29].clamp(-1,1)+other[:,29:30].clamp(-1,1)*mapping_actor.last_direction).clamp(-1,1)
            weight=float(self.settings.get('symmetry_weight',1.))
            if not 0.<=weight<=1.:raise ValueError('反射先验权重必须在[0,1]')
            mapped=((1.-.5*weight)*mapped+.5*weight*self.reflection.action(other_mapped)).clamp(-1,1)
            # 恢复真实当前Actor方向缓存；最终coord30=0，适配器仍每管理步有fresh forward。
            self.jump_policy(only)
        if self.held is None or self.steps%2==0: self.held=mapped.clone()
        action=torch.zeros_like(source)
        if self.mode=='jump':action[:,:29]=self.held
        elif self.mode in ('align','position') or (self.mode=='final_settle' and self.alignment_kind=='prepared103' and self.final_policy_blend_seconds>0):
            proprio=actor[:,:99].clone();r=self.env.scene['robot'];h=r.data.heading_w
            center,_,_=self.adapter.provider.current('middle');desired=center[:,:2].clone();desired[:,0]+=(self.settings.get('launch_point_from_middle_center_m',.1) if self.mode=='position' or self.settings.get('simultaneous_launch_preparation',False) else .1)
            gain=self.settings.get('position_feedback_gain',.8) if self.mode=='position' or self.settings.get('simultaneous_launch_preparation',False) else .8
            if not 0<gain<=2.:raise ValueError('位置反馈增益超出候选范围')
            velocity=(desired-r.data.root_link_pos_w[:,:2])*gain
            if self.mode=='align' and self.settings.get('alignment_yaw_only',False):velocity=torch.zeros_like(velocity)
            max_speed=self.settings.get('position_maximum_linear_speed_mps',self.settings.get('preparation_maximum_linear_speed_mps',.10)) if self.mode=='position' else self.settings.get('preparation_maximum_linear_speed_mps',.10)
            if not 0<max_speed<=(.3 if self.mode=='position' or self.alignment_kind=='prepared103' else .2):raise ValueError('准备阶段线速度指令越界')
            velocity=velocity.clamp(-max_speed,max_speed)
            proprio[:,96]=h.cos()*velocity[:,0]+h.sin()*velocity[:,1]
            proprio[:,97]=-h.sin()*velocity[:,0]+h.cos()*velocity[:,1]
            error=torch.atan2(torch.sin(-h),torch.cos(-h))
            max_turn=self.settings.get('preparation_maximum_turn_speed_radps',.5)
            if not 0<max_turn<=1.2:raise ValueError('准备阶段转速指令越界')
            proprio[:,98]=(error*1.5).clamp(-max_turn,max_turn)
            if self.mode=='align' and self.alignment_kind=='recovery114':
                goal_actor,_=self.adapter.for_skill(obs,'end',-1.)
                turn=TensorDict({'actor':goal_actor},batch_size=[1])
                action[:,:29]=self.alignment_actor(turn).clamp(-1,1)
            else:
                if self.alignment_kind=='prepared103' and self.mode in ('align','final_settle'):
                    from .preparation_observation import measured_launch_goal_features
                    features=measured_launch_goal_features(self.env,self.settings.get('launch_point_from_middle_center_m',.2),preserve_landing_lane=self.settings.get('preparation_preserve_landing_lane',False))
                    if self.settings.get('consistent_feedback_command',False):
                        from .preparation_command import feedback_prepare_command
                        command,_=feedback_prepare_command(self.env,offset_m=self.settings.get('launch_point_from_middle_center_m',.2),linear_gain=self.settings.get('position_feedback_gain',.8),linear_limit=self.settings.get('preparation_maximum_linear_speed_mps',.1),yaw_limit=self.settings.get('preparation_maximum_turn_speed_radps',.5),preserve_landing_lane=self.settings.get('preparation_preserve_landing_lane',False))
                        proprio[:,96:99]=command
                    self.last_preparation_features=features.clone();self.last_preparation_command=proprio[:,96:99].clone()
                    proprio=torch.cat((proprio,features),-1)
                turn=TensorDict({'actor':proprio},batch_size=[1])
                policy=self.position_actor if self.mode=='position' and self.position_actor is not None else self.alignment_actor
                action[:,:29]=policy(turn).clamp(-1,1)/self.jump_actor.jump_prior.physical_units
                if self.mode=='final_settle' and self.alignment_kind=='prepared103' and self.final_policy_blend_seconds>0:
                    assert self.final_settle_started is not None
                    weight=min(1.,max(0.,(t-self.final_settle_started+self.env.step_dt)/self.final_policy_blend_seconds))
                    recovery_action=self.recovery(only).clamp(-1,1)
                    action[:,:29]=(1.-weight)*action[:,:29]+weight*recovery_action
                    self.last_final_blend_weight=weight
        elif self.mode=='final_settle' and self.settings.get('prepared_final_settle_kind') in ('position99_zero_command','standing99_zero_command'):
            if self.position_actor is None:raise ValueError('静态停止接管缺少既有Position99策略')
            proprio=actor[:,:99].clone();proprio[:,96:99]=0.
            stop_obs=TensorDict({'actor':proprio},batch_size=obs.batch_size)
            stop_policy=self.position_actor if self.settings['prepared_final_settle_kind']=='position99_zero_command' else self.jump_actor.jump_prior.standing_prior
            action[:,:29]=stop_policy(stop_obs).clamp(-1,1)/self.jump_actor.jump_prior.physical_units
        elif self.mode=='final_settle' and self.settle_recovery_actor is not None:
            action[:,:29]=self.settle_recovery_actor(only).clamp(-1,1)
        elif self.mode=='recovery' and self.leg==1 and self.settings.get('end_recovery_kind')=='standing99':
            proprio=TensorDict({'actor':actor[:,:99]},batch_size=obs.batch_size)
            action[:,:29]=self.jump_actor.jump_prior.standing_prior(proprio).clamp(-1,1)/self.jump_actor.jump_prior.physical_units
        elif self.mode=='recovery' and self.leg==1 and self.end_recovery_actor is not None:
            action[:,:29]=self.end_recovery_actor(only).clamp(-1,1)
        else:action[:,:29]=self.recovery(only).clamp(-1,1)
        filter_mode='align' if self.last_final_blend_weight is not None else self.mode
        action[:,:29]=self.command_continuity(action[:,:29],filter_mode,self.leg)
        self.last_actor=actor.clone(); self.last_feet=feet.clone(); self.last_transition=transition
        self.last_source_updated=self.mode=='jump' and self.steps%2==0
        self.steps+=1
        if transition: self.trace.append({'time_s':t,'transition':transition,'leg':self.leg})
        return action
    def begin_second_jump(self,t):
        self.leg=1; self.mode='jump'; self.target='end'; self.started=t; self.steps=0; self.held=None
        self.public_air=False; self.air_since=None; self.support_height=None; self.stable_since=None
        self.env.course_target_id='end'; self.env.course_skill_start_s=t
        if self.settings.get('landing_lane_waypoint',False):
            center,_,half=self.adapter.provider.current('end')
            delta=float(self.env.scene['robot'].data.root_link_pos_w[0,1]-center[0,1])
            maximum=self.settings.get('maximum_landing_lane_offset_m',.30)
            if maximum>float(half[1])-.17:raise ValueError('着陆点须给双足保留实际边界裕度')
            self.env.landing_goal_y_offset=max(-maximum,min(maximum,delta))