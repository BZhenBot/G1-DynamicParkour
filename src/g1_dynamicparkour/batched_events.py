"""Batched physical jump FSM, matching the audited scalar reference; no simulation writes."""
import torch
from g1_dynamicparkour.events import TIME_COMPARISON_EPS_S

PHASES = ("reset","start_supported","takeoff","airborne","target_contact","stable_success","failed")
FAILURES = (None,"illegal_contact","fallen","other_support","target_without_airborne",
    "premature_target_contact","returned_to_start","mixed_start_target_landing","returned_to_start_after_target")

class BatchedJumpDetector:
    def __init__(self,num_envs,device,minimum_air_time=.08,stable_hold=2.,minimum_rise=.03):
        self.num_envs,self.device=num_envs,device
        self.minimum_air_time,self.stable_hold,self.minimum_rise=minimum_air_time,stable_hold,minimum_rise
        self.phase=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.failure=torch.zeros_like(self.phase)
        self.events=torch.zeros((num_envs,len(PHASES)),device=device,dtype=torch.bool)
        # Double precision preserves the reference's elapsed-time threshold semantics.
        self.support_since=torch.full((num_envs,),float("nan"),device=device,dtype=torch.float64)
        self.support_height=self.support_since.clone()
        self.air_start=self.support_since.clone()
        self.stable_start=self.support_since.clone()
        self.last_time=torch.full_like(self.support_since,-float("inf"))

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.phase[ids]=0
        self.failure[ids]=0
        self.events[ids]=False
        for value in (self.support_since,self.support_height,self.air_start,self.stable_start):
            value[ids]=float("nan")
        self.last_time[ids]=-float("inf")

    def _transition(self,mask,phase):
        self.phase[mask]=phase
        self.events[mask,phase]=True

    def _fail(self,mask,reason):
        self.failure[mask]=reason
        self._transition(mask,6)

    def update(self,time_s,start,target,height,vz,stable,illegal=None,fallen=None,other=None):
        t=torch.as_tensor(time_s,device=self.device,dtype=torch.float64)
        height=torch.as_tensor(height,device=self.device,dtype=torch.float64)
        vz=torch.as_tensor(vz,device=self.device,dtype=torch.float64)
        start=torch.as_tensor(start,device=self.device,dtype=torch.bool)
        target=torch.as_tensor(target,device=self.device,dtype=torch.bool)
        stable=torch.as_tensor(stable,device=self.device,dtype=torch.bool)
        if bool((t<=self.last_time).any()):
            raise ValueError("episode 内样本时间必须严格递增，reset 后才允许重新计时")
        self.last_time[:]=t
        self.events.zero_()
        zero=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        illegal=zero if illegal is None else torch.as_tensor(illegal,device=self.device,dtype=torch.bool)
        fallen=zero if fallen is None else torch.as_tensor(fallen,device=self.device,dtype=torch.bool)
        other=zero if other is None else torch.as_tensor(other,device=self.device,dtype=torch.bool)
        active=self.phase<5
        self._fail(active&illegal,1)
        self._fail(active&~illegal&fallen,2)
        self._fail(active&~illegal&~fallen&other,3)
        before=self.phase.clone()
        start_any,start_all=start.any(dim=-1),start.all(dim=-1)
        target_any,target_all=target.any(dim=-1),target.all(dim=-1)
        contact=start_any|target_any
        reset=(before==0)
        begin=reset&start_all&torch.isnan(self.support_since)
        self.support_since[begin]=t[begin]
        self.support_since[reset&~start_all]=float("nan")
        supported=reset&start_all&((t-self.support_since+TIME_COMPARISON_EPS_S)>=.1)
        self.support_height[supported]=height[supported]
        self._transition(supported,1)
        on_start=(before==1)
        crouch=on_start&start_all
        self.support_height[crouch]=height[crouch]
        self._fail(on_start&target_any,4)
        takeoff=on_start&~contact&(vz>.1)
        self.air_start[takeoff]=t[takeoff]
        self._transition(takeoff,2)
        taking_off=(before==2)
        self._fail(taking_off&target_any,5)
        short_contact=taking_off&~target_any&contact
        self.air_start[short_contact]=float("nan")
        self._transition(short_contact,1)
        airborne=taking_off&~contact&((t-self.air_start+TIME_COMPARISON_EPS_S)>=self.minimum_air_time)&((height-self.support_height)>=self.minimum_rise)
        self._transition(airborne,3)
        flying=(before==3)
        self._fail(flying&target_any&start_any,7)
        self._transition(flying&target_any&~start_any,4)
        self._fail(flying&~target_any&start_any,6)
        landing=(self.phase==4)
        self._fail(landing&start_any,8)
        landing=self.phase==4
        stable_landing=landing&target_all&stable
        begin_stable=stable_landing&torch.isnan(self.stable_start)
        self.stable_start[begin_stable]=t[begin_stable]
        self.stable_start[landing&~stable_landing]=float("nan")
        success=stable_landing&((t-self.stable_start+TIME_COMPARISON_EPS_S)>=self.stable_hold)
        self._transition(success,5)
        return self.events.clone()
