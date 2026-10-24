"""部署端因果命令连续化；只处理29关节命令，不访问或改写仿真状态。"""
import math
import torch

class JointCommandContinuity:
    def __init__(self,dt,settings):
        self.dt=float(dt);self.seconds=float(settings.get('command_switch_blend_seconds',0.))
        self.alpha=float(settings.get('preparation_command_ema_alpha',1.))
        if not math.isfinite(self.seconds) or not 0<=self.seconds<=.12:raise ValueError('技能切换命令渐变只允许0..120ms')
        if not math.isfinite(self.alpha) or not 0<self.alpha<=1:raise ValueError('准备命令EMA系数须在(0,1]')
        self.reset()
    def reset(self):
        self.previous=None;self.key=None;self.origin=None;self.elapsed=None
    def __call__(self,raw,mode,leg):
        assert raw.shape[-1]==29 and torch.isfinite(raw).all()
        key=(mode,leg);command=raw.clone()
        if self.previous is not None:
            if mode in ('align','position') and self.alpha<1:
                command=self.alpha*command+(1-self.alpha)*self.previous
            if key!=self.key:
                self.origin=self.previous.clone();self.elapsed=0.
            if self.origin is not None and mode in ('align','settle','position','final_settle','jump') and self.seconds>0:
                self.elapsed+=self.dt;weight=min(1.,self.elapsed/self.seconds)
                command=(1-weight)*self.origin+weight*command
                if weight>=1:self.origin=None
            elif key!=self.key:self.origin=None
        self.key=key;self.previous=command.clone()
        return command
