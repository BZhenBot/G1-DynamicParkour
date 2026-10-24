"""独立完整路线事件判定：不读取技能协调器或训练reward。"""
from dataclasses import dataclass
from .events import JumpEventDetector, Sample, Phase

@dataclass(frozen=True)
class CourseSample:
    time_s: float
    supports: dict
    root_height_m: float
    vertical_velocity_mps: float
    stable: dict
    illegal_contact: bool=False
    fallen: bool=False

class CourseEventDetector:
    def __init__(self):
        self.leg=0; self.detectors=[JumpEventDetector(),JumpEventDetector()]
        self.events=[]; self.failed=False; self.failure_reason=None; self.success=False; self.last_time=None
    def update(self,sample):
        if self.last_time is not None and sample.time_s<=self.last_time: raise ValueError('全路线时间必须递增')
        self.last_time=sample.time_s
        if sample.illegal_contact or sample.fallen:
            self.failed=True; self.success=False; self.failure_reason='illegal_contact' if sample.illegal_contact else 'fallen'
            return []
        if self.failed or self.success: return []
        source,target=(('start','middle'),('middle','end'))[self.leg]
        other=next(name for name in ('start','middle','end') if name not in (source,target))
        s=Sample(sample.time_s,tuple(sample.supports[source]),tuple(sample.supports[target]),
            sample.root_height_m,sample.vertical_velocity_mps,sample.stable[target],
            foot_other_support=any(sample.supports[other]))
        detector=self.detectors[self.leg]; changed=detector.update(s)
        for phase in changed: self.events.append({'leg_id':self.leg,'source':source,'target':target,'phase':phase,'time_s':sample.time_s})
        if detector.phase==Phase.FAILED:
            self.failed=True; self.failure_reason=f'leg{self.leg+1}:'+str(detector.failure_reason)
        elif detector.phase==Phase.STABLE_SUCCESS:
            if self.leg==0: self.leg=1
            else: self.success=True
        return changed