"""参数冻结前的物理事件判定参考实现；使用真实接触样本接入独立评估。"""
from dataclasses import dataclass
from enum import Enum


# One picosecond corrects float64 subtraction at exact decimal time boundaries;
# it is 20 billion times smaller than the20ms control sample, not a physics relaxation.
TIME_COMPARISON_EPS_S = 1e-12

class Phase(str, Enum):
    RESET = "reset"
    START_SUPPORTED = "start_supported"
    TAKEOFF = "takeoff"
    AIRBORNE = "airborne"
    TARGET_CONTACT = "target_contact"
    STABLE_SUCCESS = "stable_success"
    FAILED = "failed"


@dataclass(frozen=True)
class Sample:
    time_s: float
    start_feet: tuple[bool, bool]
    target_feet: tuple[bool, bool]
    root_height_m: float
    vertical_velocity_mps: float
    stable: bool
    illegal_contact: bool = False
    fallen: bool = False
    foot_other_support: bool = False


class JumpEventDetector:
    """接触布尔值须由真实接触力与支撑面 ID 提供，不从脚位置猜接触。"""
    def __init__(self, minimum_air_time=0.08, stable_hold=2.0, minimum_rise=0.03):
        self.minimum_air_time = minimum_air_time
        self.stable_hold = stable_hold
        self.minimum_rise = minimum_rise
        self.reset()

    def reset(self):
        self.phase = Phase.RESET
        self.support_height = None
        self.air_start = None
        self.stable_start = None
        self.last_time = None
        self.events = []
        self.failure_reason = None
        self.start_support_since = None

    def _transition(self, phase, time):
        self.phase = phase
        self.events.append({"phase": phase.value, "time_s": time})
        return phase.value

    def update(self, sample):
        if self.last_time is not None and sample.time_s <= self.last_time:
            raise ValueError("episode 内样本时间必须严格递增，reset 后才允许重新计时")
        self.last_time = sample.time_s
        if self.phase in (Phase.FAILED, Phase.STABLE_SUCCESS):
            return []
        if sample.illegal_contact or sample.fallen or sample.foot_other_support:
            self.failure_reason = "illegal_contact" if sample.illegal_contact else "fallen" if sample.fallen else "other_support"
            return [self._transition(Phase.FAILED, sample.time_s)]
        changes = []
        contact = any(sample.start_feet) or any(sample.target_feet)
        if self.phase == Phase.RESET:
            if all(sample.start_feet):
                if self.start_support_since is None:
                    self.start_support_since = sample.time_s
                if sample.time_s-self.start_support_since + TIME_COMPARISON_EPS_S >= 0.10:
                    self.support_height = sample.root_height_m
                    changes.append(self._transition(Phase.START_SUPPORTED, sample.time_s))
            else:
                self.start_support_since = None
        elif self.phase == Phase.START_SUPPORTED:
            if all(sample.start_feet):
                self.support_height = sample.root_height_m
            if any(sample.target_feet):
                self.failure_reason = "target_without_airborne"
                return [self._transition(Phase.FAILED, sample.time_s)]
            if not contact and sample.vertical_velocity_mps > 0.10:
                self.air_start = sample.time_s
                changes.append(self._transition(Phase.TAKEOFF, sample.time_s))
        elif self.phase == Phase.TAKEOFF:
            if any(sample.target_feet):
                self.failure_reason = "premature_target_contact"
                return [self._transition(Phase.FAILED, sample.time_s)]
            if contact:
                # 短暂离地/接触抖动不足以构成腾空；允许后续真实起跳。
                self.air_start = None
                changes.append(self._transition(Phase.START_SUPPORTED, sample.time_s))
            elif sample.time_s-self.air_start + TIME_COMPARISON_EPS_S >= self.minimum_air_time and sample.root_height_m-self.support_height >= self.minimum_rise:
                changes.append(self._transition(Phase.AIRBORNE, sample.time_s))
        elif self.phase == Phase.AIRBORNE:
            if any(sample.target_feet) and any(sample.start_feet):
                self.failure_reason = "mixed_start_target_landing"
                return [self._transition(Phase.FAILED, sample.time_s)]
            if any(sample.target_feet):
                changes.append(self._transition(Phase.TARGET_CONTACT, sample.time_s))
            elif any(sample.start_feet):
                self.failure_reason = "returned_to_start"
                changes.append(self._transition(Phase.FAILED, sample.time_s))
        if self.phase == Phase.TARGET_CONTACT:
            if any(sample.start_feet):
                self.failure_reason = "returned_to_start_after_target"
                return changes + [self._transition(Phase.FAILED, sample.time_s)]
            if all(sample.target_feet) and sample.stable:
                if self.stable_start is None:
                    self.stable_start = sample.time_s
                if sample.time_s-self.stable_start + TIME_COMPARISON_EPS_S >= self.stable_hold:
                    changes.append(self._transition(Phase.STABLE_SUCCESS, sample.time_s))
            else:
                self.stable_start = None
        return changes
