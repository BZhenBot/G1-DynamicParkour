"""事件逻辑回归，合成输入只用于验证判定逻辑，不能证明机器人任务成功。"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from g1_dynamicparkour.events import JumpEventDetector, Phase, Sample


def sample(t, start=(False,False), target=(False,False), height=0.76, vz=0.0, stable=False, **kwargs):
    return Sample(t, start, target, height, vz, stable, **kwargs)


class EventTests(unittest.TestCase):
    def supported(self):
        detector = JumpEventDetector()
        detector.update(sample(0, start=(True,True)))
        detector.update(sample(.11, start=(True,True)))
        self.assertEqual(detector.phase, Phase.START_SUPPORTED)
        return detector

    def airborne(self):
        detector = self.supported()
        detector.update(sample(.20, height=.77, vz=.5))
        detector.update(sample(.29, height=.81, vz=.3))
        self.assertEqual(detector.phase, Phase.AIRBORNE)
        return detector

    def test_contact_without_jump_is_not_success(self):
        detector = self.supported()
        detector.update(sample(.2, target=(True,True), stable=True))
        detector.update(sample(3.0, target=(True,True), stable=True))
        self.assertNotEqual(detector.phase, Phase.STABLE_SUCCESS)

    def test_short_contact_gap_is_not_airborne(self):
        detector = self.supported()
        detector.update(sample(.2, height=.78, vz=.4))
        detector.update(sample(.23, start=(True,True)))
        self.assertEqual(detector.phase, Phase.START_SUPPORTED)

    def test_real_sequence_requires_two_second_hold(self):
        detector = self.airborne()
        detector.update(sample(.5, target=(True,False)))
        detector.update(sample(.6, target=(True,True), stable=True))
        detector.update(sample(2.59, target=(True,True), stable=True))
        self.assertEqual(detector.phase, Phase.TARGET_CONTACT)
        self.assertEqual(detector.update(sample(2.61, target=(True,True), stable=True)), ["stable_success"])
        self.assertEqual(detector.update(sample(3.0, target=(True,True), stable=True)), [])

    def test_stability_gap_restarts_hold(self):
        detector = self.airborne()
        detector.update(sample(.5, target=(True,True), stable=True))
        detector.update(sample(1.6, target=(True,True), stable=False))
        detector.update(sample(1.7, target=(True,True), stable=True))
        detector.update(sample(3.6, target=(True,True), stable=True))
        self.assertEqual(detector.phase, Phase.TARGET_CONTACT)
        detector.update(sample(3.71, target=(True,True), stable=True))
        self.assertEqual(detector.phase, Phase.STABLE_SUCCESS)

    def test_illegal_contact_overrides_success(self):
        detector = self.airborne()
        detector.update(sample(.5, target=(True,True), stable=True, illegal_contact=True))
        self.assertEqual(detector.phase, Phase.FAILED)

    def test_walking_to_target_then_hopping_is_rejected(self):
        detector = self.supported()
        detector.update(sample(.2, target=(True,True)))
        detector.update(sample(.3, height=.8, vz=.5))
        detector.update(sample(.4, height=.83, vz=.2))
        detector.update(sample(.6, target=(True,True), stable=True))
        detector.update(sample(3.0, target=(True,True), stable=True))
        self.assertEqual(detector.phase, Phase.FAILED)
        self.assertEqual(detector.failure_reason, "target_without_airborne")

    def test_contact_before_minimum_airborne_duration_rejected(self):
        detector = self.supported()
        detector.update(sample(.2, height=.78, vz=.4))
        detector.update(sample(.23, target=(True,False)))
        self.assertEqual(detector.phase, Phase.FAILED)
        self.assertEqual(detector.failure_reason, "premature_target_contact")
    def test_crouch_updates_takeoff_height_reference(self):
        detector = self.supported()
        detector.update(sample(.2, start=(True,True), height=.60))
        detector.update(sample(.3, height=.62, vz=.5))
        detector.update(sample(.39, height=.66, vz=.3))
        self.assertEqual(detector.phase, Phase.AIRBORNE)

    def test_mixed_source_target_landing_is_rejected(self):
        detector = self.airborne()
        detector.update(sample(.5, start=(False,True), target=(True,False), stable=True))
        self.assertEqual(detector.phase, Phase.FAILED)
        self.assertEqual(detector.failure_reason, "mixed_start_target_landing")

    def test_return_to_source_after_target_contact_is_rejected(self):
        detector = self.airborne()
        detector.update(sample(.5, target=(True,True), stable=True))
        detector.update(sample(3.0, start=(True,True), target=(True,True), stable=True))
        self.assertEqual(detector.phase, Phase.FAILED)
        self.assertEqual(detector.failure_reason, "returned_to_start_after_target")
    def test_exact_decimal_support_threshold(self):
        detector=JumpEventDetector()
        detector.update(sample(.14,start=(True,True)))
        detector.update(sample(.24,start=(True,True)))
        self.assertEqual(detector.phase,Phase.START_SUPPORTED)

    def test_exact_decimal_air_time_threshold(self):
        detector=self.supported()
        detector.update(sample(.56,height=.77,vz=.5))
        detector.update(sample(.64,height=.82,vz=.3))
        self.assertEqual(detector.phase,Phase.AIRBORNE)

    def test_duration_below_threshold_still_rejected(self):
        detector=JumpEventDetector()
        detector.update(sample(.14,start=(True,True)))
        detector.update(sample(.24-1e-8,start=(True,True)))
        self.assertEqual(detector.phase,Phase.RESET)
        detector=self.airborne()
        detector.update(sample(.6,target=(True,True),stable=True))
        detector.update(sample(2.58,target=(True,True),stable=True))
        self.assertEqual(detector.phase,Phase.TARGET_CONTACT)

    def test_reset_and_monotonic_time(self):
        detector = self.supported()
        with self.assertRaises(ValueError):
            detector.update(sample(.10))
        detector.reset()
        detector.update(sample(0))
        self.assertEqual(detector.phase, Phase.RESET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
