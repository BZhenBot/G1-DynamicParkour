"""GPU FSM parity: original 11 synthetic regressions plus actual external-force contact trace."""
import csv
import json
import os
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from g1_dynamicparkour.events import JumpEventDetector
from g1_dynamicparkour.batched_events import BatchedJumpDetector,PHASES,FAILURES
import test_event_detector

device="cuda:0"
counter=0
def compare(detector,batch,sample,changes):
    global counter
    event=batch.update([sample.time_s],[sample.start_feet],[sample.target_feet],
        [sample.root_height_m],[sample.vertical_velocity_mps],[sample.stable],
        [sample.illegal_contact],[sample.fallen],[sample.foot_other_support])
    actual=[PHASES[i] for i in event[0].nonzero().flatten().tolist()]
    assert actual==changes,(actual,changes)
    assert PHASES[int(batch.phase[0])]==detector.phase.value
    assert FAILURES[int(batch.failure[0])]==detector.failure_reason
    counter+=1

class ParityDetector(JumpEventDetector):
    def __init__(self,*args,**kwargs):
        self.batch=None
        super().__init__(*args,**kwargs)
        self.batch=BatchedJumpDetector(1,device,self.minimum_air_time,self.stable_hold,self.minimum_rise)

    def reset(self):
        super().reset()
        if getattr(self,"batch",None) is not None:
            self.batch.reset()

    def update(self,sample):
        changes=super().update(sample)
        compare(self,self.batch,sample,changes)
        return changes

test_event_detector.JumpEventDetector=ParityDetector
suite=unittest.defaultTestLoader.loadTestsFromTestCase(test_event_detector.EventTests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
# Replay raw MuJoCo forces and root state; not a synthetic policy success.
from g1_dynamicparkour.events import Sample
trace=Path("experiments/20261008T170607080257Z-physical-target-detector-89d810b1/physical_event_samples.csv")
scalar=JumpEventDetector()
batch=BatchedJumpDetector(1,device)
physical_samples=0
with trace.open() as file:
    for row in csv.DictReader(file):
        sample=Sample(float(row["time_s"]),
            (float(row["start_left_n"])>5,float(row["start_right_n"])>5),
            (float(row["target_left_n"])>5,float(row["target_right_n"])>5),
            float(row["root_z"]),float(row["root_vz"]),False,fallen=float(row["root_z"])<.5)
        changes=scalar.update(sample)
        compare(scalar,batch,sample,changes)
        assert scalar.phase.value==row["phase"]
        physical_samples+=1
# Reset only one world; preserve the other's clock and terminal state.
multi=BatchedJumpDetector(2,device)
multi.update([.2,.2],[[False,False],[False,False]],[[False,False],[False,False]],[.76,.76],[0,0],[False,False])
multi.reset(torch.tensor([0],device=device))
assert multi.last_time[0]==-float("inf") and multi.last_time[1]==.2
multi.update([0,.3],[[False,False],[False,False]],[[False,False],[False,False]],[.76,.76],[0,0],[False,False])
try:
    multi.update([0,.4],[[False,False],[False,False]],[[False,False],[False,False]],[.76,.76],[0,0],[False,False])
except ValueError:
    pass
else:
    raise AssertionError("non-increasing episode timestamp accepted")
report={"pass":True,"device":device,"synthetic_tests":result.testsRun,"matched_updates":counter,
    "actual_physical_samples":physical_samples,"actual_trace":str(trace),"partial_reset_isolated":True,
    "physical_events":scalar.events,"scope":"GPU/scalar detector equivalence; actual trace is external-force calibration, not RL success"}
Path(os.environ["G1_RUN_DIR"],"batched_detector_validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))