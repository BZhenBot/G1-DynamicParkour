"""Opt-in real PPO curriculum on genuine landing resets mixed with original source starts."""
import argparse
import json
import os
from pathlib import Path
import torch
import train_jump
from g1_dynamicparkour.landing_resets import LandingResetBank
from g1_dynamicparkour import jump

class LandingRecoveryEnv(train_jump.JumpEnv):
    reset_settings=None
    def __init__(self,cfg,directory,geometry,**kwargs):
        settings=self.reset_settings
        self.reset_fraction=float(settings["landing_reset_fraction"])
        if not 0<self.reset_fraction<=1:
            raise ValueError("Explicit recovery-reset fraction must be in (0,1]")
        self.bank=LandingResetBank(settings["landing_reset_dataset"])
        self.reset_count=0
        super().__init__(cfg,directory,geometry,**kwargs)
        self.recovery_origin=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.recovery_record=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
        self.recovery_initial_step=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        (directory/"landing_curriculum_manifest.json").write_text(json.dumps({
            "scope":"Real landing resets; only source-origin episodes may count as full jump training events; frozen evaluation ALWAYS uses unchanged source reset",
            "dataset":str(self.bank.path),"snapshot_sha256":self.bank.hashes,"snapshot_cases":len(self.bank.records),
            "recovery_reset_fraction":self.reset_fraction,"original_clock_retained":True,
            "episode_duration_note":"Manager episode control_steps includes recorded pre-reset clock; recovery_elapsed_control_steps removes it",
            "actor_mode_flag":False,"inherited_privileged_critic_phase":4,
            "robot_state_writes":"reset initialization only","inherited_event_rewards":False},indent=2))

    def sample_record_indices(self,count):
        return torch.randint(len(self.bank.records),(count,),device=self.device)

    def _reset_idx(self,env_ids=None):
        super()._reset_idx(env_ids)
        if not hasattr(self,"recovery_origin"):
            return
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        self.recovery_origin[ids]=False
        self.recovery_record[ids]=-1
        self.recovery_initial_step[ids]=0
        selected=ids[torch.rand(len(ids),device=self.device)<self.reset_fraction]
        if not len(selected):
            return
        indices=self.sample_record_indices(len(selected))
        initial=self.bank.restore(self,selected,indices)
        self.recovery_origin[selected]=True
        self.recovery_record[selected]=indices
        self.recovery_initial_step[selected]=initial
        self.reset_count+=len(selected)

    def close(self):
        if hasattr(self,"recovery_origin"):
            current=[{"env_id":i,"partial_episode":True,"recovery_origin":bool(self.recovery_origin[i]),
                "snapshot_case":int(self.recovery_record[i]),"current_phase":int(jump.state(self).detector.phase[i]),
                "elapsed_control_steps":int(self.episode_length_buf[i]-self.recovery_initial_step[i])}
                for i in range(self.num_envs)]
            (self.directory/"landing_curriculum_active_snapshot.json").write_text(json.dumps({
                "scope":"Partial episodes, not completed successes; inherited phase not source jump evidence",
                "total_recovery_resets":self.reset_count,"episodes":current},indent=2))
        super().close()

    def episode_details(self,env_id):
        details=super().episode_details(env_id)
        recovery=bool(self.recovery_origin[env_id]) if hasattr(self,"recovery_origin") else False
        details["recovery_origin"]=recovery
        if recovery:
            details["task"]="real_landing_reset_recovery_only"
            details["recovery_success"]=details["success"]
            details["success"]=False
            details["jump_events"]={key:False for key in details["jump_events"]}
            details["recovery_record_index"]=int(self.recovery_record[env_id])
            details["recovery_elapsed_control_steps"]=int(self.episode_length_buf[env_id]-self.recovery_initial_step[env_id])
            details["inherited_privileged_phase"]="actual_recorded_target_contact"
        return details

if __name__=="__main__":
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config",type=Path,required=True)
    args,_=parser.parse_known_args()
    LandingRecoveryEnv.reset_settings=json.loads(args.config.read_text())
    train_jump.JumpEnv=LandingRecoveryEnv
    train_jump.main()
