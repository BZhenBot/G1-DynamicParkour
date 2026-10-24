"""PPO source jump subtask to qualified public handoff; independent full evaluation remains mandatory."""
import argparse,copy,json,math,os
from dataclasses import asdict
from pathlib import Path
import train_jump
from train_gap_finetune import build_gap_finetune_configs
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from g1_dynamicparkour import handoff_subtask

class HandoffSubtaskEnv(train_jump.JumpEnv):
    def episode_details(self,env_id):
        details=super().episode_details(env_id)
        original_full=details['success']
        details.update(task='jump_to_qualified_public_handoff_subtask_not_complete_jump',success=False,
            complete_jump_success=original_full,subtask_success=bool(handoff_subtask.state(self).qualified[env_id]),
            subtask_requires='Actual AIR/target phase4 plus unchanged public ready; original full stable2s remains separate')
        return details

def build_handoff_subtask_configs(settings):
    if settings.get('training_task_scope')!='jump_to_qualified_public_handoff_subtask_not_complete_jump':raise ValueError('Explicit subtask scope required')
    if 'landing_foot_x_margin_m' in settings:raise ValueError('This candidate changes subtask credit/termination, not landing margin')
    cfg,agent,geometry=build_gap_finetune_configs(settings)
    actor=cfg.observations['actor']
    teacher=copy.deepcopy(actor)
    teacher.enable_corruption=False
    cfg.observations['handoff_teacher_current']=teacher
    assert 'handoff_teacher_current' not in {name for group in agent.obs_groups.values() for name in group}, 'Teacher current state must not enter policy/value networks'
    assert all(t.delay_max_lag==0 and t.history_length==0 for t in actor.terms.values()), 'Readiness must not mutate delay/history buffers'
    credit=settings['handoff_subtask_event_bonus']
    if not math.isfinite(credit) or not 0<credit<=200:raise ValueError('Subtask pulse outside declared bounds')
    cfg.events['reset_handoff_subtask']=EventTermCfg(func=handoff_subtask.reset,mode='reset')
    cfg.terminations['update_handoff_subtask']=TerminationTermCfg(func=handoff_subtask.update)
    cfg.terminations['qualified_public_handoff_subtask']=TerminationTermCfg(func=handoff_subtask.completed,time_out=False)
    cfg.rewards['qualified_public_handoff_subtask_event']=RewardTermCfg(func=handoff_subtask.bonus,weight=credit/(cfg.decimation*cfg.sim.mujoco.timestep))
    return cfg,agent,geometry

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,required=True);a,_=p.parse_known_args()
    settings=json.loads(a.config.read_text());cfg,agent,geometry=build_handoff_subtask_configs(settings)
    out=Path(os.environ['G1_RUN_DIR'])
    (out/'handoff_subtask_plan.json').write_text(json.dumps({'scope':settings['training_task_scope'],'original_full_acceptance':'Unchanged original12s source/AIR/target/stable2s+200Hz collision and plant fidelity, tested separately; training subtask success is never complete jump success',
      'geometry':asdict(geometry),'algorithm':asdict(agent.algorithm),'event_credit':settings['handoff_subtask_event_bonus'],
      'control_dt_s':cfg.decimation*cfg.sim.mujoco.timestep,'readiness_sampling':'Manager termination/reward cached state, one physics substep derived-pose staleness as original mjlab managers; evaluation independently uses current forward state',
      'actor_dimensions_unchanged':114,'privileged_task_gate':'Actual phase4 and AIR history used only for training reward/termination, never added to actor','no_robot_state_write_after_reset':True},indent=2)+'\n')
    train_jump.build_jump_configs=build_handoff_subtask_configs;train_jump.JumpEnv=HandoffSubtaskEnv
    train_jump.main()
    episodes=[json.loads(line) for line in (out/'episodes.jsonl').open()]
    for row in episodes:
        assert not row['success'] and not row['complete_jump_success']
        if row['subtask_success']:assert row['jump_events']['airborne'] and row['jump_events']['target_contact_after_airborne'] and not row['jump_events']['stable_success']
    (out/'handoff_subtask_training_summary.json').write_text(json.dumps({'completed_subtask_episodes':len(episodes),'qualified_handoff_subtask_successes':sum(v['subtask_success'] for v in episodes),'complete_jump_successes':0,'scope':'Training diagnostics only; independent original full physical source evaluation required'},indent=2)+'\n')