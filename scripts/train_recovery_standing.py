"""Dedicated recovery-only PPO from genuine landing states; original source jump remains separate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from mjlab.managers.observation_manager import ObservationTermCfg
import train_jump
from train_landing_recovery import LandingRecoveryEnv
from g1_dynamicparkour import jump
from g1_dynamicparkour.recovery_standing_policy import RecoveryStandingModelCfg

_original_build = train_jump.build_jump_configs

def build_recovery_standing_configs(settings):
    if settings.get('action_basis')!='direct' or settings.get('jump_prior_checkpoint') or settings.get('landing_reset_fraction')!=1:
        raise ValueError('Dedicated recovery only: direct29, no frozen jumping prior, actual landing resets100%')
    base=dict(settings,actor_contact_observations=False)
    base.pop('standing_prior_checkpoint',None)
    base.pop('standing_prior_sha256',None)
    cfg,agent,geometry=_original_build(base)
    cfg.observations['actor'].terms['current_unlabelled_foot_contact']=ObservationTermCfg(func=jump.actor_unlabelled_foot_contacts)
    agent.actor=RecoveryStandingModelCfg(**asdict(agent.actor),prior_checkpoint=settings['standing_prior_checkpoint'],prior_sha256=settings['standing_prior_sha256'])
    agent.actor.class_name="g1_dynamicparkour.recovery_standing_policy:RecoveryStandingMLP"
    agent.algorithm.learning_rate=settings.get("ppo_learning_rate",agent.algorithm.learning_rate)
    agent.algorithm.schedule=settings.get("ppo_schedule",agent.algorithm.schedule)
    if not 0<agent.algorithm.learning_rate<=.001 or agent.algorithm.schedule not in ('fixed','adaptive'):
        raise ValueError('Declared recovery fine-tuning learning rate/schedule outside current gate')
    agent.algorithm.gamma=settings.get("ppo_gamma",agent.algorithm.gamma)
    agent.algorithm.lam=settings.get("ppo_lambda",agent.algorithm.lam)
    return cfg,agent,geometry

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,required=True)
    args,_=parser.parse_known_args()
    LandingRecoveryEnv.reset_settings=json.loads(args.config.read_text())
    train_jump.build_jump_configs=build_recovery_standing_configs
    train_jump.JumpEnv=LandingRecoveryEnv
    train_jump.main()
