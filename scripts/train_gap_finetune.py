"""Original-source jump fine-tuning with explicit PPO settings; unchanged physics/MDP."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import train_jump

_original_build = train_jump.build_jump_configs

def build_gap_finetune_configs(settings):
    if settings.get('action_basis') != 'recovery_coordinate' or not settings.get('bounded_coordinate_mean') or not settings.get('actor_contact_observations'):
        raise ValueError('Gap fine-tuning preserves the validated114/30 bounded-coordinate jump architecture')
    if settings.get('landing_reset_fraction', 0) or settings.get('recovery_only_training', False):
        raise ValueError('All gap-training episodes must start on the original source platform')
    if not .02 <= settings['gap'] <= .15:
        raise ValueError('Current predeclared gap development range is2–15cm')
    cfg, agent, geometry = _original_build(settings)
    agent.algorithm.learning_rate = settings['ppo_learning_rate']
    agent.algorithm.schedule = settings['ppo_schedule']
    if not 0 < agent.algorithm.learning_rate <= .001 or agent.algorithm.schedule not in ('fixed', 'adaptive'):
        raise ValueError('Explicit fine-tuning learning-rate/schedule outside validated bounds')
    for key, field in (('ppo_gamma', 'gamma'), ('ppo_lambda', 'lam')):
        value = settings.get(key, getattr(agent.algorithm, field))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Invalid PPO physical-horizon discount/GAE')
        setattr(agent.algorithm, field, value)
    return cfg, agent, geometry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, required=True)
    args, _ = parser.parse_known_args()
    settings = json.loads(args.config.read_text())
    cfg, agent, geometry = build_gap_finetune_configs(settings)
    manifest = {'scope': 'Independent source-start jump fine-tuning; frozen standing/jump priors, no landing bank or state rewrites after reset',
                'source_config': str(args.config), 'geometry': asdict(geometry), 'algorithm': asdict(agent.algorithm),
                'physical_dt_s': cfg.sim.mujoco.timestep, 'control_dt_s': cfg.sim.mujoco.timestep * cfg.decimation,
                'acceptance': 'Original takeoff/AIR/target/stable2s thresholds unchanged; training reward is not task acceptance'}
    Path(os.environ['G1_RUN_DIR'], 'gap_finetune_plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    train_jump.build_jump_configs = build_gap_finetune_configs
    train_jump.main()
