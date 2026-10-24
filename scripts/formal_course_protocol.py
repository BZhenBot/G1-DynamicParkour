"""正式100路线固定协议与已通过开发候选的一致性核验。"""
import hashlib,json
from pathlib import Path

def validate_formal(settings,episodes,seed,max_steps):
    freeze=settings.get('formal_protocol')
    if freeze is None:return None
    p=Path(freeze['path']);digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==freeze['sha256']
    protocol=json.loads(p.read_text());assert protocol['protocol_id']==freeze['id']
    assert episodes==protocol['episodes']==100 and seed==protocol['seed_start'] and max_steps==0
    if protocol['protocol_id']=='v2-fixed-three-platform-100-independent-v1':assert seed==70000
    else:
        assert protocol['protocol_id'].startswith(('v21-','v22-')) and seed>=81000
        assert not set(range(seed,seed+100)).intersection(protocol.get('development_seeds',[]))
    assert protocol['minimum_jumps']==2 and protocol['airborne_minimum_consecutive_physics_samples']==16
    assert protocol['physics_dt_seconds']==.005 and protocol['airborne_minimum_height_gain_m']==.03
    assert protocol['landing_stable_seconds']==2 and protocol['mid_episode_physical_reset_allowed'] is False
    assert protocol['seed_end']==seed+episodes-1 and protocol['phase_bins']==8
    assert settings['course']==protocol['scene']
    assert settings['maximum_seconds']==protocol['whole_route_limit_seconds']==24
    assert settings['preparation_maximum_seconds']==protocol['preparation_limit_seconds']==10
    assert settings['per_leg_maximum_seconds']==protocol['per_jump_recovery_limit_seconds']==12
    fa=settings['formal_acceptance'];assert fa['status']=='frozen_for_100_independent_routes'
    assert fa['episodes_minimum']==100 and fa['success_rate_minimum']==protocol['success_rate_minimum']==.7
    path=Path(freeze['development_audit']);assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze['development_audit_sha256']
    audit=json.loads(path.read_text());assert audit['audit_pass'] and audit['episodes_count']==8 and audit['qualified_successes']>=6
    assert audit['independent_unique_episode_seeds']==8 and audit['input_config_matches_source_snapshot']
    if protocol.get('actual_collision_geometry_required'):assert audit['measured_collision_geometry']['verified_compiled_box_sizes']
    run=Path(audit['run']);dev=json.loads((run/'course_evaluation_config.json').read_text())['settings']
    exclude={'purpose','formal_acceptance','formal_protocol'}
    assert {k:v for k,v in settings.items() if k not in exclude}=={k:v for k,v in dev.items() if k not in exclude},'正式候选与开发候选不一致'
    for role in ('jump','recovery','alignment','position','second_jump','end_recovery'):
        assert hashlib.sha256(Path(settings[role+'_checkpoint']).read_bytes()).hexdigest()==settings[role+'_sha256']
    if settings.get('settle_recovery_checkpoint'):
        assert hashlib.sha256(Path(settings['settle_recovery_checkpoint']).read_bytes()).hexdigest()==settings['settle_recovery_sha256']
    return {'protocol_id':freeze['id'],'protocol_sha256':digest,'episodes':episodes,'seed_start':seed,'seed_end':seed+episodes-1,
        'development_audit_sha256':freeze['development_audit_sha256'],'candidate_matches_development':True,'all_six_role_checkpoint_hashes_checked':True}
