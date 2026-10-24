"""路线时间预算独立于物理事件；保留旧协议，技能协议仍整条24秒。"""
def time_failure(t,settings,first_stable,second_start=None):
    if t>settings['maximum_seconds']+1e-12:return 'route_timeout'
    limit=settings['per_leg_maximum_seconds']
    if settings.get('time_budget_mode','legacy_from_middle_stable')=='legacy_from_middle_stable':
        start=0. if first_stable is None else first_stable
        return ('leg1_timeout' if first_stable is None else 'leg2_timeout') if t-start>limit+1e-12 else None
    if settings['time_budget_mode']!='per_skill_with_preparation':raise ValueError('未知时间预算协议')
    if first_stable is None:return 'leg1_timeout' if t>limit+1e-12 else None
    prep=settings['preparation_maximum_seconds']
    if second_start is None:return 'middle_preparation_timeout' if t-first_stable>prep+1e-12 else None
    if second_start<first_stable or second_start-first_stable>prep+1e-12:return 'invalid_second_skill_start_budget'
    return 'leg2_timeout' if t-second_start>limit+1e-12 else None
