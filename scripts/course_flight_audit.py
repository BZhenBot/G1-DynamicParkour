"""离线独立核验同一次真实腾空的完整200Hz连续区间，不供Actor使用。"""

def verify_flight(physics,event,events,episode_end_s):
    for i,x in enumerate(physics):
        assert abs(x['interval_end_s']-x['interval_start_s']-.005)<1e-12
        if i:assert abs(x['interval_start_s']-physics[i-1]['interval_end_s'])<1e-12
    t=event['time_s'];leg=event['leg_id'];source=event['source']
    takeoff=max(x['time_s'] for x in events if x['leg_id']==leg and x['phase']=='takeoff' and x['time_s']<=t)
    landing=next((x['time_s'] for x in events if x['leg_id']==leg and x['phase']=='target_contact' and x['time_s']>=t),episode_end_s)
    dual=[x for x in physics if x['interval_end_s']<=takeoff+1e-12 and min(x['surfaces'][source]['native_netforce_n'])>5]
    reference=dual[-1]['root_height_m'] if dual else None
    selected=[x for x in physics if takeoff-.02-1e-12<x['interval_end_s']<=landing+.02+1e-12]
    segments=[];current=[]
    for x in selected:
        supported=any(max(v['native_netforce_n'])>5 for v in x['surfaces'].values())
        if not supported:current.append(x)
        elif current:segments.append(current);current=[]
    if current:segments.append(current)
    containing=[q for q in segments if q[0]['interval_start_s']-1e-12<=t<=q[-1]['interval_end_s']+1e-12]
    q=max(containing,key=len,default=[])
    count=len(q);gain=max(x['root_height_m'] for x in q)-reference if q and reference is not None else None
    streak=0;prefix_max=0
    for x in [v for v in physics if v['interval_end_s']<=t+1e-12][-32:]:
        supported=any(max(v['native_netforce_n'])>5 for v in x['surfaces'].values())
        streak=0 if supported else streak+1;prefix_max=max(prefix_max,streak)
    return dict(event,protocol='complete_contiguous_physics_flight_v2',maximum_consecutive_unsupported_physics_samples=count,
        minimum_required_samples=16,legacy_event_prefix_maximum_samples=prefix_max,
        unsupported_interval_start_s=q[0]['interval_start_s'] if q else None,
        unsupported_interval_end_s=q[-1]['interval_end_s'] if q else None,
        last_dual_source_root_height_m=reference,physical_root_height_gain_m=gain,
        verified_200hz=count>=16 and gain is not None and gain+1e-12>=.03)
