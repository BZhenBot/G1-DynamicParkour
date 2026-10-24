"""CPU合成时间预算测试；不重算旧失败为成功。"""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_dynamicparkour.course_time_budget import time_failure
old={'maximum_seconds':24,'per_leg_maximum_seconds':12}
new=dict(old,time_budget_mode='per_skill_with_preparation',preparation_maximum_seconds=10)
assert time_failure(15.94,old,3.92,12.8)=='leg2_timeout'
assert time_failure(16.2,new,3.92,12.8) is None
assert time_failure(12.02,new,None,None)=='leg1_timeout'
assert time_failure(14.02,new,4,None)=='middle_preparation_timeout'
assert time_failure(17,new,4,14.02)=='invalid_second_skill_start_budget'
assert time_failure(20.02,new,4,8)=='leg2_timeout'
assert time_failure(24.02,new,4,13)=='route_timeout'
assert time_failure(10,new,4,3)=='invalid_second_skill_start_budget'
r={'pass':True,'legacy_kept':True,'whole24s_kept':True,'preparation10s_bounded':True,'each_jump12s_bounded':True,'physical_success_unchanged':True,'scope':'合成时间预算逻辑，非物理episode成绩'}
(Path(os.environ['G1_RUN_DIR'])/'course_time_budget_validation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
