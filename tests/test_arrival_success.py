import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
from humanoid_imitation.planning.arrival_success import evaluate_arrival_hold,ArrivalHoldMonitor


def logs(n=1201):
    return dict(time_s=np.arange(n)/30,position_error_m=np.full(n,.5),yaw_error_rad=np.zeros(n),
                speed_mps=np.zeros(n),abs_wz_radps=np.zeros(n),root_pos=np.zeros((n,3)))


def test_late_arrival_can_pass_and_early_motion_resets_settling():
    r=logs();r['speed_mps'][:901]=1
    result=evaluate_arrival_hold(r,'TIME')
    assert result['arrival_hold_success']
    assert result['first_arrival_hold_time_s']>35


def test_spike_after_complete_hold_does_not_erase_observed_success_but_fail_does():
    r=logs();r['abs_wz_radps'][-1]=.205
    assert evaluate_arrival_hold(r,'TIME')['arrival_hold_success']
    assert not evaluate_arrival_hold(r,'FAIL')['arrival_hold_success']


def test_sparse_short_far_and_drifting_intervals_do_not_pass():
    assert not evaluate_arrival_hold(logs(150),'TIME')['arrival_hold_success']
    r=logs();r['position_error_m'][:]=1.047
    assert not evaluate_arrival_hold(r,'TIME')['arrival_hold_success']
    r=logs();r['root_pos'][:,0]=r['time_s']*.1
    assert not evaluate_arrival_hold(r,'TIME')['arrival_hold_success']
    r={k:v[::3] for k,v in logs().items()}
    assert not evaluate_arrival_hold(r,'TIME')['arrival_hold_success']


def test_online_and_archive_use_identical_rule():
    r=logs(301);m=ArrivalHoldMonitor()
    for i,t in enumerate(r['time_s']):m.update(t,r['position_error_m'][i],0,0,0,r['root_pos'][i,:2])
    assert m.first_success_time_s==evaluate_arrival_hold(r,'TIME')['first_arrival_hold_time_s']
