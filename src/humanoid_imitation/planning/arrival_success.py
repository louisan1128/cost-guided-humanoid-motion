"""Measured five-second arrival hold, independent of arrival deadline."""
from collections import deque
import math
import numpy as np


class ArrivalHoldMonitor:
    def __init__(self, position_m=1., yaw_deg=25., speed_mps=.15,
                 yaw_rate_radps=.2, hold_s=5., drift_m=.1, max_gap_s=.05):
        values=(position_m,yaw_deg,speed_mps,yaw_rate_radps,hold_s,drift_m,max_gap_s)
        if not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError('Arrival thresholds must be finite and positive')
        self.position_m,self.yaw_rad=position_m,math.radians(yaw_deg)
        self.speed_mps,self.yaw_rate_radps=speed_mps,yaw_rate_radps
        self.hold_s,self.drift_m,self.max_gap_s=hold_s,drift_m,max_gap_s
        self.samples=deque();self.previous_time=None
        self.first_success_time_s=None

    def update(self, time_s, position_error_m, yaw_error_rad, speed_mps, yaw_rate_radps, xy):
        xy=np.asarray(xy,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite([time_s,position_error_m,yaw_error_rad,speed_mps,yaw_rate_radps]).all():
            raise ValueError('Invalid measured arrival sample')
        if self.previous_time is not None:
            gap=time_s-self.previous_time
            if gap<=0:raise ValueError('Arrival timestamps must increase')
            if gap>self.max_gap_s+1e-8:self.samples.clear()
        self.previous_time=time_s
        good=(position_error_m<=self.position_m and abs(yaw_error_rad)<=self.yaw_rad
              and speed_mps<=self.speed_mps and abs(yaw_rate_radps)<=self.yaw_rate_radps)
        if not good:
            self.samples.clear()
            return False
        self.samples.append((float(time_s),xy.copy()))
        # Keep a sample at/before the five-second boundary: discretization must
        # not convert 4.97 seconds into a complete five-second observation.
        while len(self.samples)>1 and self.samples[1][0]<=time_s-self.hold_s+1e-8:
            self.samples.popleft()
        success=(time_s-self.samples[0][0]>=self.hold_s-1e-8
                 and np.linalg.norm(xy-self.samples[0][1])<=self.drift_m)
        if success and self.first_success_time_s is None:self.first_success_time_s=float(time_s)
        return bool(success)


def evaluate_arrival_hold(rollout, ending, **thresholds):
    monitor=ArrivalHoldMonitor(**thresholds)
    for i,t in enumerate(rollout['time_s']):
        monitor.update(float(t),float(rollout['position_error_m'][i]),float(rollout['yaw_error_rad'][i]),
                       float(rollout['speed_mps'][i]),float(rollout['abs_wz_radps'][i]),rollout['root_pos'][i,:2])
    return dict(arrival_hold_success=bool(monitor.first_success_time_s is not None and ending!='FAIL'),
                first_arrival_hold_time_s=monitor.first_success_time_s,
                criterion='any measured continuous five-second nearby stationary interval; no arrival deadline; later FAIL invalidates success')
