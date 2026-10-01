#!/usr/bin/env python3
"""Decay pose/velocity offsets into a moving next clip; never pause clip time.

Experimental kinematic inertialization, not a dynamics feasibility guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from build_start_walk_probe import BALL_ROTATIONS, load_motion


def build(source, output, duration):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    info, frames = load_motion(source)
    report = json.loads(source.with_suffix('.json').read_text())
    b = report.get('boundary_frame')
    if b is None:
        b = report['boundaries'][0]['reference_boundary_frame']
    fps = float(info['fps']); dt = 1 / fps
    n = round(duration * fps)
    if n < 3 or b < 2 or b + n + 1 >= len(frames):
        raise ValueError('Invalid correction duration')
    duration = n / fps
    # Extrapolate incoming t=0 from t=dt,2dt because joins omit its first frame.
    incoming_zero = 2 * frames[b+1] - frames[b+2]
    delta = frames[b] - incoming_zero
    derivative = (frames[b]-frames[b-1])/dt - (frames[b+2]-frames[b+1])/dt
    result = frames.copy()
    u = np.arange(1,n+1) / n
    # quintic bases: start offset and derivative, zero second derivative;
    # end value, first and second derivatives all zero.
    h0 = 1-10*u**3+15*u**4-6*u**5
    h1 = u-6*u**3+8*u**4-3*u**5
    result[b+1:b+n+1] += h0[:,None]*delta + duration*h1[:,None]*derivative
    for block in (slice(3,6),*BALL_ROTATIONS):
        old0 = Rotation.from_rotvec(frames[b,block])
        oldm = Rotation.from_rotvec(frames[b-1,block])
        new1 = Rotation.from_rotvec(frames[b+1,block])
        new2 = Rotation.from_rotvec(frames[b+2,block])
        step = new2 * new1.inv()
        new0 = step.inv() * new1
        offset0 = (old0 * new0.inv()).as_rotvec()
        # Relative-offset derivative in the correction rotation's local chart.
        newm = step.inv() * new0
        offsetm = (oldm * newm.inv()).as_rotvec()
        offsetdot = (offset0-offsetm)/dt
        correction = h0[:,None]*offset0 + duration*h1[:,None]*offsetdot
        result[b+1:b+n+1,block] = (Rotation.from_rotvec(correction)*Rotation.from_rotvec(frames[b+1:b+n+1,block])).as_rotvec()
    if not np.isfinite(result).all():raise ValueError('Nonfinite output')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as f:pickle.dump(dict(loop_mode=0,fps=fps,frames=result.astype(np.float32).tolist()),f)
    meta=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              method='moving incoming clip with quintic pose/velocity offset decay; no inserted pause; no contact or dynamics guarantee',
              correction_duration_s=duration,boundary_frame=b,boundary_time_s=b/fps,
              fps=fps,duration_s=(len(result)-1)/fps,status='kinematic candidate; requires physical evaluation')
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=.3)
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.duration),indent=2))
