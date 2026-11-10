"""CPU-only goal relabelling of actual tracker states, never tracker observations."""
"""CPU-only goal relabelling of actual tracker states, never tracker observations."""
import numpy as np


def rotation_matrix(xyzw):
    q = np.asarray(xyzw, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1)
    if np.any(abs(norm - 1) > .01):
        raise ValueError('Invalid quaternion norm')
    x, y, z, w = np.moveaxis(q / norm[..., None], -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))


def heading(rotation):
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


def anchor_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def stop_at(data, goal, hold_steps, speed_limit, yaw_limit):
    end = goal + hold_steps
    if end >= len(data['q']):
        return False
    vel = data['root_vel'][goal:end+1]
    omega = data['root_ang_vel'][goal:end+1]
    return bool(np.all(np.linalg.norm(vel[:,:2],axis=1) <= speed_limit)
                and np.all(abs(omega[:,2]) <= yaw_limit))


def windows(data, history, horizon, stride, offsets, hold_steps, speed_limit, yaw_limit,
            stop_through_target=False, recovery_stop_offsets=(), include_final_stop=False):
    if any(not isinstance(k,(int,np.integer)) or k<=0 for k in recovery_stop_offsets):
        raise ValueError('Recovery offsets must be positive integer control steps')
    required = ['q','dq','root_pos','root_rot','root_vel','root_ang_vel','time_s']
    n = len(data['q'])
    for key in required:
        if len(data[key]) != n or not np.all(np.isfinite(data[key])):
            raise ValueError('Malformed actual state: '+key)
    if data['q'].shape[1] != 28:
        raise ValueError('Expected generic humanoid 28 DoFs')
    rot = rotation_matrix(data['root_rot'])
    yaws = heading(rot)
    requested_goal = data.get('requested_goal_world')
    if requested_goal is not None:
        requested_goal = np.asarray(requested_goal, dtype=float)
        if requested_goal.shape != (3,) or not np.isfinite(requested_goal).all():
            raise ValueError('Malformed fixed recovery goal')
        if not include_final_stop or 'requested_deadline_s' not in data or offsets or recovery_stop_offsets:
            raise ValueError('Fixed recovery goal requires persistent stop and original deadline only')
    final_stop = None
    if include_final_stop:
        # A measured stable suffix supplies one persistent endpoint across the
        # entire approach/turn, rather than only a short t+offset subgoal.
        stable = ((np.linalg.norm(data['root_vel'][:, :2], axis=1) <= speed_limit)
                  & (abs(data['root_ang_vel'][:, 2]) <= yaw_limit))
        unstable = np.flatnonzero(~stable)
        candidate = int(unstable[-1] + 1) if len(unstable) else 0
        if stop_at(data, candidate, hold_steps, speed_limit, yaw_limit):
            final_stop = candidate
    for t in range(history-1, n-horizon, stride):
        anchor = anchor_rotation(yaws[t])
        def pose(indices):
            pos = data['root_pos'][indices].copy()
            pos[:,:2] -= data['root_pos'][t,:2]
            pos = pos @ anchor
            local_rot = np.einsum('ij,njk->nik',anchor.T, rot[indices])
            # First column followed by second column; not row-major [:,:2].flatten().
            six = local_rot[:,:,:2].transpose(0,2,1).reshape(-1,6)
            return np.concatenate([data['q'][indices],pos,six],axis=1)
        past = np.arange(t-history+1,t+1)
        future = np.arange(t+1,t+horizon+1)
        # Reject large exp-map branch jumps; no scalar wrapping of ball joints.
        if np.any(np.linalg.norm(np.diff(data['q'][t-history+1:t+horizon+1],axis=0),axis=1)>1.):
            continue
        hist = np.concatenate([pose(past),data['dq'][past],data['root_vel'][past]@anchor,
                               data['root_ang_vel'][past]@anchor],axis=1)
        target = pose(future)
        requests=[(t+offset,False) for offset in offsets]
        requests.extend((t+offset,True) for offset in recovery_stop_offsets)
        if final_stop is not None and final_stop not in [r[0] for r in requests]:
            requests.append((final_stop,False))
        for g, recovery in requests:
            offset = g-t
            if g >= n:
                continue
            stop = stop_at(data,g,hold_steps,speed_limit,yaw_limit)
            if recovery and not stop:
                continue
            # An arrival stop remains a stop task after its deadline. Do not
            # supervise it with an uncommanded restart inside the target chunk.
            if stop and (stop_through_target or recovery) and not stop_at(
                    data,g,max(hold_steps,t+horizon-g),speed_limit,yaw_limit):
                continue
            if offset == 0 and not stop:
                continue
            goal_xy = data['root_pos'][g,:2] if requested_goal is None else requested_goal[:2]
            goal_yaw = yaws[g] if requested_goal is None else requested_goal[2]
            delta = np.r_[goal_xy-data['root_pos'][t,:2], 0.] @ anchor
            yaw = goal_yaw-yaws[t]
            arrival = data['root_vel'][g] @ anchor
            # Stop is a zero-velocity task with a tolerated measured residual.
            arrival_xy = np.zeros(2) if stop else arrival[:2]
            arrival_wz = 0. if stop else data['root_ang_vel'][g,2]
            branch_deadline = data.get('requested_deadline_s')
            if branch_deadline is not None and not include_final_stop:
                raise ValueError('Recovery branch deadline requires a persistent final endpoint')
            deadline = float(branch_deadline) if branch_deadline is not None else data['time_s'][t if recovery else g]
            condition = np.r_[delta[:2],np.sin(yaw),np.cos(yaw),
                              max(0.,deadline-data['time_s'][t]),arrival_xy,arrival_wz,float(stop)]
            row=dict(history=hist.astype('float32'),condition=condition.astype('float32'),
                       target=target.astype('float32'),anchor_world=np.r_[data['root_pos'][t],yaws[t]],
                       goal_world=np.r_[goal_xy,goal_yaw],
                       goal_timestamp=deadline,timestep=t,goal_timestep=g)
            if recovery_stop_offsets or branch_deadline is not None:
                # Branch teacher does not promise the learner's original deadline.
                # Exclude these recovery targets from exact-time endpoint losses.
                row.update(label_kind=int(recovery or branch_deadline is not None),arrival_timestamp=data['time_s'][g])
            yield row


def normalization(train):
    result = {}
    for key in ('history','condition','target'):
        values = train[key].reshape(-1,train[key].shape[-1]).astype('float64')
        result[key+'_mean'] = values.mean(axis=0).astype('float32')
        result[key+'_std'] = np.maximum(values.std(axis=0),1e-6).astype('float32')
    return result
