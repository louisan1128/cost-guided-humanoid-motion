"""NumPy contract for goal conditioning and decoding 37D motion into world poses."""
import numpy as np
from humanoid_imitation.data.goal_dataset import rotation_matrix, heading, anchor_rotation
from humanoid_imitation.data.goal_dataset import stop_at


def moving_stop_scenario(validation, captured, dt, lead_s=4.):
    """Relabel a validation endpoint after several seconds of actual movement."""
    lead=int(round(lead_s/dt))
    for i,t in enumerate(validation['timestep']):
        t=int(t);g=t+lead
        if g>=len(captured['q']): continue
        distance=np.linalg.norm(captured['root_pos'][g,:2]-captured['root_pos'][t,:2])
        speed=np.linalg.norm(captured['root_vel'][t,:2])
        if distance>.5 and speed>.3 and stop_at(captured,g,int(np.ceil(.5/dt)),.15,.2):
            return i,g
    raise ValueError('Validation has no moving-to-stop scenario with a 4-second lead; collect a suitable rollout')


def rotation6_to_matrix(six):
    six = np.asarray(six, dtype=np.float64)
    if six.shape[-1] != 6 or not np.isfinite(six).all():
        raise ValueError('Expected finite column-ordered rotation6')
    a, b = six[..., :3], six[..., 3:]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(na < 1e-7):
        raise ValueError('Degenerate first rotation column')
    x = a / na
    y = b - (x*b).sum(-1, keepdims=True)*x
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(ny < 1e-7):
        raise ValueError('Degenerate second rotation column')
    y = y / ny
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def matrix_to_quaternion(matrix):
    """SO(3) matrices to normalized xyzw, including rotations close to pi."""
    matrix = np.asarray(matrix, dtype=np.float64)
    result = []
    for r in matrix.reshape(-1, 3, 3):
        trace = np.trace(r)
        if trace > 0:
            s = 2*np.sqrt(trace+1)
            q = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s,
                 (r[1,0]-r[0,1])/s, s/4]
        else:
            i = int(np.argmax(np.diag(r)))
            j, k = (i+1)%3, (i+2)%3
            s = 2*np.sqrt(max(1+r[i,i]-r[j,j]-r[k,k], 0))
            q = np.zeros(4)
            q[i] = s/4
            q[j] = (r[j,i]+r[i,j])/s
            q[k] = (r[k,i]+r[i,k])/s
            q[3] = (r[k,j]-r[j,k])/s
        q = np.asarray(q)
        result.append(q/np.linalg.norm(q))
    return np.asarray(result).reshape(matrix.shape[:-2]+(4,))


def encode_history(states):
    """Anchor all past frames to the measured current yaw and xy; keep world z."""
    rot = rotation_matrix(states['root_rot'])
    yaw = float(heading(rot[-1]))
    anchor = anchor_rotation(yaw)
    pos = np.asarray(states['root_pos']).copy()
    pos[:, :2] -= pos[-1, :2].copy()
    local = np.einsum('ij,njk->nik', anchor.T, rot)
    six = local[:, :, :2].transpose(0,2,1).reshape(-1,6)
    hist = np.concatenate([states['q'], pos@anchor, six, states['dq'],
                           states['root_vel']@anchor, states['root_ang_vel']@anchor], axis=1)
    if hist.shape[1] != 71 or not np.isfinite(hist).all():
        raise ValueError('Invalid measured history')
    return hist.astype('float32')


def goal_condition(state, goal, elapsed):
    """World goal and deadline are immutable; only remaining time changes."""
    yaw = float(heading(rotation_matrix(state['root_rot'])))
    anchor = anchor_rotation(yaw)
    delta = (np.r_[goal['xy'], state['root_pos'][2]]-state['root_pos'])@anchor
    relative_yaw = goal['yaw']-yaw
    arrival = np.r_[goal.get('arrival_velocity_world', [0.,0.]), 0.]@anchor
    return np.r_[delta[:2], np.sin(relative_yaw), np.cos(relative_yaw),
                 max(0., goal['deadline_s']-elapsed), arrival[:2],
                 goal.get('arrival_wz', 0.), float(goal['stop'])].astype('float32')


def approach_heading_condition(condition, near_distance, far_distance):
    """Optional common heading guidance; original final goal remains scored.

    Far away, face the spatial goal; near it, restore the actual arrival heading.
    This is explicit controller assistance, not learned generator performance.
    Position/time/stop information is preserved and no simulator state is set.
    """
    if not np.isfinite([near_distance, far_distance]).all() or not 0 < near_distance < far_distance:
        raise ValueError('Invalid approach heading distances')
    result = np.asarray(condition, dtype=np.float32).copy()
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError('Invalid goal condition')
    distance = float(np.linalg.norm(result[:2]))
    if result[8] <= .5 or distance <= near_distance:
        return result, 0.
    u = np.clip((distance-near_distance)/(far_distance-near_distance), 0., 1.)
    weight = float(u*u*(3.-2.*u))
    final = float(np.arctan2(result[2], result[3]))
    bearing = float(np.arctan2(result[1], result[0]))
    delta = np.arctan2(np.sin(bearing-final), np.cos(bearing-final))
    yaw = final+weight*delta
    result[2:4] = [np.sin(yaw), np.cos(yaw)]
    return result, weight


def terminal_hold_condition(condition, was_holding, position_tolerance, yaw_tolerance):
    """Express an accepted stop endpoint as hold, with hysteresis for replanning.

    This changes only the generator query. The actual world goal, deadline and
    success checks remain fixed. Apply equally to all learned/search generators.
    """
    condition = np.asarray(condition, dtype=np.float32)
    distance = float(np.linalg.norm(condition[:2]))
    angle = abs(float(np.arctan2(condition[2], condition[3])))
    multiplier = 1.5 if was_holding else 1.
    holding = bool(condition[8] > .5 and distance <= position_tolerance*multiplier
                   and angle <= yaw_tolerance*multiplier)
    result = condition.copy()
    if holding:
        result[:5] = [0., 0., 0., 1., 0.]
        result[5:8] = 0.
    return result, holding


def decode_motion(target, current):
    """Prepend measured state at t; target contains t+1..t+H. Never teleport body."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 37 or not np.isfinite(target).all():
        raise ValueError('Expected H x 37 finite target')
    anchor = anchor_rotation(float(heading(rotation_matrix(current['root_rot']))))
    pos = target[:,28:31]@anchor.T
    pos[:,:2] += current['root_pos'][:2]
    rot = np.einsum('ij,njk->nik', anchor, rotation6_to_matrix(target[:,31:]))
    return dict(q=np.vstack([current['q'], target[:,:28]]).astype('float32'),
                root_pos=np.vstack([current['root_pos'], pos]).astype('float32'),
                root_rot=np.vstack([current['root_rot'], matrix_to_quaternion(rot)]).astype('float32'))


def anchor_root_translation(position, current_position, dt, prefix=6, max_offset=.25):
    """Optional XY-only reference-boundary correction, never a body-state write.

    Estimate the new predicted path's t=0 intercept from its first future
    samples. Removing a bounded constant translation preserves the predicted
    inter-frame motion instead of interpreting a tracking offset as velocity.
    This is execution assistance; the world goal and all joint targets stay fixed.
    """
    position = np.asarray(position)
    current = np.asarray(current_position)
    if (position.ndim != 2 or position.shape[1] != 3 or current.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(current).all()
            or not np.isfinite([dt,max_offset]).all() or dt <= 0 or max_offset <= 0
            or not isinstance(prefix, (int,np.integer)) or not 2 <= prefix < len(position)):
        raise ValueError('Invalid root translation boundary inputs')
    times = np.arange(1,prefix+1,dtype=np.float64)*dt
    centered = times-times.mean()
    points = position[1:prefix+1,:2].astype(np.float64)
    slope = (centered[:,None]*points).sum(0)/(centered*centered).sum()
    offset = points.mean(0)-slope*times.mean()-current[:2]
    length = float(np.linalg.norm(offset))
    offset *= min(1.,max_offset/max(length,1e-12))
    result = position.copy()
    result[1:,:2] -= offset.astype(result.dtype)
    return result, offset


def smooth_reference(pos, root_rot, joint_rot, boundary, dt, blend_s, kin_model, overlap=None):
    """Blend new poses from previous reference pose/velocity, in rotation space.

    Inputs are torch tensors. Joint continuation integrates local angular velocity
    via the model's DoF-to-quaternion mapping; root omega is world-frame.
    When available, blend against the already published overlapping plan.
    Velocity extrapolation is only the fallback beyond that plan or at startup.
    No actual simulator state is written, and no future actual state is used.
    """
    import torch

    def mul(a, b):
        av, aw = a[..., :3], a[..., 3:]
        bv, bw = b[..., :3], b[..., 3:]
        return torch.cat([aw*bv+bw*av+torch.linalg.cross(av,bv,dim=-1),
                          aw*bw-(av*bv).sum(-1,keepdim=True)],dim=-1)

    def exp(vector):
        angle = torch.linalg.vector_norm(vector,dim=-1,keepdim=True)
        scale = .5*torch.sinc(angle/(2*torch.pi))
        return torch.cat([vector*scale,torch.cos(angle/2)],dim=-1)

    def slerp(a,b,w):
        a=torch.nn.functional.normalize(a,dim=-1)
        b=torch.nn.functional.normalize(b,dim=-1)
        cosine=(a*b).sum(-1,keepdim=True)
        b=torch.where(cosine<0,-b,b)
        cosine=cosine.abs().clamp(0,1)
        angle=torch.acos(cosine)
        sine=torch.sin(angle).clamp_min(1e-7)
        spherical=torch.sin((1-w)*angle)/sine*a+torch.sin(w*angle)/sine*b
        linear=(1-w)*a+w*b
        return torch.nn.functional.normalize(torch.where(cosine>.9995,linear,spherical),dim=-1)

    if blend_s<=0:
        return pos,root_rot,joint_rot
    times=torch.arange(len(pos),device=pos.device,dtype=pos.dtype)*dt
    fraction=(times/blend_s).clamp(0,1)
    weight=6*fraction**5-15*fraction**4+10*fraction**3
    continuation_pos=boundary['root_pos']+times[:,None]*boundary['root_vel']
    continuation_root=mul(exp(times[:,None]*boundary['root_ang_vel']),
                          boundary['root_rot'].expand_as(root_rot))
    delta_joint=kin_model.dof_to_rot(times[:,None]*boundary['dq'])
    continuation_joint=mul(boundary['joint_rot'].expand_as(joint_rot),delta_joint)
    if overlap is not None:
        count=min(len(pos),len(overlap['root_pos']))
        continuation_pos=continuation_pos.clone()
        continuation_root=continuation_root.clone()
        continuation_joint=continuation_joint.clone()
        continuation_pos[:count]=overlap['root_pos'][:count]
        continuation_root[:count]=overlap['root_rot'][:count]
        continuation_joint[:count]=overlap['joint_rot'][:count]
    return ((1-weight[:,None])*continuation_pos+weight[:,None]*pos,
            slerp(continuation_root,root_rot,weight[:,None]),
            slerp(continuation_joint,joint_rot,weight[:,None,None]))


def encode_world_motion(future, current):
    """Diagnostic inverse of decode_motion; future ground truth is not a generator input."""
    anchor = anchor_rotation(float(heading(rotation_matrix(current['root_rot']))))
    pos = np.asarray(future['root_pos']).copy()
    pos[:, :2] -= current['root_pos'][:2]
    rot = np.einsum('ij,njk->nik', anchor.T, rotation_matrix(future['root_rot']))
    six = rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return np.concatenate([future['q'], pos@anchor, six], axis=1).astype('float32')


def align_source_target(target, source_anchor, current, alignment=None):
    """Keep a retrieved source's world placement while it advances in time.

    The returned target is still expressed in the measured current frame for
    the common decoder. Does not move the actual robot or change its goal.
    """
    source_yaw=float(source_anchor[3])
    if alignment is None:
        yaw=float(heading(rotation_matrix(current['root_rot'])))-source_yaw
        rotation=anchor_rotation(yaw)
        translation=np.asarray(current['root_pos'][:2])-np.asarray(source_anchor[:2])@rotation[:2,:2].T
        alignment=dict(yaw=yaw,translation_xy=translation.tolist())
    source=dict(q=current['q'],root_pos=np.asarray(source_anchor[:3]),
                root_rot=np.array([0.,0.,np.sin(source_yaw/2),np.cos(source_yaw/2)]))
    motion=decode_motion(target,source)
    future={k:v[1:].copy() for k,v in motion.items()}
    rotation=anchor_rotation(alignment['yaw'])
    future['root_pos']=future['root_pos']@rotation.T
    future['root_pos'][:,:2]+=np.asarray(alignment['translation_xy'])
    future['root_rot']=matrix_to_quaternion(np.einsum('ij,njk->nik',rotation,rotation_matrix(future['root_rot'])))
    return encode_world_motion(future,current),alignment
