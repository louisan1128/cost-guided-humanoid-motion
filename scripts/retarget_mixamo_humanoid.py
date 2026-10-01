#!/usr/bin/env python3
"""CPU kinematic retargeting of user-collected Mixamo BVH to MimicKit humanoid.

No simulator, learning, or changes to the source files. BVH channels are read in
declared intrinsic order. Root position channels are absolute for these Mixamo
exports (the rest OFFSET is not added a second time). Contact is a geometric
heuristic, not a physics measurement. Outputs remain preview candidates.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

# Mixamo +Z forward, +X left, +Y up -> humanoid +X forward, +Y left, +Z up.
C = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError('Degenerate bone direction')
    return v / n


def basis_yz(y, z):
    z = unit(z)
    y = unit(y - z * np.dot(y, z))
    return np.column_stack((np.cross(y, z), y, z))


def load_bvh(path):
    hierarchy, motion = path.read_text().split('MOTION', 1)
    tokens = iter(re.findall(r'[^\s{}]+|[{}]', hierarchy))
    assert next(tokens) == 'HIERARCHY'
    nodes, channel_count = [], 0

    def parse(parent, kind):
        nonlocal channel_count
        name = next(tokens)
        if kind == 'End':
            name = nodes[parent]['name'] + '_end'
        idx = len(nodes)
        node = dict(name=name.split(':')[-1], parent=parent, channels=[])
        nodes.append(node)
        assert next(tokens) == '{'
        while True:
            token = next(tokens)
            if token == '}':
                return
            if token == 'OFFSET':
                node['offset'] = np.array([float(next(tokens)) for _ in range(3)]) * .01
            elif token == 'CHANNELS':
                count = int(next(tokens))
                node['channels'] = [(next(tokens), channel_count + k) for k in range(count)]
                channel_count += count
            elif token in ('JOINT', 'End'):
                parse(idx, token)
            else:
                raise ValueError(f'Unsupported BVH token: {token}')

    assert next(tokens) == 'ROOT'
    parse(-1, 'ROOT')
    lines = motion.strip().splitlines()
    n = int(lines[0].split(':')[1])
    dt = float(lines[1].split(':')[1])
    values = np.loadtxt(lines[2:])
    if values.shape != (n, channel_count) or not np.isfinite(values).all():
        raise ValueError('Invalid BVH channel matrix')
    pos = np.zeros((n, len(nodes), 3))
    rot = np.zeros((n, len(nodes), 3, 3))
    for j, node in enumerate(nodes):
        angles = [(ch[0], k) for ch, k in node['channels'] if ch.endswith('rotation')]
        local_rot = R.from_euler(''.join(a for a, _ in angles),
                                values[:, [k for _, k in angles]], degrees=True).as_matrix() if angles else np.broadcast_to(np.eye(3), (n, 3, 3))
        translation = np.broadcast_to(node['offset'], (n, 3)).copy()
        for ch, k in node['channels']:
            if ch.endswith('position'):
                axis = 'XYZ'.index(ch[0])
                translation[:, axis] = values[:, k] * .01
        p = node['parent']
        if p < 0:
            pos[:, j], rot[:, j] = translation, local_rot
        else:
            pos[:, j] = pos[:, p] + np.einsum('nij,nj->ni', rot[:, p], translation)
            rot[:, j] = rot[:, p] @ local_rot
    return nodes, pos @ C.T, C @ rot @ C.T, dt


class Humanoid:
    def __init__(self, xml):
        self.nodes = []
        cursor = 6

        def visit(el, parent):
            nonlocal cursor
            joints = el.findall('joint')
            dim = len(joints)
            if dim not in (0, 1, 3) or el.get('quat') is not None:
                raise ValueError('Expected unrotated MimicKit humanoid rest frames')
            idx = len(self.nodes)
            self.nodes.append(dict(name=el.get('name'), parent=parent,
                                   offset=np.fromstring(el.get('pos', '0 0 0'), sep=' '),
                                   dim=dim, start=cursor,
                                   axis=np.fromstring(joints[0].get('axis'), sep=' ') if dim == 1 else None,
                                   el=el))
            cursor += dim
            for child in el.findall('body'):
                visit(child, idx)
        visit(ET.parse(xml).getroot().find('worldbody/body'), -1)
        assert cursor == 34
        self.ids = {x['name']: i for i, x in enumerate(self.nodes)}
        self.corners = {}
        for side in ('right', 'left'):
            node = self.nodes[self.ids[side + '_foot']]
            geom = node['el'].find("geom[@type='box']")
            center = np.fromstring(geom.get('pos'), sep=' ')
            size = np.fromstring(geom.get('size'), sep=' ')
            self.corners[side] = center + np.array(list(itertools.product((-1, 1), repeat=3))) * size

    def fk(self, frame):
        pos, rot = [], []
        for node in self.nodes:
            p = node['parent']
            if p < 0:
                pos.append(frame[:3]); rot.append(R.from_rotvec(frame[3:6]).as_matrix())
                continue
            k, d = node['start'], node['dim']
            local = R.from_rotvec(frame[k:k+3]).as_matrix() if d == 3 else (
                R.from_rotvec(node['axis'] * frame[k]).as_matrix() if d == 1 else np.eye(3))
            pos.append(pos[p] + rot[p] @ node['offset'])
            rot.append(rot[p] @ local)
        return np.array(pos), np.array(rot)

    def soles(self, frame):
        pos, rot = self.fk(frame)
        return np.array([np.min((self.corners[s] @ rot[self.ids[s+'_foot']].T + pos[self.ids[s+'_foot']])[:, 2]) for s in ('right', 'left')])


def seed_frames(nodes, pos, rot, target):
    ids = {x['name']: i for i, x in enumerate(nodes)}
    src = lambda name: pos[:, ids[name]]
    # Scale translation by the mean hip-to-ankle bone lengths, not total height.
    leg = np.mean([sum(np.linalg.norm(nodes[ids[s+j]]['offset']) for j in ('Leg', 'Foot')) for s in ('Right', 'Left')])
    target_leg = sum(np.linalg.norm(target.nodes[target.ids['right_'+j]]['offset']) for j in ('shin', 'foot'))
    scale = target_leg / leg
    out = np.zeros((len(pos), 34))
    out[:, :3] = src('Hips') * scale
    out[:, :2] -= out[0, :2]
    for t in range(len(pos)):
        p = lambda name: pos[t, ids[name]]
        world = {'pelvis': rot[t, ids['Hips']]}
        world['torso'] = basis_yz(p('LeftArm')-p('RightArm'), p('Neck')-p('Spine'))
        world['head'] = rot[t, ids['Head']]
        for side, s in [('right', 'Right'), ('left', 'Left')]:
            thigh = unit(p(s+'Leg') - p(s+'UpLeg'))
            shin = unit(p(s+'Foot') - p(s+'Leg'))
            # A bending knee defines its hinge axis; near extension use pelvis Y.
            axis = np.cross(thigh, shin)
            if np.linalg.norm(axis) < .03:
                axis = world['pelvis'][:, 1]
            if np.dot(axis, world['pelvis'][:, 1]) < 0:
                axis = -axis
            world[side+'_thigh'] = basis_yz(axis, -thigh)
            world[side+'_shin'] = basis_yz(axis, -shin)
            # Preserve source foot orientation, calibrated by the rest toe direction.
            toe = C @ nodes[ids[s+'ToeBase']]['offset']
            # The toe joint lies below the ankle in Mixamo rest pose.
            # Its full 3-D offset pitches the target foot box into the floor;
            # use only its horizontal heading to calibrate the foot frame.
            forward = unit(np.array([toe[0], toe[1], 0.]))
            up = np.array([0., 0., 1.])
            foot_rest = np.column_stack((forward, np.cross(up, forward), up))
            world[side+'_foot'] = rot[t, ids[s+'Foot']] @ foot_rest
            upper = unit(p(s+'ForeArm') - p(s+'Arm'))
            lower = unit(p(s+'Hand') - p(s+'ForeArm'))
            z = np.cross(upper, lower) * (1 if side == 'right' else -1)
            if np.linalg.norm(z) < .03:
                z = world['torso'][:, 2]
            sign = -1 if side == 'right' else 1
            # Exact bone direction, with elbow plane providing twist.
            def arm_basis(direction):
                y = direction * sign
                zz = unit(z - y * np.dot(z, y))
                return np.column_stack((np.cross(y, zz), y, zz))
            world[side+'_upper_arm'] = arm_basis(upper)
            world[side+'_lower_arm'] = arm_basis(lower)
        out[t, 3:6] = R.from_matrix(world['pelvis']).as_rotvec()
        for node in target.nodes[1:]:
            d, k = node['dim'], node['start']
            if d == 0:
                continue
            parent_name = target.nodes[node['parent']]['name']
            rv = R.from_matrix(world[parent_name].T @ world[node['name']]).as_rotvec()
            out[t, k:k+d] = rv if d == 3 else np.dot(rv, node['axis'])
    return out, scale, ids


def retarget(source, xml, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    nodes, pos, rot, dt = load_bvh(source)
    target = Humanoid(xml)
    frames, scale, ids = seed_frames(nodes, pos, rot, target)
    contacts = np.zeros((len(frames), 2), dtype=bool)
    for si, s in enumerate(('Right', 'Left')):
        foot = pos[:, ids[s+'ToeBase']]
        speed = np.linalg.norm(np.gradient(foot, dt, axis=0), axis=1)
        floor = np.percentile(foot[:, 2], 5)
        contacts[:, si] = (foot[:, 2] < floor + .045) & (speed < .25)
    # Reject isolated one-frame detections.
    for i in range(1, len(frames)-1):
        contacts[i] &= contacts[i-1] | contacts[i+1]
    ground = np.array([target.soles(f).min() for f in frames])
    frames[:, 2] -= np.median(ground[np.any(contacts, axis=1)]) if contacts.any() else np.median(ground)
    seed = frames.copy()
    foot_ids = [target.ids[s+'_foot'] for s in ('right', 'left')]
    desired = np.array([target.fk(f)[0][foot_ids] for f in seed])
    foot_rot = np.array([target.fk(f)[1][foot_ids] for f in seed])
    # Ease stance constraints in and out. Abrupt one-frame contact switches
    # caused knee/ankle jumps even when the horizontal root path was fixed.
    contact_weight = gaussian_filter1d(contacts.astype(float), 1.5, axis=0, mode='nearest')
    contact_xy_weight = np.maximum(contacts.astype(float), contact_weight)
    for si, side in enumerate(('right', 'left')):
        mask = contacts[:, si]
        edges = np.diff(np.r_[False, mask, False].astype(int))
        anchor = desired[:, si, :2].copy()
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            fixed = np.median(desired[start:end, si, :2], axis=0)
            anchor[max(0, start-4):min(len(frames), end+4)] = fixed
        desired[:, si, :2] = (1-contact_xy_weight[:, si, None])*desired[:, si, :2] + contact_xy_weight[:, si, None]*anchor
        for t in range(len(frames)):
            src = R.from_matrix(foot_rot[t, si])
            yaw = np.arctan2(foot_rot[t, si, 1, 0], foot_rot[t, si, 0, 0])
            flat = R.from_euler('z', yaw)
            w = contact_weight[t, si]
            foot_rot[t, si] = (src * R.from_rotvec(w * (src.inv()*flat).as_rotvec())).as_matrix()
        # Smooth short orientation spikes caused by framewise BVH foot rotations
        # and the changing stance weight; normalize the filtered quaternions.
        quat = R.from_matrix(foot_rot[:, si]).as_quat()
        for t in range(1, len(quat)):
            if np.dot(quat[t], quat[t-1]) < 0:
                quat[t] *= -1
        quat = gaussian_filter1d(quat, 1.3, axis=0, mode='nearest')
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        foot_rot[:, si] = R.from_quat(quat).as_matrix()
        for t in range(len(frames)):
            low = np.min((target.corners[side] @ foot_rot[t, si].T)[:, 2])
            grounded = -low
            airborne = max(desired[t, si, 2], grounded)
            w = contact_weight[t, si]
            desired[t, si, 2] = w*grounded + (1-w)*airborne
    # Preserve the source horizontal root path: framewise IK otherwise
    # moves the whole character backward to satisfy stance-foot targets.
    active = np.r_[2, 20:34]
    lower, upper = np.full(len(active), -np.inf), np.full(len(active), np.inf)
    # Anatomical hinge limits from the target XML. Spherical rotations remain
    # exp-maps: component-wise XML Euler bounds would not be valid limits here.
    for k in (23, 30):
        j = int(np.flatnonzero(active == k)[0]); lower[j] = 0.; upper[j] = np.deg2rad(160)
    prev = seed[0, active].copy()
    errors = []
    for t, original in enumerate(seed):
        def residual(x):
            f = original.copy(); f[active] = x
            pp, rr = target.fk(f)
            point = (pp[foot_ids] - desired[t]) * np.where(contacts[t, :, None], 30., 12.)
            orient = (rr[foot_ids] - foot_rot[t]).ravel() * .5
            prior = (x-original[active]) * .08
            continuity = (x-prev) * .03
            floor = np.minimum(target.soles(f), 0.) * 50.
            return np.r_[point.ravel(), orient, prior, continuity, floor]
        x0 = np.clip(original[active], lower+1e-8, upper-1e-8)
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=35, ftol=1e-5)
        frames[t, active] = result.x
        prev = result.x.copy()
        errors.append(float(np.max(np.linalg.norm(target.fk(frames[t])[0][foot_ids]-desired[t], axis=1))))
    if not np.isfinite(frames).all():
        raise ValueError('Nonfinite retargeted motion')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        pickle.dump(dict(loop_mode=0, fps=1/dt, frames=frames.astype(np.float32).tolist()), f)
    soles = np.array([target.soles(f) for f in frames])
    pp = np.array([target.fk(f)[0] for f in frames])
    v = np.linalg.norm(np.diff(pp[:, foot_ids, :2], axis=0)/dt, axis=-1)
    stance_pairs = contacts[1:] & contacts[:-1]
    report = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  asset=str(xml.resolve()), asset_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                  frames=len(frames), fps=1/dt, duration_s=(len(frames)-1)*dt, translation_scale=scale,
                  status='kinematic_candidate_not_physics_validated', source_owner='user-collected Mixamo',
                  contact_fraction=contacts.mean(0).tolist(), min_sole_height_m=float(soles.min()),
                  max_foot_target_error_m=max(errors),
                  mean_stance_ankle_xy_speed_mps=float(v[stance_pairs].mean()) if stance_pairs.any() else None,
                  root_speed_first_last_0p5s_mps=[float(x) for x in (
                      np.linalg.norm(np.diff(frames[:, :2], axis=0)/dt, axis=1)[:max(1, round(.5/dt))].mean(),
                      np.linalg.norm(np.diff(frames[:, :2], axis=0)/dt, axis=1)[-max(1, round(.5/dt)):].mean())])
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.qa.npz'), source_positions=pos, source_names=[n['name'] for n in nodes],
                        source_parents=[n['parent'] for n in nodes], target_positions=pp,
                        target_names=[n['name'] for n in target.nodes], target_parents=[n['parent'] for n in target.nodes],
                        contact=contacts, sole_height=soles)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--asset', type=Path, default=Path('/home/an/Desktop/MimicKit-main/data/assets/humanoid/humanoid.xml'))
    args = parser.parse_args()
    retarget(args.source, args.asset, args.output)
