"""Goal-reaching AMP teacher: actual state -> PD action, independent of tracker."""
import math
from collections import deque
import torch
from envs.amp_env import AMPEnv
from util import torch_util


def goal_observation(root_pos, heading, goal_xy, goal_yaw, max_distance=None):
    delta = goal_xy - root_pos[:, :2]
    if max_distance is not None:
        if max_distance <= 0:
            raise ValueError('max observation distance must be positive')
        delta = delta*(max_distance/delta.norm(dim=-1).clamp_min(1e-6)).clamp(max=1)[:, None]
    c, s = torch.cos(heading), torch.sin(heading)
    error = goal_yaw - heading
    return torch.stack((c*delta[:, 0]+s*delta[:, 1],
                        -s*delta[:, 0]+c*delta[:, 1],
                        torch.sin(error), torch.cos(error)), dim=-1)


def arrival_alignment_state(previous, distance, speed, config):
    """Latched arrival supervisor; keep world goal/deadline unchanged."""
    enter=config['enter_distance'];leave=config['leave_distance'];limit=config['enter_speed']
    if not (0 < enter < leave and limit > 0):
        raise ValueError('Sequential arrival requires 0 < enter < leave and positive speed')
    return (previous & (distance < leave)) | ((distance <= enter) & (speed <= limit))


def committed_turn_sign(previous, error, alignment, entered, route):
    if route not in ('shortest','left'):raise ValueError('Unknown final turn route')
    choice=torch.ones_like(error) if route=='left' else error.sign()
    sign=torch.where(entered,choice,previous)
    # Release only after reaching the final part of the selected route, or a
    # small overshoot. Negative 45deg on a left route must NOT release at start.
    reached=((sign*error>=0)&(error.abs()<=math.radians(60))) | (error.abs()<=math.radians(15))
    return torch.where(~alignment|reached,torch.zeros_like(sign),sign)


def routed_turn_error(error, sign):
    directed=sign*torch.remainder(sign*error,2*math.pi)
    return torch.where(sign!=0,directed,error)


def alignment_heading_request(heading, target, limit_deg=None):
    if limit_deg is None:return target
    if not 0 < limit_deg <= 180:raise ValueError('Invalid alignment heading request limit')
    error=torch.atan2(torch.sin(target-heading),torch.cos(target-heading))
    return heading+error.clamp(-math.radians(limit_deg),math.radians(limit_deg))


def resolve_fixed_goal(root_pos, heading, fixed):
    """Resolve a user test command ONCE at reset; return fixed world targets."""
    x, y, yaw = fixed['x'], fixed['y'], math.radians(fixed['yaw_deg'])
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError('Fixed goal must be finite')
    frame = fixed.get('frame', 'start')
    if frame == 'start':
        c, s = heading.cos(), heading.sin()
        xy = root_pos[:, :2]+torch.stack((c*x-s*y, s*x+c*y), -1)
        return xy, heading+yaw
    if frame == 'world':
        xy = root_pos.new_tensor([x, y]).expand(len(root_pos), 2).clone()
        return xy, torch.full_like(heading, yaw)
    raise ValueError('Fixed goal frame must be start or world')


def goal_metrics(root_pos, heading, velocity, angular_z, goal_xy, goal_yaw, config):
    delta = goal_xy-root_pos[:, :2]
    distance = delta.norm(dim=-1)
    direction = delta / distance.clamp_min(1e-6)[:, None]
    yaw_error = torch.atan2(torch.sin(goal_yaw-heading), torch.cos(goal_yaw-heading))
    speed = velocity[:, :2].norm(dim=-1)
    valid = ((distance <= config['position_tolerance']) &
             (yaw_error.abs() <= math.radians(config['yaw_tolerance_deg'])) &
             (speed <= config['speed_tolerance']) &
             (angular_z.abs() <= config['angular_tolerance']))
    return distance, direction, yaw_error, speed, valid


def travel_potential(distance, direction, heading, config):
    """State potential: reward turning toward a goal without positive-only progress."""
    steering = config.get('steering', {})
    amount = steering.get('turn_potential_m_per_rad', 0.)
    if amount < 0:
        raise ValueError('Turn potential scale must be nonnegative')
    if not amount:
        return -distance
    angle = torch.atan2(direction[:, 1], direction[:, 0])-heading
    angle = torch.atan2(angle.sin(), angle.cos()).abs()
    radius = steering.get('arrival_radius', .6)
    deadband = steering.get('arrival_position_deadband', 0.)
    weight = ((distance-deadband)/(radius-deadband)).clamp(0, 1)
    return -distance-amount*weight*angle


def steering_forward_speed(distance, travel_error, desired_speed, steering):
    speed = desired_speed*travel_error.cos().clamp_min(0).square()
    floor = steering.get('turn_forward_speed', 0.)
    if floor < 0:
        raise ValueError('Turn forward speed must be nonnegative')
    # Far from arrival, allow a slow forward arc even when the goal is behind.
    return torch.where(distance > steering.get('turn_forward_min_distance', 1.),
                       torch.maximum(speed, desired_speed.clamp(max=floor)), speed)


def focused_turn_magnitude(count, device, focus):
    """Include exact opposite goals without treating +180/-180 as different tasks."""
    lo, hi = focus['absolute_range_deg']
    probability = focus.get('opposite_probability', 0.)
    if not 0 <= lo < hi <= 180 or not 0 <= probability <= 1:
        raise ValueError('Invalid turn magnitude sampling')
    magnitude = lo+(hi-lo)*torch.rand(count, device=device)
    if probability:
        exact = torch.rand(count, device=device) < probability
        magnitude = torch.where(exact, torch.full_like(magnitude, 180.), magnitude)
    return magnitude


def goal_reward(distance, direction, yaw_error, velocity, angular_z, previous_distance, dt, config, heading=None, previous_potential=None, alignment=None):
    remaining = (distance-config.get('stop_deadband', 0.)).clamp_min(0)
    desired_speed = torch.minimum(torch.full_like(distance, config['walking_speed']),
                                  config['deceleration_gain']*remaining)
    desired_velocity = direction*desired_speed[:, None]
    steering = config.get('steering')
    travel_error = None
    if steering:
        if heading is None:
            raise ValueError('Steering reward requires actual heading')
        travel_yaw = torch.atan2(direction[:,1],direction[:,0])
        travel_error = torch.atan2(torch.sin(travel_yaw-heading),torch.cos(travel_yaw-heading))
        forward = torch.stack((heading.cos(),heading.sin()),-1)
        desired_velocity = forward*steering_forward_speed(distance, travel_error, desired_speed, steering)[:,None]
    if alignment is not None:
        # Once braking has finished, allow small balance steps but request no travel.
        correction=(config["sequential_arrival"].get("position_gain",0.)*remaining).clamp(max=config["sequential_arrival"].get("position_speed_cap",.08))
        alignment_velocity=direction*correction[:,None]
        desired_velocity=torch.where(alignment[:,None],alignment_velocity,desired_velocity)
    velocity_r = torch.exp(-4*(velocity[:, :2]-desired_velocity).square().sum(-1))
    proximity = torch.exp(-2*distance.square())
    near = torch.exp(-distance.square()/config['yaw_activation_distance']**2)
    yaw_r = near*torch.exp(-4*yaw_error.square())
    if alignment is not None: yaw_r=alignment.to(distance.dtype)*torch.exp(-4*yaw_error.square())
    stop_r = torch.exp(-distance.square()/config['position_tolerance']**2
                      -yaw_error.square()/math.radians(config['yaw_tolerance_deg'])**2
                      -velocity[:, :2].square().sum(-1)/config['speed_tolerance']**2
                      -angular_z.square()/config['angular_tolerance']**2)
    # Signed distance change: moving away pays back progress; never positive-only.
    progress = (previous_distance-distance)/dt
    if steering and steering.get('turn_potential_m_per_rad', 0.):
        if previous_potential is None:
            raise ValueError('Turn shaping requires previous state potential')
        progress = (travel_potential(distance, direction, heading, config)-previous_potential)/dt
    terms = dict(progress=progress, velocity=velocity_r, proximity=proximity,
                 yaw=yaw_r, stop=stop_r)
    if 'backward' in config['reward_weights']:
        if heading is None:
            raise ValueError('Backward penalty requires heading')
        forward_velocity = velocity[:,0]*heading.cos()+velocity[:,1]*heading.sin()
        terms['backward'] = -( (-forward_velocity-config.get('backward_speed_tolerance', .05)).clamp_min(0) ).square()
    if 'arrival_valid' in config['reward_weights']:
        # Reward occupancy of the SAME physical acceptance set used in evaluation.
        # A nearby stationary robot outside the position band earns no such bonus.
        terms['arrival_valid'] = ((distance <= config['position_tolerance']) &
            (yaw_error.abs() <= math.radians(config['yaw_tolerance_deg'])) &
            (velocity[:, :2].norm(dim=-1) <= config['speed_tolerance']) &
            (angular_z.abs() <= config['angular_tolerance'])).to(distance.dtype)
    if 'arrival_velocity' in config['reward_weights']:
        # Near the goal, measure velocity toward its actual residual position.
        # The broad travel reward can give an almost stationary robot high credit.
        radius = config.get('arrival_velocity_radius', .75)
        scale = config.get('arrival_velocity_scale', .12)
        if radius <= 0 or scale <= 0:
            raise ValueError('Arrival velocity radius/scale must be positive')
        direct = direction*desired_speed[:, None]
        if config.get('arrival_velocity_stop_inside',False):
            direct=torch.where((distance<=config['position_tolerance'])[:,None],torch.zeros_like(direct),direct)
        if alignment is not None:
            direct=torch.where(alignment[:,None],alignment_velocity,direct)
        terms['arrival_velocity'] = torch.exp(-(distance/radius).square())*torch.exp(
            -(velocity[:, :2]-direct).square().sum(-1)/scale**2)
    if steering:
        # Travel bearing and final yaw have different roles; reward them separately.
        radius=steering.get('arrival_radius',1.)
        deadband=steering.get('arrival_position_deadband',0.)
        if not 0 <= deadband < radius:
            raise ValueError('Steering deadband must be nonnegative and below radius')
        travel_weight=((distance-deadband)/(radius-deadband)).clamp(0,1)
        if alignment is not None:travel_weight=(~alignment).to(distance.dtype)
        terms['travel_heading']=travel_weight*torch.exp(-2*travel_error.square())
    if 'angular' in config['reward_weights']:
        # Near arrival, correct heading slowly and damp rotation inside a small yaw band.
        desired_wz = (1.5*(yaw_error.abs()-.1).clamp_min(0)*yaw_error.sign()).clamp(-.4, .4)
        if alignment is not None:
            cap=config['sequential_arrival'].get('max_alignment_rate',.5)
            desired_wz=(1.5*(yaw_error.abs()-.1).clamp_min(0)*yaw_error.sign()).clamp(-cap,cap)
        if steering:
            limit=steering.get('max_yaw_rate',.8)
            if 'moving_yaw_speed_scale' in steering:
                scale=steering['moving_yaw_speed_scale']
                minimum=steering.get('moving_yaw_rate_min',.2)
                if scale <= 0 or not 0 <= minimum <= limit:
                    raise ValueError('Invalid moving yaw rate limits')
                # Turn more gently while initial forward momentum remains.
                speed=velocity[:, :2].norm(dim=-1)
                limit=minimum+(limit-minimum)*torch.exp(-(speed/scale).square())
            travel_wz=(steering.get('yaw_gain',1.5)*travel_error).clamp(-limit,limit)
            # Blend rates, not opposite heading vectors; final yaw takes over near arrival.
            desired_wz=travel_weight*travel_wz+(1-travel_weight)*desired_wz
            rate_scale=torch.full_like(distance,steering.get('yaw_rate_scale',.4))
            if alignment is not None:
                rate_scale=torch.where(alignment,torch.full_like(rate_scale,config['sequential_arrival'].get('yaw_rate_scale',steering.get('yaw_rate_scale',.4))),rate_scale)
            terms['angular']=torch.exp(-((angular_z-desired_wz)/rate_scale).square())
            if 'turn_drive' in config['reward_weights']:
                gate=((travel_error.abs()-math.radians(30))/math.radians(30)).clamp(0,1)*travel_weight
                signed=(angular_z*travel_error.sign()/limit).clamp(-1,1)
                overspeed=((angular_z.abs()-limit).clamp_min(0)/limit).square()
                terms['turn_drive']=gate*(signed-overspeed)
        else:
            terms['angular'] = near*torch.exp(-((angular_z-desired_wz)/.2).square())
    if alignment is not None:
        if 'alignment_heading' in config['reward_weights']:
            # Dense final-heading feedback even where exp(-4*error^2) is tiny.
            # Symmetric in signed angle; active only after arrival alignment.
            terms['alignment_heading'] = alignment.to(distance.dtype)*(1-yaw_error.abs().clamp(max=math.pi)/math.pi)
        if 'alignment_position' in config['reward_weights']:
            terms['alignment_position']=alignment.to(distance.dtype)*torch.exp(-(distance/config['position_tolerance']).square())
        cap=config['sequential_arrival'].get('max_alignment_rate',.5)
        excess=((angular_z.abs()-cap).clamp_min(0)/cap).square()
        drive=(angular_z*yaw_error.sign()/cap).clamp(-1,1)
        drive-=config['sequential_arrival'].get('overspeed_weight',0.)*excess
        terms['alignment_drive']=alignment.to(distance.dtype)*drive*(yaw_error.abs()>math.radians(config['yaw_tolerance_deg']))
    reward = sum(config['reward_weights'][k]*v for k, v in terms.items())
    return reward, terms


def contact_slip(foot_velocity, foot_force, threshold=10.):
    """Mean squared horizontal link speed only while feet exert ground force."""
    contact = foot_force[..., 2] > threshold
    return (foot_velocity[..., :2].square().sum(-1)*contact).sum(-1)/contact.sum(-1).clamp_min(1)


def balanced_touchdown_reward(length, previous_other, known_other):
    # No reward for a shuffle/backward step; balance is judged at completed contacts.
    extent=((length-.12)/.25).clamp(0,1)
    balance=torch.exp(-((length-previous_other)/.2).square())
    return extent*torch.where(known_other, balance, torch.ones_like(balance))


def humanoid_foot_clearance(positions, rotations):
    """Lowest box corner above z=0; generic humanoid.xml geometry, not a G1 helper."""
    corners = torch.tensor([[.045+sx*.0885, sy*.045, -.0225+sz*.0275]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                           device=positions.device, dtype=positions.dtype)
    shape = (*positions.shape[:-1], 8, 3)
    offsets = torch_util.quat_rotate(rotations[..., None, :].expand(*shape[:-1], 4),
                                    corners.expand(shape))
    return (positions[..., None, 2]+offsets[..., 2]).min(-1).values


def upper_pose_error(rotations, references):
    """Nearest complete upper-body example; quaternion sign invariant, no phase claim."""
    dot = (rotations[:, None]*references[None]).sum(-1).abs().clamp(max=1.)
    return (2*torch.acos(dot)).square().mean(-1).min(-1).values


def reference_motion_reward(rotations, dof_velocity, refs, ref_velocity, ref_speed, desired_speed, velocity_scale, velocity_weight=.1, moving_target_speed=.4, stationary_target_speed=.1, require_active=None,
                            reference_yaw_rate=None, requested_turn=None, minimum_turn_rate=.1):
    """Phase-free nearest *whole state* reward; not canonical phase-synced DeepMimic."""
    dot = torch.einsum('njc,kjc->nkj', rotations, refs).abs().clamp(max=1.)
    pose = (2*torch.acos(dot)).square().mean(-1)
    ratio = (desired_speed[:, None]/ref_speed.clamp_min(.25)[None]).clamp(max=1.5)
    v = dof_velocity/velocity_scale
    rv = ref_velocity/velocity_scale
    velocity = (v.square().mean(-1)[:, None]+ratio.square()*rv.square().mean(-1)[None]
                -2*ratio*(v@rv.T)/v.shape[-1]).clamp_min(0)
    if torch.is_tensor(velocity_weight):velocity_weight=velocity_weight[:,None]
    cost = 2*pose+velocity_weight*velocity
    if not 0 <= stationary_target_speed < moving_target_speed:
        raise ValueError('Reference target speed thresholds must be ordered')
    allowed = torch.where(desired_speed[:, None]>moving_target_speed, ref_speed[None]>.4,
                          torch.where(desired_speed[:, None]<stationary_target_speed, ref_speed[None]<.15,
                                      torch.ones_like(cost, dtype=torch.bool)))
    if require_active is not None:
        # Slow approach outside the accepted arrival region must still have an
        # active candidate; the ordinary slow-speed mask contains only stand.
        allowed |= require_active[:, None] & (desired_speed[:, None] < stationary_target_speed) & (ref_speed[None] > .15)
        allowed &= (~require_active[:, None]) | (ref_speed[None] > .15)
    if requested_turn is not None:
        if reference_yaw_rate is None or minimum_turn_rate <= 0:
            raise ValueError('Directional motion prior requires reference yaw rate and positive threshold')
        turning=requested_turn != 0
        compatible=reference_yaw_rate[None]*requested_turn.sign()[:,None] >= minimum_turn_rate
        allowed &= (~turning[:,None]) | compatible
    if not bool(allowed.any(-1).all()):
        raise ValueError('Reference pool has no compatible motion (speed or requested turn)')
    return torch.exp(-cost.masked_fill(~allowed, float('inf')).min(-1).values)


class GoalAMPEnv(AMPEnv):
    def __init__(self, env_config, **kwargs):
        self.goal_config = env_config['goal']
        c = self.goal_config
        for name in ('distance_range', 'bearing_range_deg', 'final_yaw_offset_deg'):
            lo, hi = c[name]
            if lo > hi or (name == 'distance_range' and lo < 0):
                raise ValueError(name)
        if not 0 <= c.get('stand_hold_goal_probability', 0) <= 1:
            raise ValueError('stand_hold_goal_probability')
        if not 0 <= c['stand_reset_probability'] <= 1:
            raise ValueError('stand_reset_probability')
        for name in ('position_tolerance', 'yaw_tolerance_deg', 'speed_tolerance',
                     'angular_tolerance', 'walking_speed', 'deceleration_gain',
                     'yaw_activation_distance', 'hold_seconds'):
            if c[name] <= 0:
                raise ValueError(name)
        self._goal_reset_sampling = False
        self._manual_goal_commands = deque()
        self._manual_goal_panel = None
        if c.get('interactive_goal') and (not kwargs.get('visualize') or kwargs.get('num_envs') != 1):
            raise ValueError('Manual goal control requires a visualized single environment')
        super().__init__(env_config=env_config, **kwargs)

    def _setup_gui(self):
        super()._setup_gui()
        manual=self.goal_config.get('interactive_goal')
        if not manual:
            return
        if manual.get('click'):
            from humanoid_imitation.planning.click_goal import ClickGoalPanel
            self._manual_goal_panel=ClickGoalPanel(self._manual_goal_commands.append)
            return
        for key,action in [('I','forward'),('K','back'),('J','left'),('L','right'),
                           ('U','yaw_left'),('O','yaw_right'),('H','restore')]:
            self._engine.register_keyboard_callback(key,lambda a=action:self._manual_goal_commands.append(a))
        from humanoid_imitation.planning.manual_goal_panel import ManualGoalPanel
        self._manual_goal_panel=ManualGoalPanel(self._manual_goal_commands.append,manual)
        print('Manual target: I/K forward/back, J/L left/right, U/O final heading, H restore. Click viewport for keyboard focus.')

    def _update_misc(self):
        super()._update_misc()
        if not self.goal_config.get('interactive_goal') or not self._manual_goal_commands:
            return
        manual=self.goal_config['interactive_goal']
        basis=self._manual_goal_basis_heading
        forward=torch.stack((basis.cos(),basis.sin()))
        left=torch.stack((-basis.sin(),basis.cos()))
        while self._manual_goal_commands:
            action=self._manual_goal_commands.popleft()
            if isinstance(action,tuple) and action[0]=='set_goal':
                self.goal_xy[0]=self.goal_xy.new_tensor(action[1:3])
                self.goal_yaw[0]=action[3]
            elif action in ('forward','back','left','right'):
                direction=forward if action in ('forward','back') else left
                sign=1 if action in ('forward','left') else -1
                self.goal_xy[0] += sign*manual['position_step']*direction
            elif action in ('yaw_left','yaw_right'):
                sign=1 if action=='yaw_left' else -1
                angle=self.goal_yaw[0]+sign*math.radians(manual['yaw_step_deg'])
                self.goal_yaw[0]=torch.atan2(angle.sin(),angle.cos())
            elif action=='restore':
                self.goal_xy[0]=self._manual_initial_goal_xy
                self.goal_yaw[0]=self._manual_initial_goal_yaw
        self._refresh_changed_goal()

    def set_goal_world(self, x, y, yaw):
        """Explicit single-env task change for scripted physical collection."""
        if self.get_num_envs() != 1 or not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError('Scripted world goals require one env and finite values')
        self.goal_xy[0] = self.goal_xy.new_tensor([x, y])
        self.goal_yaw[0] = yaw
        self._refresh_changed_goal()
        self._update_observations(None)

    def _refresh_changed_goal(self):
        # Changing the task must not create fake progress or keep old hold success.
        cid=self._get_char_id()
        pos=self._engine.get_root_pos(cid)
        heading=torch_util.calc_heading(self._engine.get_root_rot(cid))
        delta=self.goal_xy-pos[:,:2];distance=delta.norm(dim=-1)
        self.previous_distance[:]=distance
        self.previous_goal_potential[:]=travel_potential(distance,delta/distance.clamp_min(1e-6)[:,None],heading,self.goal_config)
        self.hold_time[:]=0
        self.ever_held[:]=False
        if self.goal_config.get("sequential_arrival"):
            self.arrival_align.zero_();self.arrival_turn_sign.zero_()

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        n = self.get_num_envs()
        self.goal_xy = torch.zeros(n, 2, device=self._device)
        self.goal_yaw = torch.zeros(n, device=self._device)
        self.previous_distance = torch.zeros(n, device=self._device)
        self.previous_goal_potential = torch.zeros(n, device=self._device)
        self.hold_time = torch.zeros(n, device=self._device)
        self.ever_held = torch.zeros(n, device=self._device, dtype=torch.bool)
        self.reset_standing = torch.zeros(n, device=self._device, dtype=torch.bool)
        self.reward_terms = {}
        self.arrival_align = torch.zeros(n,device=self._device,dtype=torch.bool)
        self.arrival_turn_sign = torch.zeros(n,device=self._device)
        self._gait_air_steps=torch.zeros(n,2,device=self._device,dtype=torch.long)
        self._gait_last_length=torch.zeros(n,2,device=self._device)
        self._gait_known=torch.zeros(n,2,device=self._device,dtype=torch.bool)

    def _sample_motion_times(self, n):
        if not self._goal_reset_sampling:
            return super()._sample_motion_times(n)
        c = self.goal_config
        standing = torch.rand(n, device=self._device) < c['stand_reset_probability']
        walk_ids = c.get('walk_motion_ids', [c['walk_motion_id']])
        stand_ids = c.get('stand_motion_ids', [c['stand_motion_id']])
        walk_choice = torch.tensor(walk_ids,device=self._device)[torch.randint(len(walk_ids),(n,),device=self._device)] if len(walk_ids)>1 else torch.full((n,),walk_ids[0],device=self._device)
        stand_choice = torch.tensor(stand_ids,device=self._device)[torch.randint(len(stand_ids),(n,),device=self._device)] if len(stand_ids)>1 else torch.full((n,),stand_ids[0],device=self._device)
        ids = torch.where(standing, stand_choice, walk_choice).long()
        times = torch.rand(n, device=self._device)*c['walk_reset_max_time']
        lo, hi = c['stand_time_range']
        times[standing] = lo+(hi-lo)*torch.rand(int(standing.sum()), device=self._device)
        focus = c.get('walk_reset_focus')
        if focus:
            probability = focus['probability']
            lo, hi = focus['time_range']
            if not 0 <= probability <= 1 or not 0 <= lo < hi:
                raise ValueError('Invalid walk reset focus')
            mask = (~standing) & (torch.rand(n, device=self._device) < probability)
            choices = torch.tensor(focus['motion_ids'], device=self._device)
            ids[mask] = choices[torch.randint(len(choices), (int(mask.sum()),), device=self._device)]
            if bool((self._motion_lib.get_motion_length(ids[mask]) < hi).any()):
                raise ValueError('Focused reset interval exceeds motion length')
            times[mask] = lo+(hi-lo)*torch.rand(int(mask.sum()), device=self._device)
        self._last_reset_standing = standing
        return ids, times

    def _reset_ref_motion(self, ids):
        self._goal_reset_sampling = True
        try:
            super()._reset_ref_motion(ids)
        finally:
            self._goal_reset_sampling = False

    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        clearance = self.goal_config.get('reset_foot_clearance_m')
        if clearance is not None and len(env_ids):
            if not 0 <= clearance <= .02:
                raise ValueError('Reset foot clearance must be between 0 and 2cm')
            # Reset only: lift penetrating source poses without changing q/dq or
            # horizontal momentum. No pose correction during a physical rollout.
            pos, rot = self._kin_char_model.forward_kinematics(
                self._ref_root_pos[env_ids], self._ref_root_rot[env_ids],
                self._ref_joint_rot[env_ids])
            names = self._kin_char_model.get_body_names()
            feet = [names.index('right_foot'), names.index('left_foot')]
            lowest = humanoid_foot_clearance(pos[:, feet], rot[:, feet]).min(-1).values
            root = self._ref_root_pos[env_ids].clone()
            root[:, 2] += (clearance-lowest).clamp_min(0)
            self._engine.set_root_pos(env_ids, self._get_char_id(), root)
        measured = self.goal_config.get('measured_standing_reset')
        if measured and len(env_ids):
            from humanoid_imitation.envs.measured_goal_reset import MeasuredGoalReset
            if not hasattr(self, '_measured_reset_bank'):
                self._measured_reset_bank = MeasuredGoalReset(measured, self._device)
            mask, states = self._measured_reset_bank.sample(len(env_ids))
            self._measured_reset_mask = mask
            self._measured_reset_states = states
            self._measured_reset_count = getattr(self, '_measured_reset_count', 0) + int(mask.sum())
            self._measured_reset_total = getattr(self, '_measured_reset_total', 0) + len(env_ids)
            ids = env_ids[mask]
            if len(ids):
                for key, method in [('q','set_dof_pos'), ('dq','set_dof_vel'),
                                    ('root_pos','set_root_pos'), ('root_rot','set_root_rot'),
                                    ('root_vel','set_root_vel'), ('root_ang_vel','set_root_ang_vel')]:
                    getattr(self._engine, method)(ids, self._get_char_id(), states[key])
                self._last_reset_standing[mask] = True

    def _reset_disc_hist(self, env_ids):
        super()._reset_disc_hist(env_ids)
        if self.goal_config.get('measured_standing_reset') and len(env_ids):
            ids = env_ids[self._measured_reset_mask]
            if not len(ids):
                return
            states = self._measured_reset_states
            n, steps = states['history_q'].shape[:2]
            if steps != self._num_disc_obs_steps:
                raise ValueError('Measured history must match AMP discriminator history')
            q = states['history_q'].reshape(-1, 28)
            joint_rot = self._kin_char_model.dof_to_rot(q)
            body_pos, _ = self._kin_char_model.forward_kinematics(
                states['history_root_pos'].reshape(-1, 3),
                states['history_root_rot'].reshape(-1, 4), joint_rot)
            values = dict(root_pos=states['history_root_pos'], root_rot=states['history_root_rot'],
                          root_vel=states['history_root_vel'], root_ang_vel=states['history_root_ang_vel'],
                          dof_vel=states['history_dq'],
                          joint_rot=joint_rot.reshape(n, steps, *joint_rot.shape[1:]),
                          body_pos=body_pos.reshape(n, steps, *body_pos.shape[1:]))
            for key, value in values.items():
                getattr(self, '_disc_hist_'+key).fill(ids, value)

    def _reset_envs(self, ids):
        super()._reset_envs(ids)
        if len(ids):
            cid = self._get_char_id()
            pos = self._engine.get_root_pos(cid)[ids]
            heading = torch_util.calc_heading(self._engine.get_root_rot(cid)[ids])
            c = self.goal_config
            def sample(key):
                lo, hi = c[key]
                return lo+(hi-lo)*torch.rand(len(ids), device=self._device)
            distance = sample('distance_range')
            bearing_delta=sample('bearing_range_deg')
            focus=c.get('turn_goal_focus')
            if focus:
                probability=focus['probability'];lo,hi=focus['absolute_range_deg']
                if not 0 <= probability <= 1 or not 0 <= lo < hi <= 180:
                    raise ValueError('Invalid turn goal focus')
                mask=torch.rand(len(ids),device=self._device)<probability
                magnitude=focused_turn_magnitude(len(ids),self._device,focus)
                sign=torch.where(torch.rand(len(ids),device=self._device)<.5,-1.,1.)
                bearing_delta=torch.where(mask,sign*magnitude,bearing_delta)
            final_offset=sample('final_yaw_offset_deg')
            rehearsal=c.get('arrival_yaw_rehearsal')
            if rehearsal:
                probability=rehearsal['probability']
                if not 0 <= probability <= 1:
                    raise ValueError('Invalid arrival yaw rehearsal')
                mask=torch.rand(len(ids),device=self._device)<probability
                span=rehearsal['bearing_limit_deg'];yaw_span=rehearsal['yaw_offset_limit_deg']
                bearing_delta=torch.where(mask,(2*torch.rand(len(ids),device=self._device)-1)*span,bearing_delta)
                final_offset=torch.where(mask,(2*torch.rand(len(ids),device=self._device)-1)*yaw_span,final_offset)
            legacy=c.get('legacy_goal_probability',0.)
            if legacy:
                keep=torch.rand(len(ids),device=self._device)<legacy
                distance=torch.where(keep,.5+torch.rand(len(ids),device=self._device),distance)
                bearing_delta=torch.where(keep,-15+30*torch.rand(len(ids),device=self._device),bearing_delta)
                final_offset=torch.where(keep,torch.zeros_like(final_offset),final_offset)
            bearing = heading+bearing_delta*math.pi/180
            self.goal_xy[ids] = pos[:, :2]+distance[:, None]*torch.stack((bearing.cos(), bearing.sin()), -1)
            self.goal_yaw[ids] = bearing+final_offset*math.pi/180
            # Only static reset states receive zero-distance hold goals.
            # Moving states keep a stopping distance; no instantaneous stop teleport.
            probability = c.get('stand_hold_goal_probability', 0)
            hold = torch.zeros(len(ids), device=self._device, dtype=torch.bool)
            if probability > 0:
                hold = self._last_reset_standing & (torch.rand(len(ids), device=self._device) < probability)
            hold_ids = ids[hold]
            self.goal_xy[hold_ids] = pos[hold, :2]
            self.goal_yaw[hold_ids] = heading[hold]
            distance[hold] = 0
            turn_practice=c.get('stand_alignment_rehearsal')
            if turn_practice:
                if not c.get('sequential_arrival') or not 0<=turn_practice['probability']<=1:
                    raise ValueError('Stand alignment rehearsal requires sequential arrival and a valid probability')
                mask=self._last_reset_standing & ~hold & (torch.rand(len(ids),device=self._device)<turn_practice['probability'])
                limit=turn_practice['yaw_limit_deg']
                if not 0<limit<=180:raise ValueError('Invalid stand alignment rehearsal yaw limit')
                self.goal_xy[ids[mask]]=pos[mask,:2]
                self.goal_yaw[ids[mask]]=heading[mask]+(2*torch.rand(int(mask.sum()),device=self._device)-1)*math.radians(limit)
                distance[mask]=0
            if c.get('measured_standing_reset'):
                from humanoid_imitation.envs.measured_goal_reset import rehearsal_yaw_offsets
                mask = self._measured_reset_mask
                selected = ids[mask]
                self.goal_xy[selected] = pos[mask, :2] + self._measured_reset_states['goal_offset_world_xy']
                self.goal_yaw[selected] = heading[mask] + rehearsal_yaw_offsets(len(selected), self._device)
                distance[mask] = (self.goal_xy[selected]-pos[mask, :2]).norm(dim=-1)
            fixed = c.get('fixed_test_goal')
            if fixed is not None:
                self.goal_xy[ids], self.goal_yaw[ids] = resolve_fixed_goal(pos, heading, fixed)
                distance = (self.goal_xy[ids]-pos[:, :2]).norm(dim=-1)
            if c.get('interactive_goal'):
                if (c['interactive_goal'].get('click') and self._manual_goal_panel is not None
                        and self._manual_goal_panel.last_goal is not None):
                    x,y,yaw=self._manual_goal_panel.last_goal
                    self.goal_xy[0]=self.goal_xy.new_tensor([x,y])
                    self.goal_yaw[0]=yaw
                    distance=(self.goal_xy[ids]-pos[:,:2]).norm(dim=-1)
                self._manual_goal_basis_heading=heading[0].clone()
                self._manual_initial_goal_xy=self.goal_xy[0].clone()
                self._manual_initial_goal_yaw=self.goal_yaw[0].clone()
                self._manual_goal_commands.clear()
                if c['interactive_goal'].get('click') and self._manual_goal_panel is not None:
                    self._manual_goal_panel.cancel()
            self.previous_distance[ids] = distance
            delta = self.goal_xy[ids]-pos[:, :2]
            direction = delta/delta.norm(dim=-1).clamp_min(1e-6)[:, None]
            self.previous_goal_potential[ids] = travel_potential(distance, direction, heading, c)
            if c.get("sequential_arrival"):
                self.arrival_align[ids] = False
                self.arrival_turn_sign[ids] = 0
            self.hold_time[ids] = 0
            self.ever_held[ids] = False
            self.reset_standing[ids] = self._last_reset_standing
            self._gait_air_steps[ids]=0
            self._gait_last_length[ids]=0
            self._gait_known[ids]=False

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        cid = self._get_char_id()
        pos=self._engine.get_root_pos(cid)
        heading=torch_util.calc_heading(self._engine.get_root_rot(cid))
        target_yaw=self.goal_yaw
        if self.goal_config.get('sequential_arrival'):
            distance=(self.goal_xy-pos[:,:2]).norm(dim=-1)
            speed=self._engine.get_root_vel(cid)[:,:2].norm(dim=-1)
            previous_align=self.arrival_align.clone()
            self.arrival_align[:]=arrival_alignment_state(self.arrival_align,distance,speed,self.goal_config['sequential_arrival'])
            delta=self.goal_xy-pos[:,:2]
            travel_yaw=torch.atan2(delta[:,1],delta[:,0])
            request=alignment_heading_request(heading,self.goal_yaw,self.goal_config["sequential_arrival"].get("max_heading_request_deg"))
            route=self.goal_config['sequential_arrival'].get('turn_route')
            if route:
                error=torch.atan2(torch.sin(self.goal_yaw-heading),torch.cos(self.goal_yaw-heading))
                self.arrival_turn_sign[:]=committed_turn_sign(self.arrival_turn_sign,error,self.arrival_align,self.arrival_align&~previous_align,route)
                limit=math.radians(self.goal_config['sequential_arrival']['max_heading_request_deg'])
                request=heading+routed_turn_error(error,self.arrival_turn_sign).clamp(-limit,limit)
            target_yaw=torch.where(self.arrival_align,request,travel_yaw)
        goal = goal_observation(pos,heading,self.goal_xy,target_yaw,
                                self.goal_config.get('max_observation_distance'))
        if env_ids is not None:
            goal = goal[env_ids]
        return torch.cat((obs, goal), -1)

    def measured(self):
        cid = self._get_char_id()
        velocity = self._engine.get_root_vel(cid)
        angular = self._engine.get_root_ang_vel(cid)[:, 2]
        metrics = goal_metrics(self._engine.get_root_pos(cid),
                              torch_util.calc_heading(self._engine.get_root_rot(cid)),
                              velocity, angular, self.goal_xy, self.goal_yaw, self.goal_config)
        return metrics, velocity, angular

    def _update_reward(self):
        (distance, direction, yaw, speed, valid), velocity, angular = self.measured()
        dt = self._engine.get_timestep()
        reward_yaw=yaw
        if self.goal_config.get("sequential_arrival",{}).get("turn_route"):
            reward_yaw=routed_turn_error(yaw,self.arrival_turn_sign)
        self._reward_buf[:], self.reward_terms = goal_reward(
            distance, direction, reward_yaw, velocity, angular, self.previous_distance, dt, self.goal_config,
            torch_util.calc_heading(self._engine.get_root_rot(self._get_char_id())),
            self.previous_goal_potential,
            self.arrival_align if self.goal_config.get('sequential_arrival') else None)
        quality = self.goal_config.get('quality')
        if quality:
            metrics = self.motion_quality()
            self._reward_buf[:] += (quality['upright_weight']*metrics['upright_reward']
                                    +quality['upper_pose_weight']*metrics['upper_pose_reward']
                                    +quality.get('motion_weight', 0)*metrics.get('motion_reward', 0)
                                    -quality['slip_weight']*metrics['contact_slip_m2ps2'])
            self.reward_terms.update(metrics)
            if quality.get('touchdown_balance_weight',0):
                cid=self._get_char_id();contact=self._engine.get_contact_forces(cid)[:,self._quality_foot_ids,2]>10
                feet=self._engine.get_body_pos(cid)[:,self._quality_foot_ids,:2]
                heading=torch_util.calc_heading(self._engine.get_root_rot(cid))
                forward=torch.stack((heading.cos(),heading.sin()),-1)
                lead=((feet-feet[:,[1,0]])*forward[:,None]).sum(-1)
                aligned=(direction*forward).sum(-1)>.98
                eligible=(distance>1.)&(speed>.3)&(angular.abs()<.35)&aligned
                event=contact&(self._gait_air_steps>=3)&eligible[:,None]
                bonus=balanced_touchdown_reward(lead,self._gait_last_length[:,[1,0]],self._gait_known[:,[1,0]])
                reward=(bonus*event).sum(-1)
                self._reward_buf[:]+=quality['touchdown_balance_weight']*reward
                self.reward_terms['step_balance']=reward
                self._gait_last_length=torch.where(event,lead,self._gait_last_length)
                self._gait_known|=event
                self._gait_air_steps=torch.where(contact,0,self._gait_air_steps+1)
        self.previous_distance.copy_(distance)
        self.previous_goal_potential.copy_(travel_potential(distance, direction,
            torch_util.calc_heading(self._engine.get_root_rot(self._get_char_id())), self.goal_config))
        self.hold_time[:] = torch.where(valid, self.hold_time+dt, 0)
        self.ever_held |= self.hold_time >= self.goal_config['hold_seconds']
        # Keep running after arrival; success is assessed externally including post-hold.

    def motion_quality(self):
        if not hasattr(self, '_quality_foot_ids'):
            self._quality_foot_ids = self._build_body_ids_tensor(['right_foot', 'left_foot'])
            self._quality_torso_id = self._engine.find_obj_body_id(self._get_char_id(), 'torso')
            names = ['torso', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm']
            self._quality_joint_ids = torch.tensor([self._kin_char_model.get_joint_id(n) for n in names], device=self._device)
            ids = torch.arange(self._motion_lib.get_num_motions(), device=self._device).repeat_interleave(48)
            # Fixed reference-only pool, never validation rollouts. No leg phase restriction.
            phase = torch.linspace(0., .999, 48, device=self._device).repeat(self._motion_lib.get_num_motions())
            times = phase*self._motion_lib.get_motion_length(ids)
            frames = self._motion_lib.calc_motion_frame(ids, times)
            self._quality_full_refs = frames[4]
            self._quality_ref_velocity = frames[5]
            self._quality_ref_yaw_rate = frames[3][:, 2]
            self._quality_ref_speed = frames[2][:, :2].norm(dim=-1)
            self._quality_velocity_scale = frames[5].std(0).clamp_min(1.)
            self._quality_upper_refs = frames[4][:, self._quality_joint_ids]
        cid = self._get_char_id()
        force = self._engine.get_contact_forces(cid)[:, self._quality_foot_ids]
        vel = self._engine.get_body_vel(cid)[:, self._quality_foot_ids]
        torso_rot = self._engine.get_body_rot(cid)[:, self._quality_torso_id]
        up = torch_util.quat_rotate(torso_rot, torch.tensor([0., 0., 1.], device=self._device).expand(len(torso_rot), -1))
        tilt = torch.acos(up[:, 2].clamp(-1., 1.))
        rotations = self._kin_char_model.dof_to_rot(self._engine.get_dof_pos(cid))
        upper = upper_pose_error(rotations[:, self._quality_joint_ids], self._quality_upper_refs)
        out = dict(contact_slip_m2ps2=contact_slip(vel, force), torso_tilt_rad=tilt,
                    upper_pose_error_rad2=upper, upright_reward=torch.exp(-4*tilt.square()),
                    upper_pose_reward=torch.exp(-2*upper))
        if self.goal_config.get('quality', {}).get('motion_weight', 0):
            distance = (self.goal_xy-self._engine.get_root_pos(cid)[:, :2]).norm(dim=-1)
            speed = torch.minimum(torch.full_like(distance, self.goal_config['walking_speed']),
                                  self.goal_config['deceleration_gain']*(distance-self.goal_config.get('stop_deadband', 0)).clamp_min(0))
            require_active=None
            if self.goal_config.get('steering'):
                delta=self.goal_xy-self._engine.get_root_pos(cid)[:,:2]
                heading=torch_util.calc_heading(self._engine.get_root_rot(cid))
                travel=torch.atan2(delta[:,1],delta[:,0])-heading
                speed=steering_forward_speed(distance, travel, speed, self.goal_config['steering'])
                # Slow turn-in-place can use transition examples rather than a frozen stand only.
                speed=torch.maximum(speed,.2*((travel.cos()<.95)&(distance>.6)).float())
                if self.goal_config['quality'].get('active_turn_reference',False):
                    angle=torch.atan2(travel.sin(),travel.cos()).abs()
                    require_active=(distance>1.)&(angle>math.radians(45))
            if self.goal_config['quality'].get('active_approach_reference',False):
                approach=(distance>self.goal_config['position_tolerance'])&(distance<=1.)
                require_active=approach if require_active is None else require_active|approach
            if self.goal_config.get('sequential_arrival'):
                final_error=torch.atan2(torch.sin(self.goal_yaw-heading),torch.cos(self.goal_yaw-heading)).abs()
                turning=self.arrival_align & (final_error>math.radians(self.goal_config['yaw_tolerance_deg']))
                require_active=turning if require_active is None else require_active|turning
                speed=torch.where(self.arrival_align,torch.where(turning,torch.full_like(speed,self.goal_config["quality"].get("alignment_reference_speed",.2)),torch.zeros_like(speed)),speed)
            margin = self.goal_config['quality'].get('actual_speed_margin')
            if margin is not None:
                actual = self._engine.get_root_vel(cid)[:, :2].norm(dim=-1)
                # Keep a moving reference while momentum remains; task velocity still requests braking.
                speed = torch.maximum(speed, (actual-margin).clamp_min(0)).clamp(max=self.goal_config['walking_speed'])
            velocity_weight=self.goal_config['quality'].get('motion_velocity_weight',.1)
            if self.goal_config.get('sequential_arrival') and 'alignment_motion_velocity_weight' in self.goal_config['quality']:
                velocity_weight=torch.where(self.arrival_align,torch.full_like(speed,self.goal_config['quality']['alignment_motion_velocity_weight']),torch.full_like(speed,velocity_weight))
            requested_turn=None
            if self.goal_config['quality'].get('directional_alignment_reference',False):
                if not self.goal_config.get('sequential_arrival'):
                    raise ValueError('Directional alignment prior requires sequential arrival')
                heading=torch_util.calc_heading(self._engine.get_root_rot(cid))
                yaw=torch.atan2(torch.sin(self.goal_yaw-heading),torch.cos(self.goal_yaw-heading))
                requested_turn=torch.where(turning,yaw,torch.zeros_like(yaw))
            out['motion_reward'] = reference_motion_reward(rotations, self._engine.get_dof_vel(cid),
                self._quality_full_refs, self._quality_ref_velocity, self._quality_ref_speed, speed,
                self._quality_velocity_scale, velocity_weight,
                self.goal_config['quality'].get('moving_target_speed', .4),
                self.goal_config['quality'].get('stationary_target_speed', .1), require_active,
                reference_yaw_rate=self._quality_ref_yaw_rate, requested_turn=requested_turn,
                minimum_turn_rate=self.goal_config['quality'].get('minimum_reference_turn_rate',.1))
        return out

    def record_diagnostics(self):
        out = super().record_diagnostics()
        if self.goal_config.get('measured_standing_reset'):
            out['goal_measured_reset_fraction'] = torch.tensor(
                getattr(self, '_measured_reset_count', 0)/max(1, getattr(self, '_measured_reset_total', 0)),
                device=self._device)
        (distance, direction, yaw, speed, valid), _, angular = self.measured()
        out.update(goal_distance=distance.mean(), goal_yaw_error=yaw.abs().mean(),
                   goal_speed=speed.mean(), goal_valid=valid.float().mean(),
                   goal_held=self.ever_held.float().mean())
        short = {'contact_slip_m2ps2':'slip', 'torso_tilt_rad':'tilt',
                 'upper_pose_error_rad2':'upper_err', 'upright_reward':'upright',
                 'upper_pose_reward':'upper_pose'}
        short['motion_reward'] = 'motion'
        short['travel_heading'] = 'heading'
        short['arrival_velocity'] = 'arrival_vel'
        # Native text logger has fixed-width columns; long names concatenate.
        out.update({'goal_reward_'+short.get(k, k): v.mean() for k, v in self.reward_terms.items()})
        return out

    def _render_scene(self):
        super()._render_scene()
        if self._manual_goal_panel is not None:
            self._manual_goal_panel.update(self)
        from humanoid_imitation.planning.goal_overlay import goal_lines
        cid = self._get_char_id()
        p = self._engine.get_root_pos(cid)[0].detach().cpu().numpy()
        yaw = float(torch_util.calc_heading(self._engine.get_root_rot(cid))[0])
        lines = goal_lines(p, yaw, {'xy': self.goal_xy[0].detach().cpu().numpy(), 'yaw': float(self.goal_yaw[0])}, self.goal_config['position_tolerance'],
                           math.radians(self.goal_config['yaw_tolerance_deg']))
        self._engine.draw_lines(0, *lines, 3.)
        if self.goal_config.get('interactive_goal',{}).get('click'):
            self._manual_goal_panel.draw_pending(self._engine)

    def _render(self):
        if not self.goal_config.get('interactive_goal',{}).get('click'):
            return super()._render()
        from envs.sim_env import PlayMode
        app=self._engine._app_launcher.app
        while app.is_running():
            self._update_camera()
            self._render_scene()
            if self._play_mode != PlayMode.PAUSE:
                break
        if self._play_mode == PlayMode.ONE_STEP:
            self._play_mode=PlayMode.PAUSE
