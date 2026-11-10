"""Display-only world goal and heading comparison for the Isaac Lab viewer."""
import math
import numpy as np


def heading_error(actual, desired):
    return math.atan2(math.sin(actual-desired), math.cos(actual-desired))


def goal_lines(position, yaw, goal, radius, yaw_tolerance):
    starts, ends, colors = [], [], []
    yellow = [1., .8, .05, 1.]
    cyan = [.05, .85, 1., 1.]

    def segment(a, b, color):
        starts.append(a); ends.append(b); colors.append(color)

    def arrow(origin, angle, color, length=.7):
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        side = np.array([-math.sin(angle), math.cos(angle), 0.])
        tip = origin+length*direction
        segment(origin, tip, color)
        for sign in (-1, 1):
            segment(tip, tip-.16*direction+sign*.08*side, color)

    target = np.array([*goal['xy'], .06])
    angles = np.linspace(0, 2*math.pi, 49)
    ring = target+radius*np.c_[np.cos(angles), np.sin(angles), np.zeros_like(angles)]
    for a, b in zip(ring[:-1], ring[1:]):
        segment(a, b, yellow)
    arrow(target, goal['yaw'], yellow)
    origin = np.asarray(position, dtype=float)+[0., 0., .25]
    arrow(origin, goal['yaw'], yellow)
    arrow(origin, yaw, cyan)
    for angle in (goal['yaw']-yaw_tolerance, goal['yaw']+yaw_tolerance):
        segment(origin, origin+.7*np.array([math.cos(angle), math.sin(angle), 0.]), [.7, .7, .7, .65])
    return tuple(np.asarray(x, dtype=np.float32) for x in (starts, ends, colors))


class GoalOverlay:
    def __init__(self, goal, config):
        import omni.ui as ui
        self.goal, self.config = goal, config
        self.window = ui.Window('Goal and heading', width=410, height=240)
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.Label('Yellow: desired heading / Cyan: actual heading', height=24)
                ui.Label('Ground ring: goal position / Grey: yaw tolerance', height=24)
                self.time_label = ui.Label('', height=24)
                self.goal_label = ui.Label('', height=24)
                self.yaw_label = ui.Label('', height=24)
                self.error_label = ui.Label('', height=24)
                self.position_label = ui.Label('', height=24)
                self.speed_label = ui.Label('', height=24)

    def update(self, env):
        from humanoid_imitation.planning.motion_adapter import heading, rotation_matrix
        cid = env._get_char_id()
        position = env._engine.get_root_pos(cid)[0].detach().cpu().numpy()
        quaternion = env._engine.get_root_rot(cid)[0].detach().cpu().numpy()
        velocity = env._engine.get_root_vel(cid)[0].detach().cpu().numpy()
        yaw = float(heading(rotation_matrix(quaternion)))
        error = math.degrees(heading_error(yaw, self.goal['yaw']))
        distance = float(np.linalg.norm(position[:2]-self.goal['xy']))
        limit = self.config['goal_yaw_tolerance_deg']
        self.time_label.text = f"Time: {float(env.get_env_time()[0]):.2f}s / deadline: {self.goal['deadline_s']:.2f}s"
        self.goal_label.text = f"Goal XY: ({self.goal['xy'][0]:.2f}, {self.goal['xy'][1]:.2f})"
        self.yaw_label.text = f"Desired yaw: {math.degrees(self.goal['yaw']):.1f}deg / actual: {math.degrees(yaw):.1f}deg"
        self.error_label.text = f"Yaw error: {error:+.2f}deg / limit: +/-{limit:g}deg  {'OK' if abs(error)<=limit else 'OUT'}"
        self.position_label.text = f"Position error: {distance:.3f}m / limit: {self.config['goal_position_tolerance_m']:g}m"
        self.speed_label.text = f"Speed: {np.linalg.norm(velocity[:2]):.3f}m/s / stop limit: {self.config['stop_speed_mps']:g}m/s"
        return goal_lines(position, yaw, self.goal, self.config['goal_position_tolerance_m'], math.radians(limit))

    def install(self, env):
        original = env._render_scene

        def render():
            lines = self.update(env)
            original()  # Isaac Lab clears previous debug lines during render.
            env._engine.draw_lines(0, *lines, 3.)

        env._render_scene = render
