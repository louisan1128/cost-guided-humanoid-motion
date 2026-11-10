"""Viewer controls for a manual teacher demo; no planning or policy logic."""
import math


class ManualGoalPanel:
    def __init__(self, enqueue, config):
        import omni.ui as ui
        self.window=ui.Window('Manual goal control',width=430,height=290)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Manual demo: target changes are not benchmark episodes',height=22)
                ui.Label(f"I/K forward/back; J/L left/right ({config['position_step']:g} m per press)",height=22)
                ui.Label(f"U/O rotate final heading (+/-{config['yaw_step_deg']:g} deg); H restore goal",height=22)
                with ui.HStack(height=30,spacing=4):
                    for label,action in [('Forward [I]','forward'),('Back [K]','back'),('Left [J]','left'),('Right [L]','right')]:
                        ui.Button(label,clicked_fn=lambda a=action:enqueue(a))
                with ui.HStack(height=30,spacing=4):
                    for label,action in [('Yaw + [U]','yaw_left'),('Yaw - [O]','yaw_right'),('Restore [H]','restore')]:
                        ui.Button(label,clicked_fn=lambda a=action:enqueue(a))
                ui.Label('Position moves in START heading axes; yellow ring/arrow = target',height=22)
                self.target_label=ui.Label('',height=22)
                self.state_label=ui.Label('',height=22)
                self.time_label=ui.Label('',height=22)

    def update(self,env):
        distance,_,yaw,speed,_=env.measured()[0]
        xy=env.goal_xy[0].detach().cpu().tolist()
        angle=math.degrees(float(env.goal_yaw[0]))
        self.target_label.text=f'World target: ({xy[0]:.2f}, {xy[1]:.2f}) m; yaw: {angle:.1f} deg'
        self.state_label.text=f'Error: {float(distance[0]):.2f} m / {math.degrees(float(yaw[0])):.1f} deg; speed: {float(speed[0]):.2f} m/s'
        self.time_label.text=f'Time: {float(env.get_env_time()[0]):.1f} / {env._episode_length:g} s; Enter: pause; Ctrl+C: quit'
