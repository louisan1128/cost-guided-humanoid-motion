"""Two ground clicks produce one world target. Viewer imports stay lazy."""
import math


class TwoClickGoal:
    def __init__(self):
        self.pending=None

    def click(self,position):
        if len(position)<3 or not all(math.isfinite(float(v)) for v in position) or abs(position[2])>.15:
            return None
        xy=(float(position[0]),float(position[1]))
        if self.pending is None:
            self.pending=xy
            return None
        dx,dy=xy[0]-self.pending[0],xy[1]-self.pending[1]
        if math.hypot(dx,dy)<.1:
            return None
        goal=(*self.pending,math.atan2(dy,dx))
        self.pending=None
        return goal


def local_ground_position(position, origin_xy):
    """World floor click -> per-lane coordinates, retaining required ground Z."""
    if len(position) != 3 or len(origin_xy) != 2:
        raise ValueError('Expected a 3D floor point and 2D lane origin')
    return (float(position[0])-float(origin_xy[0]),
            float(position[1])-float(origin_xy[1]),float(position[2]))


def screen_ndc(pixel, rect, dpi=1.):
    """Carb physical pixels -> UI viewport rectangle -> camera NDC."""
    x,y=float(pixel[0])/dpi,float(pixel[1])/dpi
    left,top,width,height=rect
    if width<=0 or height<=0 or not (left<=x<left+width and top<=y<top+height):
        return None
    return (2*(x-left)/width-1, 1-2*(y-top)/height)


def ground_from_ndc(ndc, inverse):
    """Intersect camera ray with the project's flat z=0 floor (Gf row vectors)."""
    import numpy as np
    matrix=np.asarray(inverse,dtype=float)
    points=[]
    for depth in (-1.,0.):
        point=np.array([*ndc,depth,1.]) @ matrix
        if not np.isfinite(point).all() or abs(point[3])<1e-12:
            raise ValueError('Camera projection unavailable')
        points.append(point[:3]/point[3])
    origin=points[0];direction=points[1]-origin
    if abs(direction[2])<1e-10:
        raise ValueError('Click lower: ray is parallel to the floor')
    t=-origin[2]/direction[2]
    if t<0:
        raise ValueError('Click the floor, below the horizon')
    hit=origin+t*direction
    if not np.isfinite(hit).all():raise ValueError('Invalid floor position')
    return (float(hit[0]),float(hit[1]),0.)


class ClickGoalPanel:
    def __init__(self,enqueue):
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window
        window=get_active_viewport_window()
        if window is None:raise RuntimeError('No active viewport for goal clicks')
        self.viewport=window.viewport_api
        self.ui=ui
        self.selector=TwoClickGoal();self.enqueue=enqueue;self.last_goal=None
        self.pressed=None;self.clicks=0
        # Capture on the viewport widget itself: raw device subscriptions can be
        # consumed by Kit before reaching application subscribers.
        self.frame=window.get_frame('humanoid_click_goal')
        with self.frame:
            self.click_area=ui.Rectangle(
                style={'background_color':0x01000000},
                opaque_for_mouse_events=True,
                mouse_pressed_fn=self.on_press,
                mouse_released_fn=self.on_release)
        self.capture_enabled=True
        self.window=ui.Window('Click goal',width=430,height=215)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.prompt=ui.Label('1: left-click floor for target position',height=24)
                self.input_status=ui.Label('Viewport clicks: 0; click the floor',height=24)
                self.state=ui.Label('',height=24)
                self.stage=ui.Label('',height=20)
                ui.Label('2: click floor to choose facing direction',height=24)
                ui.Label('Viewport click capture v3; close simulator or Ctrl+C to exit.',height=24)
                self.mode_button=ui.Button('Switch to camera controls',height=26,clicked_fn=self.toggle_capture)
    @staticmethod
    def bounds(frame):
        return (frame.screen_position_x,frame.screen_position_y,frame.computed_width,frame.computed_height)

    def toggle_capture(self):
        self.capture_enabled=not self.capture_enabled
        self.click_area.visible=self.capture_enabled
        self.pressed=None
        self.mode_button.text=('Switch to camera controls' if self.capture_enabled else 'Enable goal clicks')
        self.input_status.text=('Goal clicks enabled' if self.capture_enabled else 'Camera controls enabled; goal clicks off')

    def on_press(self,x,y,button,modifier):
        # omni.ui callback coordinates are logical screen coordinates, unlike Carb.
        if button==0 and not modifier and self.capture_enabled:
            self.pressed=(float(x),float(y))

    def on_release(self,x,y,button,modifier):
        if button!=0 or not self.capture_enabled:return
        self.clicks+=1
        self.input_status.text=f'Viewport clicks: {self.clicks}'
        pressed=self.pressed;self.pressed=None
        if pressed is None or modifier:
            self.input_status.text+='; plain left-click required'
            return
        if math.hypot(x-pressed[0],y-pressed[1])>6:
            self.input_status.text+='; drag ignored'
            return
        try:
            ndc=screen_ndc((x,y),self.bounds(self.click_area))
            if ndc is None:
                self.input_status.text+='; outside viewport'
                return
            position=ground_from_ndc(ndc,self.viewport.ndc_to_world)
            self.accept_position(position)
            self.input_status.text+=f'; floor ({position[0]:.2f}, {position[1]:.2f})'
        except Exception as error:
            self.input_status.text+=f'; {error}'

    def accept_position(self,position):
        goal=self.selector.click(position)
        if goal is not None:
            self.last_goal=goal
            self.enqueue(('set_goal',*goal))
        self.prompt.text=('2: click floor for facing direction' if self.selector.pending else '1: left-click floor for next target')

    def cancel(self):
        self.pressed=None;self.selector.pending=None
        self.prompt.text='1: left-click floor for target position'

    def update(self,env):
        distance,_,yaw,speed,_=env.measured()[0]
        self.state.text=f'Error: {float(distance[0]):.2f} m / {math.degrees(float(yaw[0])):.1f} deg; speed {float(speed[0]):.2f} m/s'
        if getattr(env,'goal_config',{}).get('sequential_arrival'):
            stage='ALIGN/HOLD' if bool(env.arrival_align[0]) else 'MOVE/BRAKE'
            route=env.goal_config['sequential_arrival'].get('turn_route','shortest')
            self.stage.text=f'Stage: {stage}; final turn route: {route}'

    def draw_pending(self,engine):
        if self.selector.pending is None:return
        import numpy as np
        x,y=self.selector.pending
        starts=np.array([[x-.15,y,.07],[x,y-.15,.07]],dtype=np.float32)
        ends=np.array([[x+.15,y,.07],[x,y+.15,.07]],dtype=np.float32)
        engine.draw_lines(0,starts,ends,np.array([[.1,1.,.3,1.]]*2,dtype=np.float32),4.)

    def destroy(self):
        self.window.visible=False
        self.frame.clear()


def run_click_inference(agent,test_episodes=None):
    """No episode-count or time exit; physical falls still reset the robot."""
    import torch
    from learning.base_agent import AgentMode
    env=agent._env;app=env._engine._app_launcher.app
    agent.eval();agent.set_mode(AgentMode.TEST)
    try:
        with torch.no_grad():
            agent._curr_obs,agent._curr_info=agent._reset_envs()
            while app.is_running():
                action,_=agent._decide_action(agent._curr_obs,agent._curr_info)
                _,_,done,_=agent._step_env(action)
                if not app.is_running():break
                agent._curr_obs,agent._curr_info=agent._reset_done_envs(done)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env._manual_goal_panel.destroy()
        finally:
            app.close()
