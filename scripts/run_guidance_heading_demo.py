"""Heading-only demo extension of the frozen V2 runner.

Scene JSON may specify final_yaw_relative_initial_deg. Keeps original runner and
published protocol untouched; checks each source edit before executing it.
"""
from pathlib import Path
import hashlib

BASE = Path(__file__).with_name('run_guidance_comparison_v2.py')

def main():
    source = BASE.read_text()
    edits = [
        ('a.root=ROOT/', 'a.manifest=a.manifest.resolve();a.root=ROOT/'),
        ('heading=yaw(saved[\'root_rot\'])', "heading=yaw(saved['root_rot'])\n    final_heading=heading+torch.tensor([math.radians(v.get('final_yaw_relative_initial_deg',0)) for v in specs])"),
        ('final_yaw_relative_initial_deg=0,', "final_yaw_relative_initial_deg=specs[i].get('final_yaw_relative_initial_deg',0),"),
        ('env.goal_yaw[:]=heading;', 'env.goal_yaw[:]=final_heading.cuda();'),
        ('angle=yaws[i]-heading[i];', 'angle=yaws[i]-final_heading[i].cuda();'),
        ('yaw=float(heading[i])', 'yaw=float(final_heading[i])'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise RuntimeError('Frozen runner layout changed: '+before)
        source = source.replace(before, after)
    print('Heading demo base SHA256:', hashlib.sha256(BASE.read_bytes()).hexdigest(), flush=True)
    scope = {'__file__': __file__, '__name__': '__main__'}
    exec(compile(source, str(BASE), 'exec'), scope)

if __name__ == '__main__':
    main()
