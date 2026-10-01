#!/usr/bin/env python3
"""Plot aligned source/target skeleton samples without launching a simulator."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def preview(path):
    data = np.load(path)
    samples = np.linspace(0, len(data['target_positions'])-1, 6).astype(int)
    fig = plt.figure(figsize=(15, 7))
    for row, kind in enumerate(('source', 'target')):
        positions = data[kind+'_positions']
        parents = data[kind+'_parents']
        names = data[kind+'_names']
        for col, t in enumerate(samples):
            ax = fig.add_subplot(2, 6, row*6+col+1, projection='3d')
            p = positions[t].copy()
            p[:, :2] -= p[0, :2]
            for j, parent in enumerate(parents):
                if parent < 0:
                    continue
                # Exclude Mixamo hair/finger/accessory chains from the audit view.
                if kind == 'source' and str(names[j]) not in {
                    'Spine','Spine1','Spine2','Neck','Head','LeftShoulder','RightShoulder',
                    'LeftArm','RightArm','LeftForeArm','RightForeArm','LeftHand','RightHand',
                    'LeftUpLeg','RightUpLeg','LeftLeg','RightLeg','LeftFoot','RightFoot',
                    'LeftToeBase','RightToeBase','LeftToe_End','RightToe_End'}:
                    continue
                color = 'tab:red' if 'right' in str(names[j]).lower() else 'tab:blue'
                ax.plot(*p[[parent, j]].T, color=color, lw=2)
            ax.set(xlim=(-.65,.65), ylim=(-.65,.65), zlim=(0,2), title=f'{kind} frame {t}')
            ax.set_box_aspect((1.3,1.3,2))
            ax.view_init(elev=12, azim=-65)
            ax.set_xlabel('forward X'); ax.set_ylabel('left Y')
    fig.tight_layout()
    dest = path.with_suffix('.png')
    fig.savefig(dest, dpi=120)
    print(dest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('qa_npz', type=Path)
    preview(p.parse_args().qa_npz)
