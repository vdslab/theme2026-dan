"""Render a reproducible before/after figure (optional matplotlib dependency)."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from solve_polygon_layout import PolygonProblem, assert_preserved, face_metrics


def plot_comparison(before_path, after_path, output, largest=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before = json.loads(Path(before_path).read_text(encoding='utf-8'))
    after = json.loads(Path(after_path).read_text(encoding='utf-8'))
    assert_preserved(before, after)
    problem = PolygonProblem(before)
    problems = [problem, PolygonProblem(after)]
    arrays = [p.initial for p in problems]
    selected = sorted(problem.faces, key=lambda e: len(problem.members[e]), reverse=True)[:largest]
    metrics = [face_metrics(p, x, reorder=False) for p, x in zip(problems, arrays)]
    low, high = np.min(np.vstack(arrays), axis=0), np.max(np.vstack(arrays), axis=0)
    padding = np.maximum(high-low, 1e-3)*.05
    fig, axes = plt.subplots(len(selected)+1, 2, figsize=(11, 3*(len(selected)+1)), squeeze=False)
    for col, xy in enumerate(arrays):
        ax = axes[0, col]
        for edge in problem.faces:
            ids = problems[col].members[edge]
            points = xy[ids]
            color = problem.data['hyperedges'][edge].get('color', {'r': 80, 'g': 130, 'b': 200})
            rgb = np.clip(np.asarray([color[k] for k in ('r', 'g', 'b')])/255, 0, 1)
            ax.fill(points[:, 0], points[:, 1], color=rgb, alpha=.12, linewidth=.2, edgecolor=rgb)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c='#253648')
        ax.set_title(('Before' if col == 0 else 'After')+' — full graph')
        ax.set_xlim(low[0]-padding[0], high[0]+padding[0])
        ax.set_ylim(low[1]-padding[1], high[1]+padding[1])
        ax.set_aspect('equal'); ax.axis('off')
        for row, edge in enumerate(selected, 1):
            ids = problems[col].members[edge]
            points = xy[ids]
            ax = axes[row, col]
            ax.fill(points[:, 0], points[:, 1], color='#78adc7', alpha=.4)
            closed = np.vstack((points, points[0]))
            ax.plot(closed[:, 0], closed[:, 1], color='#347c9e', linewidth=.65)
            ax.scatter(points[:, 0], points[:, 1], s=5, c='#254859')
            label = str(problem.data['hyperedges'][edge].get('label', edge))
            ratio = metrics[col][row-1]['regularity_ratio']
            ax.set_title(f'{label} | n={len(ids)} | ratio={ratio:.2f}', fontsize=10)
            ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle('All members retained. Overview: common scale. Individual faces: fitted scale.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .99))
    fig.savefig(output, dpi=140, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('output')
    parser.add_argument('--largest', type=int, default=8)
    args = parser.parse_args()
    if args.largest < 1:
        parser.error('--largest must be positive')
    plot_comparison(args.before, args.after, args.output, args.largest)
