"""Uniform, exhaustive polygon judgments; independent of the search objective.

Quality thresholds are repository review settings, not constants from a paper.
Passing a non-regression check does not mean that every polygon is regular.
"""
from collections import Counter
import html
from pathlib import Path

import numpy as np


DEFAULT_LIMITS = {'regularity_ratio': 1.20, 'hull_deficit': .02, 'side_length_cv': .25}


def judge_face_metrics(before, after, relative_tolerance=0., limits=None):
    if not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance <= 1:
        raise ValueError('Relative shape tolerance must be between zero and one')
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or not np.isfinite(list(limits.values())).all():
        raise ValueError('Expected three finite polygon quality limits')
    if limits['regularity_ratio'] < 1 or min(limits['hull_deficit'], limits['side_length_cv']) < 0:
        raise ValueError('Invalid polygon quality limits')
    old = {f['id']: f for f in before}
    new = {f['id']: f for f in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new):
        raise ValueError('Missing, extra or duplicate faces in all-face judgment')
    rows = []
    for candidate in after:
        baseline = old[candidate['id']]
        if baseline['vertices'] != candidate['vertices']:
            raise ValueError('Face cardinality changed during judgment')
        baseline_valid = baseline['is_valid'] and not baseline['is_degenerate']
        candidate_valid = candidate['is_valid'] and not candidate['is_degenerate']
        failures = []
        if not candidate['is_valid']:
            failures.append('invalid_boundary')
        if candidate['is_degenerate']:
            failures.append('degenerate_area')
        failures += [name for name, limit in limits.items() if candidate[name] > limit+1e-8]
        delta = candidate['regularity_ratio']-baseline['regularity_ratio']
        epsilon = 1e-8*max(1., baseline['regularity_ratio'])
        if not candidate_valid:
            change = 'invalid'
        elif not baseline_valid:
            change = 'repaired'
        else:
            change = 'worsened' if delta > epsilon else 'improved' if delta < -epsilon else 'unchanged'
        # A broken input face does not exempt any OTHER valid input face.
        regression = (not candidate_valid or (baseline_valid and
            candidate['regularity_ratio'] > baseline['regularity_ratio']*(1+relative_tolerance)+epsilon))
        rows.append({'id': candidate['id'], 'label': candidate['label'], 'vertices': candidate['vertices'],
                     'before': baseline, 'after': candidate, 'change': change,
                     'regularity_delta': delta, 'regression': bool(regression),
                     'quality_pass': not failures, 'quality_failures': failures})
    counts = Counter(row['change'] for row in rows)
    regressed = [row['id'] for row in rows if row['regression']]
    failed = [row['id'] for row in rows if not row['quality_pass']]
    failure_counts = Counter(reason for row in rows for reason in row['quality_failures'])
    ratios = [row['after']['regularity_ratio'] for row in rows]
    return {'checked_faces': len(rows), 'quality_pass_count': len(rows)-len(failed),
            'quality_fail_count': len(failed), 'all_faces_pass_quality': not failed,
            'quality_failure_counts': dict(failure_counts),
            'quality_status': 'not_applicable' if not rows else 'passed' if not failed else 'incomplete',
            'all_faces_non_regressing': not regressed, 'regression_count': len(regressed),
            'regressed_face_ids': regressed, 'failed_quality_face_ids': failed,
            'changes': {name: counts[name] for name in ('improved', 'unchanged', 'worsened', 'repaired', 'invalid')},
            'worst_regularity_ratio': float(max(ratios)) if ratios else None,
            'p95_regularity_ratio': float(np.percentile(ratios, 95)) if ratios else None,
            'relative_regularity_tolerance': relative_tolerance, 'limits': limits,
            'limit_origin': 'Repository review policy; not paper constants',
            'replacement_policy': 'Every polygon must be valid and within its own regularity regression limit',
            'quality_policy': 'Every polygon must meet all three absolute review limits', 'faces': rows}


def all_face_judgment(before_problem, after_problem, relative_tolerance=0., limits=None):
    from solve_polygon_layout import face_metrics
    if set(before_problem.ids) != set(after_problem.ids):
        raise ValueError('Node identities changed in all-face judgment')
    old_edges = {e['id']: e for e in before_problem.data['hyperedges']}
    new_edges = {e['id']: e for e in after_problem.data['hyperedges']}
    if set(old_edges) != set(new_edges):
        raise ValueError('Hyperedge identities changed in all-face judgment')
    for eid, edge in old_edges.items():
        if Counter(edge['nodes']) != Counter(new_edges[eid]['nodes']):
            raise ValueError('Hyperedge incidence changed in all-face judgment')
    judgment = judge_face_metrics(face_metrics(before_problem, before_problem.initial, reorder=False),
                                  face_metrics(after_problem, after_problem.initial, reorder=False),
                                  relative_tolerance, limits)
    excluded = []
    for edge, members in zip(after_problem.data['hyperedges'], after_problem.members):
        n = len(members)
        if n < 3:
            excluded.append({'id': edge['id'], 'label': edge.get('label', ''), 'vertices': n,
                             'kind': ('empty', 'unary', 'binary')[n], 'status': 'not_a_polygon_area',
                             'reason': 'No area polygon; glyph/line quality is not evaluated here'})
    judgment.update(total_hyperedges=len(new_edges), non_polygon_hyperedges=excluded,
                    all_hyperedges_accounted_for=len(new_edges) == len(excluded)+judgment['checked_faces'])
    return judgment


def write_quality_review(path, before_problem, after_problem, judgment):
    """Self-contained, searchable HTML showing EVERY judged face and all IDs."""
    def esc(value):
        return html.escape(str(value), quote=True)

    def polygon_svg(problem, edge):
        points = problem.initial[problem.members[edge]]
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high-low, 1e-12)
        scale = min(250/span[0], 150/span[1])
        display = (points-(low+high)/2)*[scale, -scale]+[140, 90]
        vertices = ' '.join(f'{x:.5f},{y:.5f}' for x, y in display)
        dots = ''.join(f'<circle cx="{x:.5f}" cy="{y:.5f}" r="2"/>' for x, y in display)
        return f'<svg viewBox="0 0 280 180" role="img" aria-label="polygon boundary"><polygon points="{vertices}"/>{dots}</svg>'

    before_index = {e['id']: i for i, e in enumerate(before_problem.data['hyperedges'])}
    after_index = {e['id']: i for i, e in enumerate(after_problem.data['hyperedges'])}
    parts = ['<!doctype html><html lang="ja"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1"><title>全ポリゴン判定</title>',
             '<style>body{font:15px system-ui;margin:24px;background:#f4f6f8;color:#192b3b}h1{font-size:24px}',
             'header{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}input,select{padding:8px;margin:4px}',
             'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}',
             '.card{background:white;border:1px solid #ccd7df;border-radius:8px;padding:14px}.card.failed{border-left:5px solid #bb6741}',
             '.card.regressed{border-left:5px solid #b51d38}.card[hidden]{display:none}.pair{display:flex}.panel{width:50%;text-align:center}',
             'svg{width:100%}polygon{fill:#89bace66;stroke:#356b86;stroke-width:1}circle{fill:#192b3b}',
             'table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dde4ea;text-align:left;padding:4px}',
             '.bad{color:#a42d21}small{color:#526372}h2{font-size:16px;margin:0 0 8px}</style>',
             '<header><h1>全ポリゴン判定</h1>',
             f'<p>全ハイパーエッジ {judgment["total_hyperedges"]}件／面として判定 {judgment["checked_faces"]}件。',
             f'品質基準達成 {judgment["quality_pass_count"]}件、未達 {judgment["quality_fail_count"]}件、',
             f'許容範囲を超えた悪化 {judgment["regression_count"]}件。</p>',
             '<p>品質基準の達成と、入力より悪化しないことは別の判定です。基準は本実装の確認用設定で、論文の定数ではありません。</p>',
             f'<p>基準：{esc(judgment["limits"])}。形状比は1が理想。全メンバーを表示。各パネルは独立した表示倍率。</p>',
             '<input id="query" placeholder="名前・IDで検索" aria-label="名前・IDで検索">',
             '<select id="filter" aria-label="判定で絞り込み"><option value="all">全件</option><option value="failed">品質未達</option>',
             '<option value="regressed">悪化して不採用</option><option value="passed">品質達成</option></select>',
             '<span id="visible"></span></header><main>']
    for row in judgment['faces']:
        classes = 'card'+(' failed' if not row['quality_pass'] else '')+(' regressed' if row['regression'] else '')
        parts.append(f'<article class="{classes}" data-pass="{str(row["quality_pass"]).lower()}" data-regressed="{str(row["regression"]).lower()}">')
        parts.append(f'<h2>{esc(row["label"])} — ID {esc(row["id"])} / {row["vertices"]} vertices</h2>')
        parts.append(f'<p class="{esc("bad" if row["quality_failures"] else "")}">品質：{esc("達成" if row["quality_pass"] else "未達")} / 変化：{esc(row["change"])} / 未達項目：{esc(", ".join(row["quality_failures"]) or "なし")}</p>')
        parts.append('<div class="pair">')
        for title, problem, index in (('参照', before_problem, before_index), ('評価対象', after_problem, after_index)):
            parts.append(f'<div class="panel">{title}{polygon_svg(problem, index[row["id"]])}</div>')
        parts.append('</div><table><tr><th>指標</th><th>参照</th><th>評価対象</th></tr>')
        for key in ('regularity_ratio', 'hull_deficit', 'side_length_cv'):
            parts.append(f'<tr><td>{key}</td><td>{row["before"][key]:.6g}</td><td>{row["after"][key]:.6g}</td></tr>')
        parts.append('</table></article>')
    parts.append('</main><h2>面の品質判定の対象外（所属は保存）</h2><table><tr><th>ID</th><th>名前</th><th>種類</th></tr>')
    for edge in judgment['non_polygon_hyperedges']:
        parts.append(f'<tr><td>{esc(edge["id"])}</td><td>{esc(edge["label"])}</td><td>{esc(edge["kind"])}</td></tr>')
    parts += ['</table><script>',
              'const cards=[...document.querySelectorAll(".card")],query=document.querySelector("#query"),filter=document.querySelector("#filter");',
              'function update(){let n=0;const q=query.value.toLowerCase(),f=filter.value;for(const c of cards){const matches=c.textContent.toLowerCase().includes(q)&&(f==="all"||(f==="failed"&&c.dataset.pass==="false")||(f==="regressed"&&c.dataset.regressed==="true")||(f==="passed"&&c.dataset.pass==="true"));c.hidden=!matches;if(matches)n++;}document.querySelector("#visible").textContent=n+"件表示";}',
              'query.addEventListener("input",update);filter.addEventListener("change",update);update();</script></html>']
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(parts), encoding='utf-8', newline='\n')
