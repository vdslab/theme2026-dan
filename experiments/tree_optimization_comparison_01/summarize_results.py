"""Summarize verified layouts and render their fixed-view comparison."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


STUDY = "tree_optimization_comparison_01"
SCRIPT = f"experiments/{STUDY}"
DEFAULT_OUTPUT = Path("results/experiments") / STUDY
SEEDS = range(5)
METRICS = {
    "l2_regularized": "二乗目的値 F",
    "l2_scaled_qp": "正規化二乗目的値",
    "used_layer_count": "レイヤー数",
    "height_range": "高さ幅",
    "pillar_length": "柱の総長",
    "l1_regularized": "重み付き L1",
    "total_seconds": "総時間（秒）",
    "search_seconds": "探索時間（秒）",
    "attempts": "試行回数",
    "overlap_violation_count": "重なり違反",
    "optimized_value": "最適化した値",
}
MAIN_COLUMNS = (
    "l2_scaled_qp", "used_layer_count", "height_range", "pillar_length",
    "total_seconds", "search_seconds", "attempts", "overlap_violation_count",
)
TRADEOFFS = ("l2_scaled_qp", "used_layer_count", "height_range", "pillar_length")
CSV_CASES = (
    "stage", "condition", "case_key", "example", "method", "objective", "seed", "status", "execution_status",
    "stop_reason", "previous_attempt_count", "reused_from", *METRICS,
    "initial_l2_scaled_qp", "initial_used_layer_count",
    "initial_height_range", "initial_pillar_length", "initial_overlap_before_repair", "initial_repair_steps",
    "preprocessing_seconds", "initialization_seconds", "calibration_seconds", "repair_seconds",
    "repair_steps", "return_to_current_ratio", "repair_return_to_current_ratio", "uphill_accepted",
    "restarts", "last_best_at_seconds", "last_best_at_attempt", "small_exact_optimum",
    "small_exact_gap", "input_sha256", "report", "layout", "errors",
)
CONDITIONS = {
    "greedy-squared": ("stage1", "greedy", "Greedy"),
    "anneal-squared": ("stage1", "anneal", "Anneal | squared"),
    "restart-greedy-squared": ("stage1", "再スタート付き greedy", "Restart greedy"),
    "anneal-weighted-l1": ("stage2", "重み付き L1", "Anneal | weighted L1"),
    "anneal-pillar": ("stage2", "柱の総長", "Anneal | pillars"),
    "anneal-squared-plus-layers": ("stage2", "二乗距離＋層数罰則", "Anneal | layer penalty"),
}
STAGES = {
    "探索法の比較": tuple(method for method, (stage, _, _) in CONDITIONS.items() if stage == "stage1"),
    "目的関数の比較": (
        "anneal-squared",
        *(method for method, (stage, _, _) in CONDITIONS.items() if stage == "stage2"),
    ),
}
BASELINE = "anneal-squared"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def number(value, precision=3):
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{precision}f}"


def formatted_range(values, precision=3):
    if not values:
        return "—"
    return (
        f"{number(median(values), precision)} "
        f"[{number(min(values), precision)}–{number(max(values), precision)}]"
    )


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(cell).replace("|", r"\|") for cell in cells) + " |"

    return "\n".join(
        [line(headers), line("---" for _ in headers), *(line(row) for row in rows)]
    )


def condition_label(condition, stage):
    if condition == BASELINE and stage == 2:
        return "現行二乗距離"
    return CONDITIONS[condition][1]


def successful(rows):
    return [row for row in rows if row["status"] == "verified"]


def rows_for(progress, review, output, phase):
    checked = {case["key"]: case for case in review["cases"]}
    rows = []
    for case in progress["cases"]:
        if case["phase"] != phase:
            continue
        inspection = checked.get(case["key"], {})
        status = inspection.get("status", "pending")
        if inspection and inspection["path"] != case["path"]:
            raise ValueError(f"Independent verification refers to a previous attempt of {case['key']}")
        row = {
            "stage": case["stage"],
            "condition": f"{case['method']}-{case['objective']}",
            "case_key": case["key"],
            "example": case["fixture"],
            "method": case["method"],
            "objective": case["objective"],
            "seed": case["seed"],
            "status": status,
            "execution_status": case["status"],
            "stop_reason": case.get("reason"),
            "previous_attempt_count": len(case.get("previous_attempts", [])),
            "report": f"{case['path']}/report.json",
            "layout": f"{case['path']}/layout.json",
            "errors": " | ".join(inspection.get("errors", [])),
        }
        if status == "verified":
            report = read(output / row["report"])
            metrics = inspection["metrics"]
            timing = report["timing_seconds"]
            search = report["search"]
            best = report["history"][-1]
            row.update({key: metrics[key] for key in METRICS if key in metrics})
            for key in ("used_layer_count", "height_range", "pillar_length", "overlap_violation_count"):
                row[key] = int(row[key])
            row.update(
                total_seconds=case["wall_seconds"],
                search_seconds=timing["search"],
                attempts=search["attempts"],
                initial_overlap_before_repair=report["initial_repair"]["initial_violations"],
                initial_repair_steps=len(report["initial_repair"]["steps"]),
                preprocessing_seconds=timing["preprocessing"],
                initialization_seconds=timing["initialization"],
                calibration_seconds=timing["calibration"],
                repair_seconds=timing["repair"],
                repair_steps=search["repair_steps"],
                return_to_current_ratio=search["returned_to_current_ratio"],
                repair_return_to_current_ratio=search["repair_returned_to_current_ratio"],
                uphill_accepted=search["uphill_accepted"],
                restarts=len(search["restarts"]),
                last_best_at_seconds=best["elapsed_seconds"],
                last_best_at_attempt=best["attempt"],
                small_exact_optimum=metrics.get("exact_optimum"),
                small_exact_gap=metrics.get("exact_gap"),
                input_sha256=report["provenance"]["input_sha256"],
            )
            row.update({f"initial_{key}": metrics["initial_metrics"][key] for key in TRADEOFFS})
        rows.append(row)
    return rows


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return groups


def dashboard(rows):
    by_method = grouped(successful(rows))
    sections = []
    for stage, (title, methods) in enumerate(STAGES.items(), start=1):
        fields = MAIN_COLUMNS if stage == 1 else (
            "optimized_value", *MAIN_COLUMNS[:4], "l1_regularized", *MAIN_COLUMNS[4:]
        )
        chart = []
        paired = []
        baseline = {row["seed"]: row for row in by_method[BASELINE]}
        for condition in methods:
            cases = by_method[condition]
            chart.append(
                [
                    condition_label(condition, stage),
                    f"{len(cases)}/{len(SEEDS)}",
                    *(formatted_range([case[key] for case in cases]) for key in fields),
                ]
            )
            if condition == BASELINE:
                continue
            seed_pairs = [(row, baseline[row["seed"]]) for row in cases if row["seed"] in baseline]
            comparisons = []
            for key in TRADEOFFS:
                deltas = [row[key] - base[key] for row, base in seed_pairs]
                tied = sum(math.isclose(delta, 0, abs_tol=1e-8) for delta in deltas)
                lower = sum(delta < -1e-8 for delta in deltas)
                comparisons.append(
                    f"{number(median(deltas))} ({lower}/{tied}/{len(deltas) - tied - lower})"
                    if deltas else "—"
                )
            paired.append([condition_label(condition, stage), len(seed_pairs), *comparisons])
        sections.extend(
            [
                f"## {title}",
                "各セルは中央値 [最小–最大]。同じ seed のケースを対応付けた。"
                + ("最適化した値は各目的関数の尺度で記載した。" if stage == 2 else ""),
                table(["条件", "検証済み", *(METRICS[key] for key in fields)], chart),
                "焼きなまし・現行二乗距離に対する差の中央値と、同じ seed での減少／同値／増加の件数。",
                table(["条件", "ペア数", *(METRICS[key] for key in TRADEOFFS)], paired),
            ]
        )
    return sections


def summary_rows(rows):
    by_method = grouped(successful(rows))
    for stage_number, methods in enumerate(STAGES.values(), start=1):
        for condition in methods:
            cases = by_method[condition]
            for key in METRICS:
                values = [case[key] for case in cases]
                yield {
                    "stage": stage_number,
                    "condition": condition,
                    "metric": key,
                    "verified_cases": len(values),
                    "median": median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }


def small_section(rows, oracle):
    groups = defaultdict(list)
    for row in successful(rows):
        groups[(row["example"], row["condition"])].append(row)
    result = []
    for (example, condition), cases in sorted(groups.items()):
        gaps = [case["small_exact_gap"] for case in cases]
        exact = sum(math.isclose(gap, 0, abs_tol=1e-8) for gap in gaps)
        result.append(
            [example, CONDITIONS[condition][1], number(cases[0]["small_exact_optimum"], 7),
             formatted_range(gaps, 8), f"{exact}/{len(cases)}"]
        )
    return [
        "## 小規模例と全列挙",
        "レイヤーを 0 から始まる隙間のない整数に限定し、幾何学的な重なりの制約を満たす配置を全列挙した。"
        "差は各手法が最適化した尺度での探索最良値から全列挙最適値を引いた値。"
        "全列挙値と探索結果は保存済み配置から独立に照合した。",
        table(["例", "ハイパーエッジ", "重なり組数", "実行可能配置数"], [
            [entry[key] for key in ("name", "hyperedge_count", "overlap_count", "feasible_count")]
            for entry in oracle.get("fixtures", [])
        ]),
        table(["例", "条件", "全列挙最適値", "差の中央値 [最小–最大]", "最適値到達"], result),
    ]


def interpretation(rows, small_rows):
    by_method = grouped(successful(rows))
    if not any(by_method.values()):
        return ["## 結果の解釈", "本計測に検証済みの比較結果がない。"]

    paragraphs = ["## 結果の解釈"]
    search_methods = STAGES["探索法の比較"]
    objectives = STAGES["目的関数の比較"]
    if all(len(by_method[method]) == len(SEEDS) for method in search_methods):
        greedy = lower_count(by_method, BASELINE, "greedy-squared", "l2_scaled_qp")
        restart = lower_count(by_method, BASELINE, "restart-greedy-squared", "l2_scaled_qp")
        paragraphs.append(
            "180 秒で正規化二乗目的値の中央値が最小だった探索法は"
            f"{lowest_median(by_method, search_methods, 'l2_scaled_qp', 1)}。"
            f"同じ seed で焼きなましの二乗目的値が下回ったのは greedy に対し {greedy}、"
            f"再スタート付き greedy に対し {restart}だった。greedy の採否は改善と同値に限られる。"
            "今回の再スタートは別の探索経路を試せる一方、1 回の経路に使える時間を約 60 秒に区切る。"
        )
    if all(len(by_method[method]) == len(SEEDS) for method in objectives):
        weighted = lower_count(by_method, "anneal-weighted-l1", BASELINE, "l1_regularized")
        pillars = lower_count(by_method, "anneal-pillar", BASELINE, "pillar_length")
        fewer_layers = lower_count(by_method, "anneal-squared-plus-layers", BASELINE, "used_layer_count")
        lower_l2 = lower_count(by_method, "anneal-squared-plus-layers", BASELINE, "l2_scaled_qp")
        paragraphs.append(
            f"二乗目的値の中央値が最小だった条件は{lowest_median(by_method, objectives, 'l2_scaled_qp', 2)}、"
            f"使用レイヤー数では{lowest_median(by_method, objectives, 'used_layer_count', 2)}、"
            f"柱の総長では{lowest_median(by_method, objectives, 'pillar_length', 2)}。"
            f"層数罰則は現行二乗距離と比べ、同じ seed の {fewer_layers}で層を削減し、"
            f"正規化二乗目的値も小さかったのは {lower_l2}だった。"
        )
        paragraphs.append(
            "重み付き L1 はハイパーエッジのペアの距離を一次で扱う。今回の予算と修復規則で、"
            f"重み付き L1 値が現行二乗距離の配置を下回ったのは {weighted}だった。"
            "柱の総長は共有頂点ごとの端点間を直接評価する。"
            f"同じ seed で現行二乗距離より短い柱を得たのは {pillars}だった。"
            "層数と共有頂点の多い面どうしの近さを両方重視する場合は、"
            "層数罰則の削減幅と二乗目的値の増分を、上表の seed 別の対応差で判断できる。"
        )
    completed = successful(small_rows)
    if completed:
        at_optimum = sum(math.isclose(row["small_exact_gap"], 0, abs_tol=1e-8) for row in completed)
        paragraphs.append(
            f"小規模例では独立検証済み {len(completed)} 件中 {at_optimum} 件が全列挙値に到達した。"
            "実データの比較は予算内に発見した解にもとづく。探索法と目的関数の順位は、この入力、"
            "180 秒、seed 5 個、共通の二乗距離ベースの修復規則、層数罰則係数 100 の下で得た。"
        )
    return paragraphs


def lower_count(groups, condition, reference, key):
    baseline = {row["seed"]: row[key] for row in groups[reference]}
    values = [(row[key], baseline[row["seed"]]) for row in groups[condition] if row["seed"] in baseline]
    return f"{sum(value < other - 1e-8 for value, other in values)}/{len(values)} 件"


def lowest_median(groups, methods, key, stage):
    medians = {method: median(row[key] for row in groups[method]) for method in methods}
    best = min(medians.values())
    labels = "・".join(
        condition_label(method, stage)
        for method, value in medians.items()
        if math.isclose(value, best, rel_tol=1e-10)
    )
    return f" **{labels}**（{number(best)}）"


def figures(output, rows):
    if not successful(rows):
        return ["## 代表配置", "描画対象の検証済み本計測ケースなし。"]
    from plot_comparison import make_figures

    by_method = grouped(successful(rows))
    sample = successful(rows)[0]
    representatives = [("BFS + repair", (output / sample["report"]).parent / "initial.json")]
    for method, (_, _, label) in CONDITIONS.items():
        candidates = by_method[method]
        if not candidates:
            continue
        midpoint = median(row["optimized_value"] for row in candidates)
        selected = min(candidates, key=lambda row: (abs(row["optimized_value"] - midpoint), row["seed"]))
        representatives.append((f"{label} | seed {selected['seed']}", output / selected["layout"]))
    focus = make_figures(representatives, output)
    spans = table(
        ["代表配置", "注目頂点の最高・最低層の差"],
        [[row["panel"], number(int(row["vertex_span"]))] for row in focus["panels"]],
    )
    return [
        "## 代表配置",
        "各条件で最適化値が中央値に最も近い seed を選んだ。距離が同じ場合は seed の小さい方を選んだ。"
        "正射影の方位角は −60°、仰角は 34°。1 層の高さは XY 座標の 0.15 単位で、各図内で表示範囲も共通。",
        "[全体図](layout_comparison.png)・[共有頂点を強調した図](shared_vertex_comparison.png)",
        f"ポリゴン所属数が最多の共有頂点 **{focus['focus_vertex_label']}**"
        f"（ID {focus['focus_vertex_id']}、{focus['incident_polygons']} 面）を強調した。",
        spans,
        "全体図では、探索後も中央に輪郭と柱が密集し、その部分では個々の面を見分けにくい。"
        "強調図では、黄色の柱と所属層の点で同じ頂点を上下に追える。"
        "柱の総長は全頂点の合計なので、特定の共有頂点を追うときはその頂点の層差も確認する。",
    ]


def diagnostics(rows):
    keys = (
        "last_best_at_seconds", "last_best_at_attempt", "repair_steps", "repair_seconds",
        "repair_return_to_current_ratio", "uphill_accepted", "restarts",
    )
    by_method = grouped(successful(rows))
    return [
        "## 探索中の挙動",
        "各セルは中央値 [最小–最大]。最良値の更新履歴と近傍操作別の採用率は各ケースのレポートに記録した。",
        table(
            ["条件", "最後の更新（秒）", "最後の更新（試行）", "修復手数", "修復時間（秒）",
             "修復後原状復帰率", "悪化を許容した回数", "再スタート回数"],
            [[CONDITIONS[condition][1], *[
                formatted_range([case[key] for case in by_method[condition]]) for key in keys
            ]] for condition in CONDITIONS],
        ),
    ]


def settings_section(manifest, progress, rows, small_rows, review_path):
    verified = successful(rows)
    initial = verified[0] if verified else None
    plan = manifest["plan"]
    source = manifest["provenance"]
    env = source["environment"]
    return [
        "## 計測条件",
        f"入力は `original.json`（SHA-256 `{plan['input_sha256']}`）。"
        "頂点 627 件、ハイパーエッジ 383 件、重なり 4,431 組。頂点の XY と所属頂点順を固定した。",
        "全手法は同じ BFS＋修復から開始し、近傍比率と二乗距離による修復規則を共用した。"
        "正則化は 1e-5。焼きなましは各目的値に合わせ、同じ方法で温度を校正した。",
        "再スタート付き greedy は、終了処理用の 1 秒を除いたケース予算の 1/3 と 2/3 で"
        "派生 seed に切り替えて同じ初期解に戻り、通算最良を保持した。"
        "罰則係数は事前に 100 とした。1 層の削減を正規化二乗目的値 100 と同じ重みに置いた。",
        f"seed 0〜4。各ケース 180 秒、計測全体は 7,200 秒まで。"
        f"本計測の経過は {number(progress['measurement_seconds'])} 秒。"
        f"本計測は {len(verified)}/{len(rows)} 件、小規模は {len(successful(small_rows))}/{len(small_rows)} 件を検証済み。",
        "第 2 段階の現行二乗距離は第 1 段階の焼きなまし 5 件を共用し、"
        "異なる予算の過去の配置は今回の比較集計から外した。",
        (f"BFS＋修復の初期解は正規化二乗目的値 {number(initial['initial_l2_scaled_qp'])}、"
         f"{int(initial['initial_used_layer_count'])} 層、高さ幅 {int(initial['initial_height_range'])}、"
         f"柱の総長 {int(initial['initial_pillar_length']):,}。"
         f"BFS の重なり {initial['initial_overlap_before_repair']:,} 件を"
         f" {initial['initial_repair_steps']} 手で 0 件にした。") if initial else "",
        f"Git `{source['git_commit']}`。Python {env['python_version']}、NumPy {env['numpy']}、"
        f"SciPy {env['scipy']}、Shapely {env['shapely']}。数値ライブラリのスレッド数は 1。",
        f"[実験条件とソースのハッシュ](manifest.json)・[進捗と終了状態](progress.json)"
        f"・[独立検証]({review_path.name})",
    ]


def incomplete_section(rows, small_rows, smoke_rows, progress, review):
    unresolved = [row for row in [*rows, *small_rows, *smoke_rows] if row["status"] != "verified"]
    sections = ["## 失敗・中断・未実行"]
    if unresolved:
        sections.append(
            table(
                ["ケース", "実行状態", "検証状態", "終了理由・不一致"],
                [[row["report"].removesuffix("/report.json"), row["execution_status"], row["status"],
                  row["errors"] or row["stop_reason"] or "—"] for row in unresolved],
            )
        )
    else:
        sections.append("採用したケースはすべて実行と独立検証を完了した。未実行は 0 件。")
    previous = [
        [
            case["key"], f"[実行ログ]({attempt['path']}/run.log)", attempt["status"],
            number(attempt.get("wall_seconds")), attempt.get("reason", "—"),
        ]
        for case in progress["cases"] for attempt in case.get("previous_attempts", [])
    ]
    if previous:
        sections.extend(
            [
                "再試行前の記録を保管した。集計には検証済みの試行を用いた。",
                table(["ケース", "元の保存先", "実行状態", "経過秒", "終了理由"], previous),
            ]
        )
    if review["errors"]:
        sections.append("\n".join(f"- {error}" for error in review["errors"]))
    if progress.get("measurement_stop_reason"):
        sections.append(f"全体の停止理由は `{progress['measurement_stop_reason']}`。")
    return sections


def rerun_section():
    target = f"results/experiments/{STUDY}/reproduction"
    python = r".\.venv\Scripts\python.exe -B"
    return [
        "## 再実行",
        "リポジトリ直下から実行する。既存の結果は保管し、再試行ごとに新しい出力先を指定する。",
        f"```powershell\n{python} {SCRIPT}/run_experiment.py --output {target}\n"
        f"{python} {SCRIPT}/review_results.py {target} --output {target}/verification.json\n"
        f"{python} {SCRIPT}/summarize_results.py {target}\n```",
        "中断後は検証済み結果の JSON を `--review` に指定し、`--phase measure --resume` で再開する。"
        "検証済みケースは飛ばし、中断ケースには新しい試行ディレクトリを使う。",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    review_path = args.review or output / "verification.json"
    review = read(review_path)
    manifest = read(output / "manifest.json")
    progress = read(output / "progress.json")
    oracle_path = output / "small" / "oracle.json"
    oracle = read(oracle_path) if oracle_path.exists() else {}
    rows, small_rows, smoke_rows = [
        rows_for(progress, review, output, phase) for phase in ("measure", "small", "smoke")
    ]

    reused = []
    by_key = {row["case_key"]: row for row in rows}
    for descriptor in manifest["plan"]["reused_baselines"]:
        source = descriptor["source_case_key"]
        row = by_key.get(source)
        if row is not None:
            reused.append({**row, "stage": "stage2", "reused_from": source})
    write_csv(output / "cases.csv", CSV_CASES, [*rows, *reused])
    write_csv(output / "small_cases.csv", CSV_CASES, small_rows)
    write_csv(
        output / "medians.csv", ("stage", "condition", "metric", "verified_cases", "median", "min", "max"),
        summary_rows(rows),
    )

    report = [
        "# 根付き木によるレイヤー割り当ての比較",
        *settings_section(manifest, progress, rows, small_rows, review_path),
        "[本計測の seed 別データ](cases.csv)・[指標ごとの中央値と範囲](medians.csv)"
        "・[小規模の seed 別データ](small_cases.csv)",
        *dashboard(rows),
        *diagnostics(rows),
        *small_section(small_rows, oracle),
        *interpretation(rows, small_rows),
        *([] if args.skip_figures else figures(output, rows)),
        *incomplete_section(rows, small_rows, smoke_rows, progress, review),
        *rerun_section(),
    ]
    temporary = output / "summary.md.tmp"
    temporary.write_text("\n\n".join(paragraph for paragraph in report if paragraph) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.md")


if __name__ == "__main__":
    main()
