# VisNotes 改訂計画の実装条件

2026年11月25日の判断を記録する。研究の位置付け、提案手法、評価の方針は、2026年11月24日更新の esa 原文を確認して進める。 [esa 原文](https://vdslab.esa.io/posts/9193)・[研究計画の見直し](../research_plan_revision.md)

## 確定した方針

- 改訂案を採用する。DSatur を起点とする整数層の近傍探索と複合目的を主力とし、根付き木は既存手法として比較する。固定 XY で整形条件の違いを計測し、Three.js 版のビューアを別ファイルに実装する。
- 作業ブランチは `codex/visnotes-layer-optimization`。実装と検証には新規ファイルを作り、原本、既存表示用 JSON、保存済みの結果を参照する。
- `masuta.csv` の出典は独自データ。今回の実装と動作確認は既存の `original.json` と保存済みの XY 配置をそのまま入力に使う。公開範囲は未決事項として扱う。
- メインの Codex が条件の決定と完了判定を担当する。実装担当は自分の配下に敵対的検証担当を置く。検証担当が反例と再現方法を示し、実装担当が修正して検証をやり直し、監督者が差分と結果を確認する。

## 実験条件

- 目的関数の二乗距離項は既存の正規化二乗目的値 `l2_scaled_qp` を使う。この定義と係数を記録する。esa の予備試作と同じ尺度であるかは、予備試作の設定を入手できた時点で照合する。 [既存の定義](../research_overview.md#共有頂点数で重み付けした目的関数)
- DSatur の彩色数は上界とし、確認したクリーク下界と一致する場合に最小層数を確定する。固定上限と上限を1層・2層増やしたケースについて、実際に使った層数、柱総長、正規化二乗距離、ハブのスパンを記録する。
- 初回の動作確認は seed 0 と短い試行または小規模の厳密解を使う。比較用の各 CLI で、実行条件、配置、実行時間、終了理由を出力し、別計算で実行可能性と指標を確かめる。全 seed の本計測では、着手前にケースと全体の計算予算を確定する。

## 実行方法

リポジトリのルートから、レイアウト JSON とレポートに新規のパスを指定する。ウォームスタートの指定時は `--max-iterations` が全段の試行数上限となり、各段の上限、解、指標、探索履歴がレポートに入る。 [レイヤー割り当てソルバー](../solve_layers_compact.py)

~~~powershell
.\.venv\Scripts\python.exe -B solve_layers_compact.py original.json results/experiments/exp03_objective_hub_pareto/local-smoke/layout.json --report results/experiments/exp03_objective_hub_pareto/local-smoke/report.json --seed 0 --max-iterations 250 --warm-start-extra-layers 2
~~~

実験ランナーの `--output` には毎回新規ディレクトリを指定する。各ランナーの `--help` で入力、比較条件、seed、試行数、時間予算の指定を確認できる。子プロセスはランナーと同じ Python で実行する。

| 実験 | 実装内容 |
| --- | --- |
| [exp02](../experiments/exp02_clean_benchmarks_prep/run_experiment.py) | 保存済み入力の来歴、重複所属、共有頂点数別の重なり、クリーク下界、DSatur 上界を記録する。 |
| [exp03](../experiments/exp03_objective_hub_pareto/run_experiment.py) | 目的関数と層上限を比較する。小規模入力を指定すると全列挙と HiGHS MILP を照合する。 |
| [exp04](../experiments/exp04_solver_benchmark/run_experiment.py) | 貪欲、DSatur、QP・LP、根付き木、提案法と近傍アブレーションを比較し、固定正射影のオーバードローと投影柱長を測る。 |
| [exp05](../experiments/exp05_xy_z_paradox_study/run_experiment.py) | 未整形と3種の整形済み入力を比べる。各変種の所属を照合し、XY 座標とポリゴン境界順序の変化を記録する。 |

ビューアはローカル HTTP サーバーから開く。Three.js の読み込みには CDN への接続が必要。配置の JSON は画面のファイル選択で開くか、同じサーバー上のパスを `?a=...` と `&b=...` に指定する。 [新ビューア](../viewer_visnotes.html)

~~~powershell
.\.venv\Scripts\python.exe -m http.server --bind 127.0.0.1 8000
~~~

[ローカルのビューア](http://127.0.0.1:8000/viewer_visnotes.html)

## 動作確認

現行 `original.json` で、DSatur と提案法（seed 0、250試行、上限39層）を同じ条件で検証した。独立した STRtree 判定で重なり4,431組を確認し、保存済み2配置の違反はともに0件だった。39色の実行可能な解と39クリークにより、現行入力の最小層数は39。 [保存済み集計](../results/experiments/exp04_solver_benchmark/smoke_seed0_20261125/summary.json)・[入力、条件とソースのハッシュ](../results/experiments/exp04_solver_benchmark/smoke_seed0_20261125/manifest.json)

| 手法 | 層数 | 柱総長 | 正規化二乗距離 | 投影の多重被覆画素数 | 投影の重複描画サンプル数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSatur | 39 | 9,013 | 7,375.94 | 14,673 | 52,775 |
| 提案法 | 39 | 7,457 | 4,836.68 | 13,972 | 55,284 |

同じケースは以下のコマンドで再実行できる。全 seed の比較、esa の予備性能の再現、論文用の視認性評価は未実施。

~~~powershell
.\.venv\Scripts\python.exe -B experiments/exp04_solver_benchmark/run_experiment.py --output results/experiments/exp04_solver_benchmark/retry_seed0 --inputs original --conditions dsatur full --seeds 0 --max-iterations 250 --case-budget 25 --total-budget 75
~~~

## Windows での検証

この作業環境では、既存テストの一時ファイルを `os.replace` で保存する処理が通常のサンドボックスで `WinError 5` になった。自動審査を通した実行で同じ `.venv` とテストコマンドを使い、既存テスト93件が通過することを確認した。旧ソルバーの保存を含む検証で同じエラーが出た場合も、この方法で再実行する。 [実行承認の手順](../.agents/skills/run-experiment/references/execution.md#実行承認と一括実行)
