# VisNotes 改訂案の残作業と検証条件

esa 原文 §2.1・§3.2・§4 の残作業を、2026年11月25日に合意した現行データで進める。Git ブランチは `codex/visnotes-remaining-validation`。数値は現行の独自データと、その保存済み XY 変種の結果として扱う。図版・事例・動画はリポジトリ内で確認する草案とする。 [esa 原文](https://vdslab.esa.io/posts/9193)・[前回の実装条件](visnotes-implementation.md)

## 実装と検証の分担

監督者が入力、比較条件、予算、採否を決める。各実装担当は自分の配下に第三段の検証担当を置く。検証担当が差分評価、保存結果、図版のそれぞれで反例と再現方法を示し、実装担当が修正して再検証する。

## 計測条件

- `.venv` の Python と単一スレッド設定を共用し、ケースを逐次実行する。乱数を使う本計測の seed は 0、1、2、3、4。正則化は `1e-5`、複合目的の二乗距離は `l2_scaled_qp` を使う。反復上限は各ケース全体で20,000試行、壁時計上限は35秒。決定的な条件は1回実行する。
- 四つの実験は `results/experiments/visnotes_remaining_01/` の下に分けて保存する。exp02・exp03・exp04・exp05 の全体上限は順に40秒・720秒・540秒・480秒とする。
- exp02 は `original`、`paper`、`scalable`、`arranged` の入力監査。exp03 は `original` で4種類の目的族と全ての $\alpha$ を基準層数で比較し、$\alpha=0.25$ では上限を1層・2層増やした探索を加える。既存の小規模入力には別の出力先で全列挙・HiGHS MILP の照合を行う。
- exp04 は `original` で既存の12条件を比較する。時間切れの条件は記録し、連続 QP・LP が35秒で完了しない場合は参考計測として別ディレクトリに180秒の上限で実行する。共通上限の比較には実際の終了理由を用いる。
- exp05 は現行入力と3種の XY 変種で $K,B,F_{\mathrm{scaled}}$ と上位制約ハブを比較する。これらの4条件は一つの元データから得られた配置である。
- 検証では保存 JSON から所属、面積交差、目的値、層数、柱長を再計算する。証明できたクリーク下界と彩色上界が一致した場合に最小層数を確定する。失敗、未実行、予定した層段階の未到達と理由を集計に残す。

## 再実行

リポジトリのルートから起動する。ランナーは exp04、exp05、exp02、exp03、小規模の厳密照合、必要な連続解の参考計測の順に実行し、各段の開始、終了、計測条件を JSONL に記録する。既存の出力を持つ段は保持される。再計測時は全体または対象段に新しい出力パスを指定する。 [一括ランナー](../experiments/visnotes_remaining_01/run_experiment.py)

~~~powershell
.\.venv\Scripts\python.exe -B experiments/visnotes_remaining_01/run_experiment.py --plan-only
.\.venv\Scripts\python.exe -B experiments/visnotes_remaining_01/run_experiment.py --output results/experiments/visnotes_remaining_01_retry
~~~

## 図と事例

Fig. 1 草案は同じ入力の平面図、入力順 first-fit の2.5D、提案法の seed 0 の2.5D を比較する。事例の頂点は制約のある所属先が最多の頂点を固定し、全体の保存済み配置から同じ所属先を抜き出す。図内に抽出条件を示し、投影の違いとレイアウトの違いを区別する。主要比較図は本計測の集計を読み込み、各値を元の配置またはレポートまで追跡できるようにする。

## 保存済みの計測結果

exp04 の共通35秒枠で25/32ケースを検証した。連続 QP、連続 LP、根付き木の5 seed は時間切れだった。提案法の5 seed は39層・重なり違反0件で、柱総長の中央値は5,891、投影の多重被覆画素数の中央値は13,028。層挿入シフトを除いた条件の中央値は柱総長5,802、多重被覆画素数13,357だった。入力順 first-fit は45層、柱総長9,492、多重被覆画素数15,709。 [手法比較の集計](../results/experiments/visnotes_remaining_01/exp04/summary.json)

exp05 は保存済みの4入力・5 seed の全20ケースを検証した。未整形は39層、`arranged` は71層だった。共有頂点が0個の重なりは未整形の2,422組に対し、`arranged` は5,255組だった。XY 座標と頂点順序を含む保存済みの配置条件を比較した。 [XY 配置比較の集計](../results/experiments/visnotes_remaining_01/exp05/summary.json)・[入力監査](../results/experiments/visnotes_remaining_01/exp02/summary.json)

exp03 は全45ケースを検証した。$B+0.25F_{\mathrm{scaled}}$ の基準と上限を1層・2層増やした条件では、柱総長の中央値がそれぞれ5,891、6,284、6,336で、実使用層数はいずれも39だった。総試行数20,000を各段に配分した比較である。 [目的関数と追加層の集計](../results/experiments/visnotes_remaining_01/exp03/summary.json)

小規模27条件のうち19件を初回で検証した。実行ホスト中断後に残り8件を新しい出力先で完了し、計36層段で探索値、全列挙、HiGHS の最適値が一致した。初回の中断記録も保存している。 [初回の集計](../results/experiments/visnotes_remaining_01/exp03-small/summary.json)・[追試の実行条件](../results/experiments/visnotes_remaining_01/exp03-small-recovery-01/manifest.json)・[追試の集計](../results/experiments/visnotes_remaining_01/exp03-small-recovery-01/summary.json)

連続 QP・LP は別の180秒枠で検証した。QP は壁149.828秒、柱総長8,643.020、正規化二乗距離6,182.946。LP は壁60.719秒、柱総長8,614.0、正規化二乗距離6,233.656。共通枠の7件の時間切れは元の集計に残す。 [参考計測の集計](../results/experiments/visnotes_remaining_01/exp04-legacy-reference/summary.json)

層挿入の差分評価は第三段で40,080候補を完全再計算と照合し、固定 seed の探索比較250件でも配置、受理数、履歴の一致を確認した。同じ状態に対するキャッシュ利用時の候補評価は1.54〜2.05倍だった。キャッシュ構築を含む混合探索全体の高速化には追加の計測が必要。 [差分評価の実装](../solve_layers_compact.py)・[スパンの上下界の証明](visnotes-span-bound.md)

独立監査で保存済み119配置と141探索段の所属・制約・評価値を再計算した。根付き木5件の途中保存値も照合し、保存時点の参考値として監査結果に掲載した。初回監査では追試の相対パスと引数順の違いを検出した。実質的な実行条件を比較するように監査器を修正し、条件を変えた反例が拒否されることと、実際の結果が合格することを確認した。 [独立監査のコード](../experiments/visnotes_remaining_01/review_results.py)・[最終監査結果](../results/experiments/visnotes_remaining_01/independent-review-final.json)

## 図版と事例の出力

Fig. 1 の事例では、同じ22辺を固定して平面、入力順 first-fit、提案法の順に示した。入力順 first-fit から提案法 seed 0 で全体の層数は45から39、対象頂点のスパンは37から33になった。読み取りの正確さと所要時間は未評価。 [Fig. 1](../results/experiments/visnotes_remaining_01/figures/fig1-teaser.svg)・[事例記述](../results/experiments/visnotes_remaining_01/figures/case-study.md)

[Fig. 2 XY 配置比較](../results/experiments/visnotes_remaining_01/figures/fig2-xy-z.svg)・[Fig. 3 手法と探索の比較](../results/experiments/visnotes_remaining_01/figures/fig3-benchmark.svg)・[Fig. 4 目的関数と層上限](../results/experiments/visnotes_remaining_01/figures/fig4-objectives.svg)・[図版の入力とハッシュ](../results/experiments/visnotes_remaining_01/figures/figure-manifest.json)

補足動画は Chrome で操作・撮影した画面と字幕を収録した。両配置の表示、同じ頂点の選択、層範囲の絞り込み、連動回転を確認し、デコードした長さは59.94秒だった。図版・動画ともに現行データを使用した内部草案。 [補足動画](../results/experiments/visnotes_remaining_01/figures/video/visnotes-demo.webm)・[動画のメタデータ](../results/experiments/visnotes_remaining_01/figures/video/video-manifest.json)

同じ結果から図版と動画を再生成する場合は、リポジトリのルートで次を実行する。出力先は新規ディレクトリにする。 [図版生成器](../tools/make_visnotes_figures.py)・[動画生成器](../tools/make_visnotes_video.mjs)

~~~powershell
$run = 'results/experiments/visnotes_remaining_01'
$figures = "$run/figures-retry"
.\.venv\Scripts\python.exe -B tools/make_visnotes_figures.py --input original.json --first-fit "$run/exp04/original/greedy-input/deterministic/layout.json" --proposal "$run/exp04/original/full/seed-0/layout.json" --benchmark-summary "$run/exp04/summary.json" --xy-summary "$run/exp05/summary.json" --objective-summary "$run/exp03/summary.json" --output $figures
node --experimental-websocket tools/make_visnotes_video.mjs --figure-manifest "$figures/figure-manifest.json" --output "$figures/video"
~~~
