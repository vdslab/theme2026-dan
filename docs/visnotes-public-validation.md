# VisNotes の公開データ検証

公開データでの定量比較と Fig. 1、追加層の効果、XY 座標と頂点順序の切り分け、所属関係の読み取りを検証する。独自データは従来どおり内部比較に使う。投稿用の事例は公開入力で作る。 [esa 原文](https://vdslab.esa.io/posts/9193)・[研究概要](../research_overview.md)

## 公開データとベースライン

- HGPolyVis の DBLP TVCG と DBLP TPAMI を、固定した取得元から変換する。座標とポリゴンの頂点順序を保持する。2件は同じドメインの異なる比較ケースとして扱う。
- Marseille は SocioPatterns の時系列接触から決定的に抽出した極大クリークをハイパーエッジとする。比較のために生成した座標を全手法で固定する。使用規約、抽出条件、座標生成法は入力のマニフェストに記録する。
- Qu et al. §6 の first-fit は面を既存の最下層から試す。原論文は入力順を指定していないため、入力順、面の大きさ順、重なり次数順を比較する。平面配置を全手法で共有し、論文の手動編集は比較条件から外す。 [HGPolyVis](https://github.com/peterdanieloliver/HGPolyVis/tree/main/Data/PaperAuthor_DBLP)・[SocioPatterns](https://sociopatterns.org/datasets/high-school-dynamic-contact-networks/)・[Qu et al., §6](https://web.engr.oregonstate.edu/~zhange/images/Nary.pdf)

## 比較の実行

実行コードは [公開検証ランナー](../experiments/visnotes_public_validation_01/run_experiment.py)。[入力変換](../experiments/visnotes_public_validation_01/prepare_inputs.py) は、取得元のバイト列を固定ハッシュで照合する。原本、過去の結果、配布制約のあるデータを push に含めない。各試行を1スレッドで逐次実行し、通常は seed 0～4、1ケースの壁時計上限35秒、基準20,000試行とする。結果には各入力・ソース・利用したネイティブライブラリのハッシュ、計算予算、終了理由を残す。C++ 版が標準パスにあれば既存CLIが使用し、レポートに実際のバックエンドを記録する。

| フェーズ | 条件と確認内容 |
| --- | --- |
| `benchmark` | first-fit 3順、DSatur、段階的な近傍追加、根付き木、連続QP・LP、4種の目的族を選択できる。最終配置の重なり制約と評価値を独立再計算し、固定正射影での多重被覆と投影柱長を保存する。 |
| `warm` | 基準上限を固定した20,000試行、同じ総数を追加層の各段に配分した条件、各段20,000試行の条件、追加層を許さず同じ追加試行数を使った条件を比較する。実使用層数、柱総長、二乗距離、ハブのスパンと各段の最良解を保存する。 |
| `xy` | 座標と境界上の頂点順序を2×2に交差する。無効面と補修後に空になる面を記録する。全条件で有効な共通の辺に限定した比較も実行し、全入力と部分集合の結果を分けて解釈する。 |

~~~powershell
$run = 'results/experiments/visnotes_public_validation_01'
python -B experiments/visnotes_public_validation_01/prepare_inputs.py --output "$run/inputs"
python -B experiments/visnotes_public_validation_01/run_experiment.py --phase benchmark --input dblp "$run/inputs/dblp.json" --input marseille "$run/inputs/marseille.json" --input-manifest "$run/inputs/dblp.manifest.json" --input-manifest "$run/inputs/marseille.manifest.json" --output "$run/benchmark-retry"
~~~

中断や予算超過は集計に記録する。再開時は新しい出力先を指定し、未完了の入力・条件・seed を `--input`、`--condition`、`--seed` で選ぶ。図版は [図の生成](../experiments/visnotes_public_validation_01/figures.py) を用い、入力順の first-fit と seed 0 の提案法から作る。対象頂点は重なり制約のある所属辺数、全所属辺数、入力順で選ぶ。

[XYの交差条件の生成](../experiments/visnotes_public_validation_01/xy_controls.py) は `--common-valid` で無効面のない共通部分を出力する。[複数視点の投影比較](../experiments/visnotes_public_validation_01/projection_sensitivity.py) は、ラスタの多重被覆画素に加えて面積のある交差対を数える。[小規模厳密照合](../experiments/visnotes_public_validation_01/exact_subgraphs.py) は、同じ公開入力から選ばれた6辺で、全列挙とHiGHSの結果を保存する。

## 読み取り評価と投稿判断

2D、Qu 型 first-fit、提案法の所属判定で正誤と応答時間を測る。参加者から取得した測定値を、ブラウザーの動作確認や配置の代理指標と区別する。12月14日の判定では、公開入力での図の読み取りと、複数入力における独立検証済みの定量結果を確認する。Fig. 1 の同一視点比較では、層数と全体の柱総長に加えて、選択頂点のスパンも比較する。 [読み取り課題の実施・採点](../experiments/visnotes_reading_study_01/README.md)
