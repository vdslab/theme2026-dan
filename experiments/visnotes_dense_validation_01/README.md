# 全データでの共有頂点・遮蔽の検証

入力は現在の `original.json` と HGPolyVis の公開 DBLP TVCG 全体。入力順 first-fit と提案法の保存済み配置を使い、元の XY、頂点・辺 ID、所属順序を固定する。実験対象はレイヤー割り当ての成立、表示・操作、見かけの交差と実際の所属をどこまで区別できるかである。人の読み取り成績はこの実験の対象外とする。 [原計画](https://vdslab.esa.io/posts/9193)・[図と評価の改訂](https://vdslab.esa.io/posts/9274)

**検証結果。** 両入力合わせて12配置で重なり制約違反0。original は45→39層となった。seed 0 の提案では、長さを持つ411本の柱のうち298本が非所属面を貫いた。DBLP は209本中15本だった。元の全入力を扱えても、個々の所属を画面上で追えるかは密度と視点に左右される。 [実際のビューアを含む6枚の図と考察](https://vdslab.esa.io/posts/9278)・[全 seed の集計](../../results/experiments/visnotes_dense_validation_01/analysis/final/summary.json)・[柱と投影交差のグラフ](../../results/experiments/visnotes_dense_validation_01/summary-figure.png)

## 比較条件

- 全体の配置は決定的な入力順 first-fit と提案 seed 0〜4 を独立に再検証する。全共有頂点について所属層の最大値と最小値の差を求める。平面、方位45°・仰角35°および75°の正射影において、全対象面対で「投影面積が重なる」と「頂点IDを共有する」を照合する。面積を持たない辺も全体の配置・所属の検証に含める。配置は再探索しない。目安は各入力180秒、解析全体600秒。
- 実ビューアでは全体表示、事前に選定した共有頂点の選択、Zスライス、全範囲への復帰を seed 0 で比較する。比較画面は両ペインで視点・倍率を合わせる。対象は従来の実験と同じ `original.json` の ID 198 と DBLP の ID 48。チーム用の図では ID を表示する。
- 垂直柱の検査は全共有頂点と入力順 first-fit・提案 seed 0 を対象とする。非所属の有効な正面積ポリゴンがその頂点の XY を**内部**に含み、その面の層が長さのある柱の両端を含む区間に入る場合を数える。面との境界接触、輪郭のみの面、点・線はこの指標から除く。同じ層に真の所属点がある交差も区別する。面積交差・物体交差の件数はいずれも幾何学的な量である。
- 局所図は上記の頂点を含む面Aを起点にする。共有ありは共通IDが対象頂点1個のペア、共有なしは共通IDが0個のペア。元の有効なXY面で小さい方の10%以上が交差するペアのうち、交差領域の内部代表点を厳密に内包する第三の有効面が最も多い組を、同数なら辺IDの順に採る。手法別の結果を見る前に同じペア・真値を固定する。真上2Dと両2.5Dを各行共通の倍率で描く。解説用ペアの提示に絞ったことを明記し、全体の結論には全対集計を用いる。

出力には毎回新しいディレクトリを指定する。PDFは各著者・大学または arXiv の配信元から手元に保存し、ファイルとハッシュを取得記録に残す。

## 再現と閲覧

リポジトリのルートで実行する。`original.json` 単独なら、追跡済みの原本と保存配置を使って次の1行で全5 seedを照合できる。

~~~powershell
python -B experiments/visnotes_dense_validation_01/run_experiment.py --inputs original --output results/experiments/visnotes_dense_validation_01/original-retry
~~~

DBLP の入力と保存配置がすでにある作業ツリーでは次の準備を飛ばす。新しいチェックアウトで用意する場合は、公開データ用コードの固定版を別の作業ツリーに置き、そのランナーで現在のリポジトリ内の新しい出力先に生成する。保存実験はネイティブ探索を使用した。再生成する環境・バックエンドにより時間や解が変わり得る。 [公開データ準備と環境](https://github.com/vdslab/theme2026-dan/blob/f724b97e26e1f3c35b9173041a460bda90eee3a8/docs/visnotes-public-validation.md)

~~~powershell
$public = Join-Path (Get-Location) 'results/experiments/visnotes_public_validation_01'
$toolsDir = Join-Path ([IO.Path]::GetTempPath()) ('visnotes-public-' + [guid]::NewGuid())
git worktree add --detach $toolsDir f724b97e26e1f3c35b9173041a460bda90eee3a8
python -B "$toolsDir/experiments/visnotes_public_validation_01/prepare_inputs.py" --dataset dblp --output "$public/inputs"
python -B "$toolsDir/experiments/visnotes_public_validation_01/run_experiment.py" --phase benchmark --input dblp "$public/inputs/dblp.json" --input-manifest "$public/inputs/dblp.manifest.json" --condition greedy-input --condition full --output "$public/public-main"
~~~

次の一連のコマンドは両入力で新しい結果を作る。Chrome のキャプチャは Windows と、ビューアが使う Three.js の CDN 接続を前提とする。

~~~powershell
$code = 'experiments/visnotes_dense_validation_01'
$run = 'results/experiments/visnotes_dense_validation_01/retry'
$original = 'results/experiments/visnotes_remaining_01/exp04/original'
$public = 'results/experiments/visnotes_public_validation_01'
$dblp = "$public/public-main/dblp"
python -B "$code/run_experiment.py" --output "$run/analysis/final"
python -B "$code/pillar_contacts.py" --source original.json --first-fit "$original/greedy-input/deterministic/layout.json" --proposed "$original/full/seed-0/layout.json" --focus 198 --name original --output "$run/contacts/final/original"
python -B "$code/pillar_contacts.py" --source "$public/inputs/dblp.json" --first-fit "$dblp/greedy-input/deterministic/layout.json" --proposed "$dblp/full/seed-0/layout.json" --focus 48 --name DBLP-TVCG --output "$run/contacts/final/DBLP-TVCG"
python -B "$code/make_summary_figure.py" "$run" "$run/summary-figure.png"
node --experimental-websocket "$code/capture_viewer.mjs" "$run/viewer/full"
~~~

**手元で original を見る。** リポジトリのルートで `python -m http.server 8080 --bind 127.0.0.1` を実行し、次のリンクを開く。頂点検索で ID 198 を選ぶと、左に入力順 first-fit、右に提案 seed 0 が並ぶ。上限を11にしてから「全範囲を表示」で戻すと、報告と同じスライスを検査できる。 [全 original の比較ビューア](http://127.0.0.1:8080/viewer_visnotes.html?a=results/experiments/visnotes_remaining_01/exp04/original/greedy-input/deterministic/layout.json&b=results/experiments/visnotes_remaining_01/exp04/original/full/seed-0/layout.json)
