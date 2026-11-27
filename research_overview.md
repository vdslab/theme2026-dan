# 2.5次元表現を用いたポリゴンベースのハイパーグラフ可視化

IEEE PacificVis 2027 の VisNotes（Visualization Notes）への投稿を目指す。 [VisNotes 募集要項](https://pacificvis2027.github.io/contribute/short-papers/)

## 研究の背景

ハイパーグラフでは、複数の頂点が関わる関係を一つのハイパーエッジで表す。ポリゴンベースの可視化では、各ハイパーエッジに属する頂点を結んだポリゴンを描き、頂点の位置と所属関係を示す。データセットが大きくなると、平面上でポリゴンが重なり、個々のハイパーエッジや共有頂点を見分けにくくなる。重なりの判定やレイアウトの最適化に要する計算時間も課題となる。

- **Multi-Plane CW Complex Layout** は、平面で重なるポリゴンを first-fit で平行な層に分離し、共有頂点を垂直な柱で結ぶ。固定した XY 配置に対し、このレイヤー割り当てを比較の基準にする。 [Qu et al. (2017), §6](#参考文献)
- **Scalable Hypergraph Visualization** は、頂点やハイパーエッジを統合・削除する操作で入力を段階的に簡略化し、小規模な状態でレイアウトを求める。その後、簡略化を逆順に戻しながら局所的にレイアウトを更新し、最終的には元のハイパーグラフ全体の配置を復元する。 [Oliver et al. (2024)](#参考文献)
- **Structure-Aware Simplification for Hypergraph Visualization** は、ハイパーグラフを二部グラフとして分析して、平面上で重なりが避けられない箇所を特定する。その情報を用い、サイクルなどの重要な構造の保存を考慮しながら簡略化する。 [Oliver et al. (2025)](#参考文献)

簡略化された表示では、一部の要素や所属関係が集約される。個々の頂点・ハイパーエッジ・所属関係を同じ表示で確認する必要があるデータセットでは、元の構造を保ったまま、重なりによる見づらさを減らす方法が求められる。

## 研究の目的

入力の頂点、ハイパーエッジ、所属関係をすべて保持したまま、Qu et al. の多層ポリゴン表現の層数と柱長を抑える。平面上の頂点座標とポリゴンの頂点順序を固定し、XY 平面上で面積を持って重なるポリゴンを異なる高さのレイヤーに割り当てる。複数レイヤーに現れる共有頂点には、同じ XY 座標のまま、所属先の最低レイヤーから最高レイヤーまで垂直な柱を描く。異なる XY 配置が積層に与える影響と、配置が所属関係の読み取りに与える影響も評価する。 [比較ビューア](viewer_visnotes.html)

レイヤーを増やすと、同一面内の重なりを解消しやすくなる。可視化全体の高さや柱の長さが増えると、共有頂点を追いにくくなる可能性がある。そこで、使用レイヤー数を抑えつつ、共有頂点の多いハイパーエッジ同士を近いレイヤーに置く方法を検討する。

## 現行手法と数理定式化

### レイヤー割り当てと重なり制約

ハイパーグラフを $H=(V,E)$、ハイパーエッジを $e_i\in E$、その所属頂点集合を $V_i\subseteq V$、平面上のポリゴンを $P_i$ とする。実装では、交差面積が $10^{-9}$ を超えるポリゴンの組を重なり集合 $C$ に含める。境界のみで接する組は、同じレイヤーへの配置を許す。

$$
C=\{(i,j)\mid i<j,\ \operatorname{area}(P_i\cap P_j)>10^{-9}\}.
$$

ハイパーエッジ $e_i$ のレイヤー番号 $z_i$ を直接探索する。$C$ の各対に層が異なるという制約を課し、上限 $K_{\mathrm{cap}}$ の範囲で順序付きの彩色を求める。

$$
z_i\in\{0,\ldots,K_{\mathrm{cap}}-1\},\qquad
|z_i-z_j|\ge 1\quad ((i,j)\in C).
$$

彩色数の下界と実行可能解の層数が一致した入力では、最小層数を証明できる。 [重なり判定](solve_layers_dsatur_l1.py#L46)・[直接探索](solve_layers_compact.py)

### 共有頂点数で重み付けした目的関数

$W_{ij}=|V_i\cap V_j|$ を、異なる二つのハイパーエッジが共有する頂点の数とする。平面上で面積が重ならない組も含む全ペアを評価に用いる。実装では非対角の重みに正則化項 $\varepsilon$（既定値 $10^{-5}$）を加え、重み行列 $A=(a_{ij})$ を構成する。

$$
a_{ij}=W_{ij}+\varepsilon\quad (i\ne j),\qquad a_{ii}=0.
$$

二乗距離項 $F$ は、共有頂点の多い組ほど同じ高さ差に対する費用が大きくなるように定める。

$$
F(z)=\sum_{i<j} a_{ij}(z_i-z_j)^2.
$$

探索と結果報告では、既存の `solve_layers.py` と尺度を揃えた値 $F_{\mathrm{scaled}}$ を使う。$L$ は $A$ から作るラプラシアン行列で、$\mathbf{1}$ は全要素が 1 のベクトルである。

$$
L=\operatorname{diag}(A\mathbf{1})-A,\qquad
s=\begin{cases}
1/\max_{i,j}|L_{ij}| & (L\ne 0),\\
1 & (L=0),
\end{cases}
\qquad F_{\mathrm{scaled}}(z)=2sF(z).
$$

コード上の `l2_scaled_qp` が $F_{\mathrm{scaled}}$ に対応する。主条件では柱総長との複合目的 $B(z)+0.25F_{\mathrm{scaled}}(z)$ を最小化する。共有頂点数で重み付けした二乗距離の上下界はハブのスパンと関係するため、ハブのスパンも別に計測する。 [重みと正規化係数](solve_layers_dsatur_l1.py#L100)・[スパンの上下界](docs/visnotes-span-bound.md)

### 探索手順と評価指標

提案手法では DSatur で初期彩色し、単体移動、Kempe 鎖交換、層交換、層挿入シフトを使って制約を保ちながら焼きなまし法で探索する。重なり制約のないハイパーエッジは一座標ごとの厳密な最小化を繰り返す。上限を増やす条件は、前段の最良解を引き継いで評価する。根付き木ベースの探索は比較対象に含める。 [直接探索の実装](solve_layers_compact.py)・[根付き木の設計](rooted_tree_layer_assignment_plan.md)

使用レイヤー数 $K(z)$、高さ幅 $R(z)$、柱の総長 $B(z)$ は評価指標として記録する。頂点 $v$ が属するハイパーエッジの添字集合を $I(v)=\{i\mid v\in V_i\}$ とすると、表示時の Z 方向の倍率を適用する前の値は次のとおりである。

$$
K(z)=|\{z_i\mid i=1,\dots,|E|\}|,\qquad R(z)=\max_i z_i-\min_i z_i,
$$

$$
B(z)=\sum_{v:\,I(v)\ne\varnothing}
\left(\max_{i\in I(v)}z_i-\min_{i\in I(v)}z_i\right).
$$

比較では $K$、$R$、$B$、$F_{\mathrm{scaled}}$、ハブのスパン、固定正射影での多重被覆画素数、実行時間を併記する。多重被覆と柱長は配置指標であり、実際の読み取りは正解率と応答時間で評価する。 [独立した指標計算](experiments/visnotes_metrics.py)

## 現時点の結果

`original.json` では39クリークと39色の実行可能解が一致した。二乗距離を共通の目的とした5 seed の比較で、根付き木から直接探索に変更すると使用層数の中央値は45から39、正規化二乗距離の中央値は4,578から3,260になった。複合目的の直接探索では使用層数39、柱総長の中央値5,891で、重なり違反は0件だった。 [手法比較とビューア](https://vdslab.esa.io/posts/9224)・[根付き木の過去の結果](results/reports/layers_hypergraph_report.json)

同じ入力と試行数を使ったC++探索の比較では、全6組で最終配置と目的値が一致した。固定上限5 seed の実行ごとの速度比の中央値は、探索部分12.80倍、CLI全体1.74倍だった。 [C++の計測](https://vdslab.esa.io/posts/9229)

## 目指している結果

- **定量評価**。同一レイヤーの重なり制約違反を 0 件に保ち、少ないレイヤー数で、比較対象より小さい目的値を得ることを目指す。高さ幅、柱の総長、計算時間も併記し、目的値の改善とレイヤー数の削減をどの程度両立できるかを調べる。
- **可視化による評価**。ポリゴンの輪郭と頂点の所属が見分けやすく、異なるレイヤーにある共有頂点を柱に沿って追いやすい表示を目指す。同じデータを用いた比較図で、重なりや柱の密集が表示にどう影響するかを確認する。

## 今後の課題

1. **データと再現性**。公開データの複数領域で比較し、再現できる Fig. 1 を作る。独自データの論文掲載と再配布の範囲は未確定。
2. **追加層と XY の効果**。追加の層と追加の試行予算を分けて評価する。XY 座標と境界上の頂点順序を別々に変え、共有頂点のない面同士の重なりと彩色数への影響を調べる。
3. **読み取りの評価**。2D、Qu 型 first-fit、提案法で共通の所属関係を問う。正解率と所要時間を記録し、計算された配置・投影指標と区別して報告する。 [研究計画の見直し](research_plan_revision.md)

## 参考文献

1. Botong Qu et al. *Interactive Design and Visualization of N-ary Relationships*. SIGGRAPH Asia 2017 Symposium on Visualization. [原論文 §6](https://web.engr.oregonstate.edu/~zhange/images/Nary.pdf)
2. Peter Oliver, Eugene Zhang, and Yue Zhang. *Scalable Hypergraph Visualization*. IEEE Transactions on Visualization and Computer Graphics, 30(1), 595–605, 2024. [DOI: 10.1109/TVCG.2023.3326599](https://doi.org/10.1109/TVCG.2023.3326599)
3. Peter Oliver, Eugene Zhang, and Yue Zhang. *Structure-Aware Simplification for Hypergraph Visualization*. IEEE Transactions on Visualization and Computer Graphics, 31(1), 667–676, 2025. [DOI: 10.1109/TVCG.2024.3456367](https://doi.org/10.1109/TVCG.2024.3456367)
