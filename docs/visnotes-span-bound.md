# 共有頂点のスパンと二乗距離

esa の改訂計画で示された上下界を、実装で比較する目的関数との関係を含めて導く。 [esa 原文 §1・§2.1](https://vdslab.esa.io/posts/9193)・[研究計画の見直し](../research_plan_revision.md)

## 上下界

頂点 $v$ の所属ハイパーエッジの集合を $I(v)$、その個数を $d_v$ とする。$d_v\ge 1$ のとき、所属エッジの層番号の最小値を $m_v$、最大値を $M_v$、平均を $\mu_v$、スパンを $s_v=M_v-m_v$ とする。$d_v=0$ のときは $s_v=0$ とする。共有頂点数 $W_{ij}=|V_i\cap V_j|$ による二乗距離は、頂点ごとの和として書ける。

$$
F_{\mathrm{shared}}(z)
=\sum_{i<j}W_{ij}(z_i-z_j)^2
=\sum_v\sum_{\substack{i<j\\ i,j\in I(v)}}(z_i-z_j)^2.
$$

固定した頂点 $v$ について、内側の和を展開すると $d_v\sum_{i\in I(v)}(z_i-\mu_v)^2$ になる。$s_v>0$ の場合、最小値と最大値を取る2項だけで

$$
(m_v-\mu_v)^2+(M_v-\mu_v)^2
=\frac{s_v^2}{2}
 +2\left(\mu_v-\frac{m_v+M_v}{2}\right)^2
\ge \frac{s_v^2}{2}.
$$

一方、$(z_i-m_v)(M_v-z_i)\ge 0$ を各 $i$ について平均すると、分散の上界が得られる。

$$
\frac{1}{d_v}\sum_{i\in I(v)}(z_i-\mu_v)^2
\le (M_v-\mu_v)(\mu_v-m_v)
\le \frac{s_v^2}{4}.
$$

スパンが0の頂点では二乗距離の和も0である。頂点ごとの上下界を足し合わせると

$$
\frac12\sum_v d_v s_v^2
\le F_{\mathrm{shared}}(z)
\le \frac14\sum_v d_v^2 s_v^2.
$$

## 目的関数との関係

柱の総長 $B(z)=\sum_v s_v$ は、所属先の最小層と最大層が変わらない移動を評価しない。二乗距離は層の組ごとに差を評価し、下界では同じスパンでも所属数が多い頂点に大きな重みが付く。例えば $d_v>0$ なら $s_v\le\sqrt{2F_{\mathrm{shared}}(z)/d_v}$ が成り立つ。配置間で複合目的を比較しても、個々の頂点のスパンが一律に減る保証はないため、ハブのスパンは別に集計する。

実装で $B+\alpha F_{\mathrm{scaled}}$ に用いる二乗距離は $F_{\mathrm{scaled}}=2s(F_{\mathrm{shared}}+\varepsilon\sum_{i<j}(z_i-z_j)^2)$ である。$s$ は入力ごとの正規化係数、$\varepsilon$ は非対角の正則化係数である。上の上下界の対象は共有頂点重みだけの $F_{\mathrm{shared}}$ である。$\varepsilon\ge 0$ なら下界に $2s$ を掛けた値は $F_{\mathrm{scaled}}$ の下界にもなり、上界には正則化分を加える。 [現行の目的関数の定義](../research_overview.md#共有頂点数で重み付けした目的関数)
