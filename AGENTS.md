# AGENTS.md

[研究概要](research_overview.md) に、研究の目的と現行の数理定式化をまとめている。

## 共通ルール

- **実装・実験・検証に着手する前に**、対象の作業ツリーのルートにある `AGENTS.md` をファイルから読み、担当箇所に適用される指示とスキルを確認する。読める状態を確認してから作業を始める。 [着手前の指示確認](docs/development-workflow.md#着手前の指示確認)
- 作業を委任する場合は、担当する作業ツリーのルートと `AGENTS.md` のパスを渡す。委任先は着手前に自分でファイルを読み、確認したパスと担当作業に適用するルールを親エージェントに伝える。再委任先にも同じ手順を適用する。
- 研究計画の見直しについて判断、質問、実装・実験の計画を行う前に、[esa の原文](https://vdslab.esa.io/posts/9193) を確認し、原文の方針と未決事項に基づいて進める。リポジトリ内の関連資料は [研究計画の見直し](research_plan_revision.md) を参照する。原文を取得できない場合は、その旨を明示する。
- 新しい実験に着手する前に、実験専用の Git ブランチを作成し、そのブランチで作業する。
- 実験用ブランチでの `git add` と `git commit`、同ブランチへの `git push`、同ブランチからのプルリクエスト作成を許可する。ステージする差分と送信先は [Git とプルリクエスト](docs/development-workflow.md#git-とプルリクエスト) に沿って確認する。
- このリポジトリに関する esa への投稿を常時許可する。実験終了後は [esa への実験報告](.agents/skills/report-experiment-to-esa/SKILL.md) に従い、指定カテゴリに投稿して記事 URL を記録する。
- 原本データ、既存の表示用 JSON、過去の実験結果を上書きしない。
- 作業中に手順やスキルの誤り・不足が分かったら、その場で関連する Markdown を更新し、必要な手順書を追加する。 [作業手順とスキルの改善](docs/development-workflow.md#作業手順とスキルの改善)
- 完了報告には、変更した箇所で確かめたことと未検証の項目を記載する。検証方法は [変更別の確認手順](docs/development-workflow.md#変更別の確認手順) を参照する。

## スキル

作業に着手するときに、該当するスキルを読む。

- [実験の実行と検証](.agents/skills/run-experiment/SKILL.md) — 比較条件と予算、動作確認、成果物の保存、結果の検証、中断と再開。
- [esa への実験報告](.agents/skills/report-experiment-to-esa/SKILL.md) — 実験終了後の報告内容、投稿先、認証、投稿できなかった場合の保存。

## 参照資料

- 全体の構成とコマンドは [README](README.md)。
- レイヤー割り当ての設計は [根付き木の実装計画](rooted_tree_layer_assignment_plan.md) と [ダミーレイヤー探索の実装計画](dummy_layers_implementation_plan.md)。
- XY 平面の処理は [自動配置手法の調査](polygon_auto_layout_research.md)、[ポリゴン整形](polygon_layout_reproduction.md)、[整形後の配置調整](polygon_arrangement.md)。
- 保存済みの計測条件と結果は [過去の実験と結果](docs/experiment-history.md) と [根付き木の探索法・目的関数の比較](results/experiments/tree_optimization_comparison_01/summary.md)。
