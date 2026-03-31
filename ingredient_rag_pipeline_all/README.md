# Ingredient RAG Pipeline All

## 本流程目标
本目录用于对 `other.ingredients.json` 中抽取出的全部 `items` 直接执行 RAG 检索，不做先验分类。流程目标是把明显可命中的食品添加剂规则召回出来，把无命中或低质量命中的项置空，作为后续模型生成的前置输入。

## 输入输出说明
输入：
- `other.ingredients.json`
- `chroma_db/`，collection=`gb2760_a1_grouped`
- `data/processed/gb2760_a1_grouped_min_final_prod.jsonl`
- 若 prod 不存在则回退 `data/processed/gb2760_a1_grouped_min_final_v2.jsonl`

输出：
- `ingredient_rag_pipeline_all/data/ingredient_rag_results.json`
- `ingredient_rag_pipeline_all/reports/ingredient_rag_report.md`
- `ingredient_rag_pipeline_all/scripts/retrieve_all_ingredients.py`

## 全量检索策略说明
1. 读取 `other.ingredients.json` 中的 `ingredients_text` 和 `items`
2. 对每个 item 的术语直接做检索，不做术语分类
3. 优先查询本地 Chroma collection `gb2760_a1_grouped`
4. 查询文本使用 item 的规范化术语
5. 命中后根据返回 id 回查 JSONL，补全 `term / normalized_term / aliases / function_category / rules / keywords`

## 命中质量判定逻辑
- `high`：top1 结果的 `term / normalized_term / aliases / keywords` 任一字段命中查询词
- `weak`：top1 不命中，但 top3 中存在任一结果命中查询词
- `empty`：无结果，或 top3 中没有任何一条结果命中查询词

## 如何运行
在项目根目录执行：

```powershell
python ingredient_rag_pipeline_all/scripts/retrieve_all_ingredients.py
```

## 下一步如何接模型生成
下一步可以把：
- 原始 `ingredients_text`
- 本流程输出的 `retrieval_results`

一起提供给模型。推荐策略是：
1. 只让模型重点解释 `match_quality=high/weak` 的项
2. 对 `empty` 项保持保守，不要强行映射到法规术语
3. 把回源的 `rules` 作为结构化上下文，避免模型自由发挥
