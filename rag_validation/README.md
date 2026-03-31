# GB 2760 RAG Validation

## 测试目标
本目录用于对 GB 2760-2024 A.1 已清洗后的最终 JSONL 和本地 Chroma 向量库做独立验证，重点覆盖数据层、索引层、检索层和结果层。

验证目标参考 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 的工程化思路：
- 数据校验：字段完整性、空文本、重复 ID
- 索引校验：Chroma collection 创建、批量写入、可查询性
- 检索校验：标准名、别名、并列术语、规则型、类别型查询
- 结果校验：top1 / top3 命中、metadata 合理性、document 可支撑后续生成

## 输入文件
优先使用：
- `data/processed/gb2760_a1_grouped_min_final_prod.jsonl`

自动回退：
- `data/processed/gb2760_a1_grouped_min_final_v2.jsonl`

## 输出文件
- `rag_validation/data/test_queries.json`
- `rag_validation/scripts/build_test_chroma.py`
- `rag_validation/scripts/run_validation.py`
- `rag_validation/reports/build_summary.json`
- `rag_validation/reports/validation_results.json`
- `rag_validation/reports/validation_report.md`
- `rag_validation/chroma_db/`

## 运行方式
在项目根目录执行：

```powershell
python rag_validation/scripts/build_test_chroma.py
python rag_validation/scripts/run_validation.py
```

## 指标说明
- `term_hit`：top3 结果中任意 `term` 或 `normalized_term` 命中期望术语
- `category_hit`：top3 结果中任意 `function_category` 命中期望类别
- `top1_term_hit`：top1 结果命中期望术语
- `top3_term_hit`：等价于 `term_hit`
- `query_type_hit_rate`：按查询类型统计 top3 术语命中率
