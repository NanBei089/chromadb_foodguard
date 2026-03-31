# RAG 开发文档

## 1. 目录定位

`rag/` 目录负责把《GB 2760-2024 食品添加剂使用标准》附录 A 表 A.1 转成可检索的本地知识库，并在此基础上完成：

- 法规 PDF 抽取与结构化
- 添加剂级别知识单元构建
- 本地 Chroma 向量库入库
- 检索验证与报表输出
- 面向 `demo/` 产物的配料 RAG 召回实验

这里的“权威数据源”不是 `chroma_db/`，而是 `data/processed/*.jsonl`。Chroma 只是可重建索引层。

## 2. 目录结构

| 路径 | 作用 |
| --- | --- |
| `data/GB2760.pdf` | 原始法规 PDF |
| `scripts/` | 主链路脚本，负责抽取、聚合、清洗、入库、查询 |
| `output/` | PDF 首轮抽取产物 |
| `data/processed/` | 经过聚合、清洗、冲突剔除后的正式 JSONL/调试文件/报表 |
| `chroma_db/` | 正式检索使用的本地 Chroma 库 |
| `rag_validation/` | 独立验证集、独立验证 collection、验证报表 |
| `ingredient_rag_pipeline_all/` | 对 `other.ingredients.json` 做“直接召回”的实验管线 |
| `ingredient_rag_pipeline_ranked/` | 在召回基础上增加规则过滤与 rerank 的实验管线 |
| `tmp/`、`tmp_chroma_meta_test/` | 临时调试输出，不应作为长期数据源 |
| `pylibs/` | 预留或本地依赖目录，当前不是主入口 |

## 3. 开发环境

建议从 `e:\GraduationProject\foodguard\rag` 目录执行下面的命令。该目录下多数脚本都依赖相对路径，直接在仓库根目录运行会找不到 `data/GB2760.pdf`、`data/processed` 等资源。

推荐环境：

- Python 3.13
- Windows PowerShell
- 本地磁盘可写，用于创建 `output/`、`data/processed/`、`chroma_db/`

安装依赖：

```powershell
cd e:\GraduationProject\foodguard\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gb2760-a1.txt
python -m pip install chromadb
```

说明：

- `requirements-gb2760-a1.txt` 只覆盖 PDF 抽取链路的依赖：`PyMuPDF`、`pdfplumber`、`pandas`
- 向量入库、查询、验证脚本额外依赖 `chromadb`
- 当前检索嵌入不是外部模型，而是项目内实现的 512 维哈希 n-gram 向量，因此不依赖云端 embedding 服务

## 4. 主链路总览

完整数据流如下：

```text
data/GB2760.pdf
  -> scripts/extract_gb2760_a1.py
  -> output/gb2760_a1_rules.jsonl
  -> scripts/build_gb2760_a1_grouped_min.py
  -> data/processed/gb2760_a1_grouped_min.jsonl
  -> scripts/finalize_gb2760_grouped_min.py
  -> data/processed/gb2760_a1_grouped_min_final.jsonl
  -> scripts/finalize_gb2760_grouped_min_v2.py
  -> data/processed/gb2760_a1_grouped_min_final_v2.jsonl
  -> scripts/build_gb2760_a1_prod.py
  -> data/processed/gb2760_a1_grouped_min_final_prod.jsonl
  -> scripts/ingest_chroma_gb2760.py
  -> chroma_db/gb2760_a1_grouped
```

正式开发时，`*_prod.jsonl` 应视为“线上可用数据集”；`*_final.jsonl`、`*_final_v2.jsonl` 是中间清洗结果。

## 5. 主链路脚本说明

### 5.1 `scripts/extract_gb2760_a1.py`

作用：

- 从 `data/GB2760.pdf` 中定位附录 A 表 A.1
- 解析添加剂元信息、食品分类、最大使用量、备注
- 输出首轮逐行规则 JSONL

关键实现特征：

- 默认假定 A.1 位于 PDF 第 9 到 103 页
- 通过检查第 104 页是否出现 `A.2` 来验证页码边界
- 对跨页元信息、续表、续行做了合并
- 将使用量分成 `numeric`、`qs`、`forbidden`、`text` 等类型

输出：

- `output/gb2760_a1_rules.jsonl`
- `output/gb2760_a1_preview.json`
- `output/gb2760_a1_debug.txt`

适用场景：

- 法规版本更新后重抽
- 发现食品分类、备注、使用量抽取不准确时调整抽取逻辑

### 5.2 `scripts/build_gb2760_a1_grouped_min.py`

作用：

- 将“逐条规则记录”按 `(normalized_term, function_category)` 聚合为“添加剂级别记录”
- 去掉完全重复的 rule
- 为每个添加剂生成 `keywords` 和 `embedding_text`

输出：

- `data/processed/gb2760_a1_grouped_min.jsonl`
- `data/processed/gb2760_a1_grouped_pretty.json`
- `data/processed/gb2760_a1_grouped_report.md`

这一层的数据结构已经适合做向量入库，但还没有充分处理括号别名、术语归一化和冲突项。

### 5.3 `scripts/finalize_gb2760_grouped_min.py`

作用：

- 从 `term` 中恢复 `aliases`
- 修正 `normalized_term`
- 再次去重 `rules`
- 生成更适合检索的 `keywords` / `embedding_text`

输出：

- `data/processed/gb2760_a1_grouped_min_final.jsonl`
- `data/processed/gb2760_a1_grouped_min_final_pretty.json`
- `data/processed/gb2760_a1_grouped_min_final_report.md`
- `data/processed/gb2760_a1_grouped_min_final_debug.json`

这一层开始出现“人工复核样例”，适合排查括号内容到底应不应该被当成别名。

### 5.4 `scripts/finalize_gb2760_grouped_min_v2.py`

作用：

- 对 `normalized_term` 做进一步规范化
- 清洗 `food_category_name`
- 检查同一 `(food_category_code, food_category_name)` 下是否存在互相冲突的 `usage_limit`
- 缩短 `embedding_text`
- 为并列术语补充关键词

输出：

- `data/processed/gb2760_a1_grouped_min_final_v2.jsonl`
- `data/processed/gb2760_a1_grouped_min_final_v2_pretty.json`
- `data/processed/gb2760_a1_grouped_min_final_v2_report.md`
- `data/processed/gb2760_a1_grouped_min_final_v2_debug.json`

如果你要调“召回质量”，通常应优先从这一层开始看，而不是直接改 Chroma。

### 5.5 `scripts/build_gb2760_a1_prod.py`

作用：

- 根据 `*_final_v2_debug.json` 中识别出的冲突样例，把有冲突的记录单独剔出
- 生成“正式可用”的 prod JSONL

输出：

- `data/processed/gb2760_a1_grouped_min_final_prod.jsonl`
- `data/processed/gb2760_a1_conflict_records.json`
- `data/processed/gb2760_a1_grouped_min_final_prod_report.md`

这一步的原则是保守：宁可把存在冲突的添加剂剔出去，也不直接带入检索库。

### 5.6 `scripts/ingest_chroma_gb2760.py`

作用：

- 将最终 JSONL 入库到本地 Chroma collection `gb2760_a1_grouped`
- 入库前会清空目标 collection
- 为每条记录生成本地哈希向量

输出：

- `chroma_db/`
- 可选 `--summary-json`

注意：

- 该脚本会清空并重建指定 collection，不要把 `chroma_db/` 当成手工维护资产
- 嵌入逻辑和查询逻辑是项目内自定义实现，改动时要同步更新多个脚本

### 5.7 `scripts/query_chroma_gb2760.py`

作用：

- 对正式 collection 做命令行查询
- 适合快速检查术语召回、类别召回、得分组成是否合理

输出：

- 控制台结果
- 可选 `--summary-json`

## 6. 推荐构建顺序

从零构建一次法规知识库，建议按下面顺序执行：

```powershell
cd e:\GraduationProject\foodguard\rag

python scripts\extract_gb2760_a1.py --input data\GB2760.pdf --output-dir output
python scripts\build_gb2760_a1_grouped_min.py --input output\gb2760_a1_rules.jsonl
python scripts\finalize_gb2760_grouped_min.py
python scripts\finalize_gb2760_grouped_min_v2.py
python scripts\build_gb2760_a1_prod.py
python scripts\ingest_chroma_gb2760.py --summary-json tmp\chroma_ingest_summary.json
python scripts\query_chroma_gb2760.py --summary-json tmp\chroma_query_summary.json
```

如果只是在已有 `*_prod.jsonl` 上重建索引，可直接从 `ingest_chroma_gb2760.py` 开始。

## 7. 验证子系统

`rag_validation/` 是正式索引之外的一套隔离验证环境，目的是在不污染主 collection 的前提下评估召回质量。

目录职责：

- `rag_validation/data/test_queries.json`：验证问题集
- `rag_validation/scripts/build_test_chroma.py`：把当前 JSONL 入库到独立 collection `gb2760_a1_validation`
- `rag_validation/scripts/run_validation.py`：对验证问题集批量跑 top-k，输出指标和失败样例
- `rag_validation/reports/`：构建摘要、验证结果 JSON、Markdown 报表

运行方式：

```powershell
cd e:\GraduationProject\foodguard\rag
python rag_validation\scripts\build_test_chroma.py
python rag_validation\scripts\run_validation.py
```

关注指标：

- `top1_term_hit_rate`
- `top3_term_hit_rate`
- `category_hit_rate`
- `worst_query_type`

这套脚本已经把数据层问题也纳入验证，包括：

- 空 `embedding_text`
- 重复 `id`
- 必要字段缺失

## 8. 与 `demo/` 联调的两个实验管线

### 8.1 `ingredient_rag_pipeline_all/`

特点：

- 输入 `other.ingredients.json`
- 不做术语先验分类
- 对每个配料词直接召回 Chroma
- 只要 top1 或 top3 中有匹配项，就保留结果

适合：

- 观察“纯召回层”能力
- 给大模型提供更宽松的候选

运行示例：

```powershell
cd e:\GraduationProject\foodguard\rag
python ingredient_rag_pipeline_all\scripts\retrieve_all_ingredients.py `
  --input ..\demo\upload_results\<image_id>\other.ingredients.json `
  --jsonl data\processed\gb2760_a1_grouped_min_final_prod.jsonl `
  --chroma-dir chroma_db `
  --collection gb2760_a1_grouped
```

输出：

- `ingredient_rag_pipeline_all/data/ingredient_rag_results.json`
- `ingredient_rag_pipeline_all/reports/ingredient_rag_report.md`

### 8.2 `ingredient_rag_pipeline_ranked/`

特点：

- 先取更多候选，再做规则过滤和 rerank
- 只保留 primary 命中，必要时保留一个接近候选
- 重点抑制长词干扰、弱相关别名和错误并列项

适合：

- 减少噪声召回
- 给后续问答或报告生成提供更干净的上下文

运行示例：

```powershell
cd e:\GraduationProject\foodguard\rag
python ingredient_rag_pipeline_ranked\scripts\retrieve_and_rerank_ingredients.py `
  --input ..\demo\upload_results\<image_id>\other.ingredients.json `
  --jsonl-source data\processed\gb2760_a1_grouped_min_final_prod.jsonl `
  --chroma-dir chroma_db `
  --collection gb2760_a1_grouped
```

输出：

- `ingredient_rag_pipeline_ranked/data/ingredient_rag_ranked_results.json`
- `ingredient_rag_pipeline_ranked/reports/ingredient_rag_ranked_report.md`

## 9. 关键数据文件说明

| 文件 | 是否建议手改 | 说明 |
| --- | --- | --- |
| `output/gb2760_a1_rules.jsonl` | 否 | 首轮抽取结果，主要用于回溯抽取逻辑 |
| `data/processed/gb2760_a1_grouped_min.jsonl` | 否 | 添加剂聚合后的最小单元 |
| `data/processed/gb2760_a1_grouped_min_final.jsonl` | 否 | 首轮清洗后的正式候选 |
| `data/processed/gb2760_a1_grouped_min_final_v2.jsonl` | 否 | 二次清洗与冲突检测后的候选 |
| `data/processed/gb2760_a1_grouped_min_final_prod.jsonl` | 是，作为发布源头进行人工复核 | 当前最接近“上线数据”的文件 |
| `data/processed/gb2760_a1_conflict_records.json` | 是 | 记录被排除的冲突项，适合人工检查 |

## 10. 调试建议

### 10.1 改 PDF 抽取规则时

先看：

- `output/gb2760_a1_preview.json`
- `output/gb2760_a1_debug.txt`

重点检查：

- 是否误把续表标题当正文
- 是否丢了跨页的添加剂元信息
- `usage_limit` 是否被截断成半个数字

### 10.2 改术语归一化时

优先查看：

- `data/processed/gb2760_a1_grouped_min_final_debug.json`
- `data/processed/gb2760_a1_grouped_min_final_v2_debug.json`
- `data/processed/gb2760_a1_conflict_records.json`

重点检查：

- 括号内容被当成别名是否合理
- 并列化学名是否被错误拆散
- 同类食品分类下是否出现相互冲突的限量

### 10.3 改检索算法时

需要同时关注以下文件中的嵌入/评分逻辑是否一致：

- `scripts/ingest_chroma_gb2760.py`
- `scripts/query_chroma_gb2760.py`
- `rag_validation/scripts/validation_common.py`
- `ingredient_rag_pipeline_all/scripts/retrieve_all_ingredients.py`
- `ingredient_rag_pipeline_ranked/scripts/retrieve_and_rerank_ingredients.py`

如果只改其中一个脚本，主链路与验证链路会出现评分标准不一致的问题。

## 11. 已知限制

1. 若干脚本和历史报告里仍保留旧的绝对路径前缀 `E:\GraduationProject\project\...`。在当前仓库路径 `e:\GraduationProject\foodguard` 下开发时，建议显式传入 `--input`、`--jsonl`、`--jsonl-source`。
2. 当前向量方案是本地哈希向量，不是语义模型 embedding。优点是离线、稳定；缺点是对复杂别名、类别型问句和长术语泛化一般。
3. `build_gb2760_a1_prod.py` 采用“冲突即剔除”的保守策略，因此 prod 数据可能比 v2 少。
4. `data/processed/` 和若干历史 README/报告中已有部分乱码内容，通常是旧文件在不同编码环境下生成的结果。新脚本统一按 UTF-8 写出。

## 12. 推荐维护原则

1. `data/processed/*.jsonl` 才是正式数据源，任何上线发布或学术报告都优先引用这里的最终产物。
2. 变更抽取或归一化规则后，至少重跑到 `*_final_v2.jsonl`，不要只改中间层。
3. 调整检索评分后，必须同步重跑 `rag_validation`，否则无法判断改动是真的提升还是只改变了排序表面现象。
4. 需要接入 `demo/` 时，优先使用 `*_prod.jsonl + chroma_db/`，不要让实验脚本直接依赖更早阶段的中间 JSONL。
