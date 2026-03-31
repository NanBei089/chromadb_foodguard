# Ingredient RAG Ranked Pipeline

## Goal
This pipeline adds post-retrieval filtering and reranking on top of the existing ingredient RAG flow.
It is designed to reduce noisy Chroma hits before the retrieved results are passed to a model.

## Why Filtering Is Needed
Vector recall can return partially related additives, parallel term groups, or long terms that only overlap on a fragment.
This layer keeps the strongest match as the primary result, optionally keeps one close secondary result, and clears low-confidence hits.

## Inputs And Outputs
- Input JSON: `other.ingredients.json`
- Knowledge source: `data/processed/gb2760_a1_grouped_min_final_prod.jsonl`
- Fallback source: `data/processed/gb2760_a1_grouped_min_final_v2.jsonl`
- Chroma dir: `chroma_db/`
- Collection: `gb2760_a1_grouped`
- Result JSON: `ingredient_rag_pipeline_ranked/data/ingredient_rag_ranked_results.json`
- Report: `ingredient_rag_pipeline_ranked/reports/ingredient_rag_ranked_report.md`

## Matching Rules
- High match:
  - exact term
  - query contained in `term` or `normalized_term`
  - exact alias hit
  - exact keyword hit
- Weak match:
  - strong character overlap
  - partial keyword overlap
- Empty:
  - no reliable candidate after filtering

## Rerank Scoring
- Exact normalized term: `+100`
- Exact term: `+95`
- Query contained in normalized term: `+80`
- Query contained in term: `+70`
- Alias hit: `+90`
- Keyword hit: `+60`
- Raw rank bonus: rank1 `+30`, rank2 `+20`, rank3 `+10`, rank4-5 `+5`
- Long irrelevant candidate penalty: `-20`
- No direct or weak relation penalty: `-30`

## How To Run
```powershell
python ingredient_rag_pipeline_ranked\scripts\retrieve_and_rerank_ingredients.py `
  --input "E:\GraduationProject\project\demo\upload_results\9d26ee9e-IMG_20251031_173657\other.ingredients.json" `
  --jsonl-source "data\processed\gb2760_a1_grouped_min_final_prod.jsonl" `
  --chroma-dir "chroma_db" `
  --collection "gb2760_a1_grouped"
```

## Next Integration
The ranked JSON is suitable for:
- FastAPI endpoints that need structured retrieval output
- prompt assembly where only the primary hit and one close backup should be provided to the model
- later business rules that treat `empty` items as non-additive or unresolved terms
