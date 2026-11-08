#!/usr/bin/env bash
FUSIONRAG=false python3 search_locomo_BGE_rerank.py \
  --db-path ./lancedb_data_GLM-4.5-Air \
  --dataset ./test_ref/locomo10.json \
  --output-dir ./results_locomo_GLM_BGE_rerank --index-dir ./indexes/GLM_BGE \
  --embedding-model-path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --reranker-model /mnt/qjhs-sh-lab-04/models/bge-reranker-v2-m3 \
  --devices cuda:4,cuda:5,cuda:6,cuda:7 \
  --candidate-limit 96 --total-limit 10 --batch-size 32 --reranker-max-length 192 \
  --workers 128 --judge-workers 8 \
  --llm-base-url http://127.0.0.1:30013/v1 --llm-model GLM-4.5-Air \
  --judge-base-url http://192.168.0.239:31002/v1 --judge-model GLM-5.3
