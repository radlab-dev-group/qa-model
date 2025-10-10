#!/bin/bash

CACHE_DIR=../cache/

GEN_MODEL_MAP_NAME=google-gemma3
GEN_MODEL_PATH=/mnt/data2/llms/models/community/google/gemma-3-12b-it

QA_MODEL_MAP_NAME=radlab-qa
QA_MODEL_PATH=/mnt/data2/llms/models/radlab-open/qa/best_model/quantized/best_model-bitsandbytes

CUDA_VISIBLE_DEVICES=1 python3 generate-with-qa.py \
  --gen-model-path="${GEN_MODEL_PATH}" \
  --qa-model-path="${QA_MODEL_PATH}" \
  --cache-dir="${CACHE_DIR}" \
  --qa-name="${QA_MODEL_MAP_NAME}" \
  --gen-name="${GEN_MODEL_MAP_NAME}"
