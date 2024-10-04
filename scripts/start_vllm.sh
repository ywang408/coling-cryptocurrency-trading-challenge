#!/bin/bash

JSON_FILE="configs/main.json"
MODEL=$(jq -r '.chat_config.chat_model' "$JSON_FILE")
LORA=$(jq -r '.chat_config.lora // "false"' "$JSON_FILE")
TENSOR_PARALLEL_SIZE=$(jq -r '.chat_config.tensor_parallel_size // 2' "$JSON_FILE")

if [ "$LORA" = "false" ]; then

  docker run --runtime nvidia --gpus all \
    --env-file .env \
    -v .:/workspace \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.2 \
    --model $MODEL \
    --dtype auto \
    --guided-decoding-backend lm-format-enforcer \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --download-dir /workspace/model_data/$MODEL

else
    LORA_BASE_MODEL=$(jq -r '.chat_config.lora_base_model' "$JSON_FILE")
    LORA_PATH=$(jq -r '.chat_config.lora_path' "$JSON_FILE")
  
    docker run --runtime nvidia --gpus all \
      --env-file .env \
      -v .:/workspace \
      -p 8000:8000 \
      --ipc=host \
      vllm/vllm-openai:v0.6.2 \
      --model $LORA_BASE_MODEL \
      --enable-lora \
      --lora-modules $MODEL=/workspace/$LORA_PATH \
      --dtype auto \
      --guided-decoding-backend lm-format-enforcer \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --download-dir /workspace/model_data/$LORA_BASE_MODEL
fi
