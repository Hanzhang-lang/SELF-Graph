#!/bin/bash

# Script to run inference using the ArG model.
# This script is based on the example provided in the README.md.
#
# Before running:
# - Ensure you have completed the steps in the "Setup" section of README.md.
# - Modify the placeholder variables below to match your environment and choices.
# - Make this script executable: chmod +x scripts/inference.sh

echo "Starting ArG inference process..."
echo "Please ensure you have configured the variables in this script."

# --- Configuration ---
# TODO: Set your GPU devices, model path, input file, output file, and other parameters.

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1" # Specify the GPU devices to use (e.g., "0,1" for two GPUs, "0" for one)

# Model and File Paths
MODEL_NAME_OR_PATH="/path/to/your/trained_model_checkpoint" # Path to the pre-trained model checkpoint
INPUT_FILE="./data/merged/WebQSP_test.json"                 # Path to the input file for inference
OUTPUT_DIR="./output/inference"                             # Base directory for output files
OUTPUT_FILENAME="webqsp_test_results.json"                  # Name for the output results file
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILENAME}"              # Full path to save the inference results

# Inference Parameters
WORLD_SIZE=2                # Number of GPUs for tensor parallelism (should match active CUDA_VISIBLE_DEVICES count)
MAX_NEW_TOKENS=100          # Maximum new tokens for the model to generate

# Embedding Caching (Redis is used by the script if caching is enabled)
# By default, caching is ENABLED in src/inference.py (default=True, action='store_false').
# To DISABLE caching, set ENABLE_CACHING to false or add --cached_embedding to the command.
ENABLE_CACHING=true # Set to "false" to disable caching, or "true" to enable it.

# --- End Configuration ---

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running inference with the following settings:"
echo "  CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "  Model Path: $MODEL_NAME_OR_PATH"
echo "  Input File: $INPUT_FILE"
echo "  Output File: $OUTPUT_FILE"
echo "  World Size (Num GPUs): $WORLD_SIZE"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
if [ "$ENABLE_CACHING" = true ]; then
    echo "  Embedding Caching: Enabled (via Redis, if configured in script)"
    CACHING_FLAG=""
else
    echo "  Embedding Caching: Disabled"
    CACHING_FLAG="--cached_embedding" # Presence of the flag disables caching
fi
echo ""

# Construct the command
CMD="python -m src.inference \
  --model_name \"$MODEL_NAME_OR_PATH\" \
  --input_file \"$INPUT_FILE\" \
  --output_file \"$OUTPUT_FILE\" \
  --world_size $WORLD_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS"

if [ "$ENABLE_CACHING" = false ]; then
    CMD="$CMD --cached_embedding"
fi

# Uncomment the line below to actually run the command, or run it manually after configuration.
echo "Executing command: $CMD"
# eval $CMD

# Check if the command was successful (if you uncomment 'eval $CMD')
# if [ $? -eq 0 ]; then
#     echo "Inference completed successfully."
#     echo "Results saved to $OUTPUT_FILE"
# else
#     echo "Inference failed. Please check the logs."
# fi

echo ""
echo "Inference script setup complete. Review the command above."
echo "To run, uncomment the 'eval \$CMD' line or copy the command into your terminal after setting variables."
