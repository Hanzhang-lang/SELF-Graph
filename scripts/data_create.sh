#!/bin/bash

# This script automates the data creation process for the ArG project.
# It runs three main Python scripts:
# 1. relation_data.py: Extracts relation paths from a knowledge graph dataset.
# 2. generate_data.py: Uses LLMs to score and augment the data from step 1.
# 3. composite_data.py: Combines the scored data into a final training format.
#
# Before running:
# - Ensure you have completed the steps in the "Setup" section of README.md.
# - Modify the placeholder variables below to match your environment and choices.
# - Uncomment the execution blocks you wish to run.

echo "Starting ArG data creation process..."
echo "Please ensure you have configured the variables in this script."

# --- Configuration ---
# Common settings
DATASET_NAME="webqsp"  # Options: "webqsp", "cwq", or your custom dataset identifier
DATA_SPLIT="train"     # Options: "train", "validation", "test", etc.
NUM_PROCESSES=8        # Number of processes for relation_data.py

# Output directories (ensure these exist or the script can create them)
BASE_OUTPUT_DIR="./output"
CHAIN_DATA_DIR="${BASE_OUTPUT_DIR}/chain_data"
GENERATED_DATA_DIR="${BASE_OUTPUT_DIR}/generate"
FINAL_DATA_DIR="${BASE_OUTPUT_DIR}/final_training_data"

# Create directories if they don't exist
mkdir -p "$CHAIN_DATA_DIR"
mkdir -p "$GENERATED_DATA_DIR"
mkdir -p "$FINAL_DATA_DIR"

# File paths based on variables
RELATION_DATA_OUTPUT_PATH="${CHAIN_DATA_DIR}/${DATASET_NAME}_${DATA_SPLIT}_chain_data.json"

# generate_data.py specific settings
LLM_TASK="all" # Options: "all", "r_relevance", "e_relevance", "reasoness", "utility"
LLM_N_COMPLETIONS=3
LLM_TEMPERATURE=1.0

# OpenAI settings (uncomment and set if using OpenAI)
OPENAI_MODEL_NAME="gpt-3.5-turbo"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
OPENAI_BASE_URL="YOUR_OPENAI_BASE_URL_IF_NEEDED" # e.g., for proxies
GENERATED_OPENAI_OUTPUT_PATH="${GENERATED_DATA_DIR}/${DATASET_NAME}_${DATA_SPLIT}_generated_openai.json"

# Azure OpenAI settings (uncomment and set if using Azure)
AZURE_API_KEY="YOUR_AZURE_API_KEY"
AZURE_ENDPOINT="YOUR_AZURE_ENDPOINT"
AZURE_DEPLOYMENT_NAME="YOUR_AZURE_DEPLOYMENT_NAME" # e.g., "gpt-35-turbo"
AZURE_API_VERSION="YOUR_AZURE_API_VERSION"       # e.g., "2023-05-15"
GENERATED_AZURE_OUTPUT_PATH="${GENERATED_DATA_DIR}/${DATASET_NAME}_${DATA_SPLIT}_generated_azure.json"

# composite_data.py settings - choose one of the generated files as input
# INPUT_FOR_COMPOSITE="${GENERATED_OPENAI_OUTPUT_PATH}" # If using OpenAI output
# INPUT_FOR_COMPOSITE="${GENERATED_AZURE_OUTPUT_PATH}" # Or if using Azure output
FINAL_COMPOSED_OUTPUT_PATH="${FINAL_DATA_DIR}/${DATASET_NAME}_${DATA_SPLIT}_composed.json"


# --- Step 1: Process Relation Data ---
echo ""
echo "Step 1: Processing relation data using src.data_creation.relation_data.py"
echo "Dataset: ${DATASET_NAME}, Split: ${DATA_SPLIT}"
echo "Output will be saved to: ${RELATION_DATA_OUTPUT_PATH}"
echo "Number of processes: ${NUM_PROCESSES}"
# Uncomment the following block to run this step:
# ------------------------------------------------------------------------------
# python -m src.data_creation.relation_data \
#   --dataset "$DATASET_NAME" \
#   --split "$DATA_SPLIT" \
#   --save_path "$RELATION_DATA_OUTPUT_PATH" \
#   --n_proc "$NUM_PROCESSES" \
#   --save
#
# if [ $? -ne 0 ]; then
#     echo "Error in relation_data.py. Exiting."
#     exit 1
# fi
# echo "Relation data processing complete."
# ------------------------------------------------------------------------------


# --- Step 2: Generate Data using LLMs ---
# Choose either OpenAI or Azure OpenAI by uncommenting the relevant block.
# Ensure the CHAIN_DATA_PATH variable points to the output of Step 1.

# Option 2a: Using OpenAI
echo ""
echo "Step 2a: Generating data using OpenAI (src.data_creation.generate_data.py)"
echo "Input chain data: ${RELATION_DATA_OUTPUT_PATH}"
echo "OpenAI Model: ${OPENAI_MODEL_NAME}"
echo "Output will be saved to: ${GENERATED_OPENAI_OUTPUT_PATH}"
# Uncomment the following block to run this step with OpenAI:
# ------------------------------------------------------------------------------
# python -m src.data_creation.generate_data \
#   --chain_data "$RELATION_DATA_OUTPUT_PATH" \
#   --model_name "$OPENAI_MODEL_NAME" \
#   --task "$LLM_TASK" \
#   --output_file "$GENERATED_OPENAI_OUTPUT_PATH" \
#   --openai_api_key "$OPENAI_API_KEY" \
#   --openai_base_url "$OPENAI_BASE_URL" \
#   --n "$LLM_N_COMPLETIONS" \
#   --temperature "$LLM_TEMPERATURE"
#
# if [ $? -ne 0 ]; then
#     echo "Error in generate_data.py (OpenAI). Exiting."
#     exit 1
# fi
# echo "Data generation with OpenAI complete."
# INPUT_FOR_COMPOSITE="${GENERATED_OPENAI_OUTPUT_PATH}" # Set for next step
# ------------------------------------------------------------------------------


# Option 2b: Using Azure OpenAI
echo ""
echo "Step 2b: Generating data using Azure OpenAI (src.data_creation.generate_data.py)"
echo "Input chain data: ${RELATION_DATA_OUTPUT_PATH}"
echo "Azure Deployment: ${AZURE_DEPLOYMENT_NAME}"
echo "Output will be saved to: ${GENERATED_AZURE_OUTPUT_PATH}"
# Uncomment the following block to run this step with Azure OpenAI:
# ------------------------------------------------------------------------------
# python -m src.data_creation.generate_data \
#   --chain_data "$RELATION_DATA_OUTPUT_PATH" \
#   --use_azure \
#   --task "$LLM_TASK" \
#   --output_file "$GENERATED_AZURE_OUTPUT_PATH" \
#   --azure_api_key "$AZURE_API_KEY" \
#   --azure_endpoint "$AZURE_ENDPOINT" \
#   --azure_deployment "$AZURE_DEPLOYMENT_NAME" \
#   --azure_api_version "$AZURE_API_VERSION" \
#   --n "$LLM_N_COMPLETIONS" \
#   --temperature "$LLM_TEMPERATURE"
#
# if [ $? -ne 0 ]; then
#     echo "Error in generate_data.py (Azure OpenAI). Exiting."
#     exit 1
# fi
# echo "Data generation with Azure OpenAI complete."
# INPUT_FOR_COMPOSITE="${GENERATED_AZURE_OUTPUT_PATH}" # Set for next step
# ------------------------------------------------------------------------------


# --- Step 3: Composite Data ---
echo ""
echo "Step 3: Compositing data using src.data_creation.composite_data.py"
echo "Input generated data path: \${INPUT_FOR_COMPOSITE} (Ensure this is set correctly based on Step 2)"
echo "Output will be saved to: ${FINAL_COMPOSED_OUTPUT_PATH}"
# Ensure INPUT_FOR_COMPOSITE is set (e.g., uncomment one of the lines at the end of Step 2 blocks)
# Uncomment the following block to run this step:
# ------------------------------------------------------------------------------
# if [ -z "\$INPUT_FOR_COMPOSITE" ] || [ ! -f "\$INPUT_FOR_COMPOSITE" ]; then
#     echo "Error: INPUT_FOR_COMPOSITE is not set or the file does not exist."
#     echo "Please ensure Step 2 (generate_data.py) completed successfully and uncomment the appropriate INPUT_FOR_COMPOSITE line."
#     # exit 1 # Optionally exit if the input is missing
# else
#     python -m src.data_creation.composite_data \
#       --input_file "\$INPUT_FOR_COMPOSITE" \
#       --output_file "\$FINAL_COMPOSED_OUTPUT_PATH"
#
#     if [ \$? -ne 0 ]; then
#         echo "Error in composite_data.py. Exiting."
#         exit 1
#     fi
#     echo "Data compositing complete."
# fi
# ------------------------------------------------------------------------------

echo ""
echo "Data creation script steps outlined. Please uncomment and configure the sections you need to run."
echo "Final outputs, if all steps are run, should be in '${FINAL_DATA_DIR}'."
echo "Make this script executable with: chmod +x scripts/data_create.sh"
