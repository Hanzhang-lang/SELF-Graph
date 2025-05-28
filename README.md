# ArG: Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection

Official implementation of the paper "ArG: Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection". This project introduces ArG, a framework designed to enhance Knowledge Base Question Answering (KBQA) by actively learning how to retrieve relevant information and reason over it through a process of self-reflection. We warmly welcome discussions and collaborations in the field of KBQA!

The ArG framework, illustrating its workflow, is depicted in the following diagram:

![WordFlow](./ArG.png)

## Getting Started

Welcome to ArG! This project enables you to leverage active self-reflection for retrieving and reasoning on knowledge graphs. Here's a typical workflow to get you started:

1.  **Set up your environment**: Begin by preparing your Python environment and installing all necessary dependencies. Detailed instructions can be found in the **[Setup](#setup)** section.
2.  **Prepare your data**: The next step is to process your knowledge graph data and generate the specific formats required for training. This is primarily done using the `scripts/data_create.sh` script. Please refer to the **[Data Creation](#data-creation)** subsection under **[Training](#training)** for detailed instructions on configuring and running this script.
3.  **Train your model**: Once the data is created, you can proceed to train your model using the generated datasets. (Further details on the training script and its parameters would typically be here.)
4.  **Run inference**: After you have a trained model, you can use it to answer questions via the `scripts/inference.sh` script. The **[Inference](#inference)** section provides details on configuring and running this script, and interpreting the outputs.

Following these steps will guide you through using the ArG project from initial setup to performing inference on your knowledge graph data.

## Setup

This project requires Python 3.10 or newer.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rmanluo/ArG.git
    cd ArG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is provided. Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on FAISS:** The `requirements.txt` includes `faiss-cpu`. If you have a CUDA-enabled GPU and want to use GPU-accelerated FAISS, you might need to install `faiss-gpu` instead. Please refer to the [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for specific instructions and ensure it's compatible with your CUDA version.
    *   **Note on VLLM:** VLLM has specific CUDA requirements. Please check the [VLLM documentation](https://vllm.ai/) for compatibility with your GPU and CUDA version.
    *   **Note on `walker`:** The import `import walker` is present in `src/data_creation/relation_data.py`. This appears to be a local module or a specific library not commonly found on PyPI. If it's a custom module within this project, ensure it's correctly placed in the `PYTHONPATH`. If it's an external dependency, its specific installation source needs to be identified.

4.  **Redis Server (Optional for Caching):**
    The scripts can use `RedisStore` for caching embeddings, which can speed up repeated initializations. If you plan to use this feature:
    *   Ensure you have a Redis server running. You can install Redis via your system's package manager (e.g., `sudo apt-get install redis-server` on Debian/Ubuntu) or run it using Docker.
    *   The default Redis URL used in the scripts is `redis://localhost:6379`. This is relevant for both `src/data_creation/relation_data.py` and `src/inference.py`.

5.  **Environment Variables (for `generate_data.py`):**
    If using Azure OpenAI services with `src.data_creation.generate_data.py` (or the `scripts/data_create.sh` script), you'll need to set environment variables or pass them as arguments within the script:
    *   `AZURE_OPENAI_API_KEY`
    *   `AZURE_OPENAI_ENDPOINT`
    *   `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
    *   `AZURE_OPENAI_API_VERSION`
    For standard OpenAI, you might need:
    *   `OPENAI_API_KEY`
    *   `OPENAI_BASE_URL` (if using a proxy or a non-standard endpoint)

## Training

The training process involves two main stages: data creation and model training.

### Data Creation
This stage involves processing the knowledge graph, generating scored data using language models, and finally composing it into a training format. The primary method for this is the provided shell script.

**Using the `scripts/data_create.sh` script:**

The recommended way to perform data creation is by using the `scripts/data_create.sh` script. This script automates the sequential execution of the necessary Python scripts.

1.  **Configure the script**: Open `scripts/data_create.sh` in a text editor.
    *   Review and modify the variables in the "Configuration" section at the top of the script. This includes dataset names, paths, API keys for OpenAI/Azure, and other parameters.
    *   Carefully read the comments and uncomment the execution blocks for the steps you wish to run.
2.  **Make the script executable**:
    ```bash
    chmod +x scripts/data_create.sh
    ```
3.  **Run the script from the project root directory**:
    ```bash
    bash scripts/data_create.sh
    ```
    The script will guide you through the steps and indicate where outputs are saved.

**Manual Execution of Individual Scripts (Advanced):**

If you prefer to run each Python script individually or need more fine-grained control, you can use the commands below. These are the same commands orchestrated by the `data_create.sh` script. Ensure that the output path of one step correctly feeds into the input path of the next.

<details>
<summary>Click to expand for individual script commands</summary>

1.  **`src.data_creation.relation_data`**: Extracts relation paths between entities from a given knowledge graph dataset (e.g., WebQSP, CWQ) and saves the processed path information.
    ```bash
    python -m src.data_creation.relation_data \
      --dataset webqsp \
      --split train \
      --save_path ./output/chain_data/webqsp_train_chain_data.json \
      --n_proc 8 \
      --save
    ```

2.  **`src.data_creation.generate_data`**: Takes the output from `relation_data.py`, uses language models (OpenAI or Azure) to score aspects like relation relevance, entity relevance, path utility, and reasoning steps, then saves this augmented data.
    ```bash
    # Example using OpenAI
    python -m src.data_creation.generate_data \
      --chain_data ./output/chain_data/webqsp_train_chain_data.json \
      --model_name gpt-3.5-turbo \
      --task all \
      --output_file ./output/generate/webqsp_train_generated_data.json \
      --openai_api_key "YOUR_OPENAI_API_KEY" \
      --openai_base_url "YOUR_OPENAI_BASE_URL_IF_NEEDED" \
      --n 3 \
      --temperature 1.0

    # Example using Azure OpenAI
    # Ensure relevant Azure environment variables are set or passed as arguments.
    python -m src.data_creation.generate_data \
      --chain_data ./output/chain_data/webqsp_train_chain_data.json \
      --use_azure \
      --task all \
      --output_file ./output/generate/webqsp_train_generated_data_azure.json \
      --azure_api_key "YOUR_AZURE_API_KEY" \
      --azure_endpoint "YOUR_AZURE_ENDPOINT" \
      --azure_deployment "YOUR_AZURE_DEPLOYMENT_NAME" \
      --azure_api_version "YOUR_AZURE_API_VERSION" \
      --n 3 \
      --temperature 1.0
    ```

3.  **`src.data_creation.composite_data`**: Combines the scored and processed data from `generate_data.py` into a final format using special tokens, suitable for model training.
    ```bash
    python -m src.data_creation.composite_data \
      --input_file ./output/generate/webqsp_train_generated_data.json \
      --output_file ./output/final_training_data/webqsp_train_composed.json
    ```
</details>

### Model Training
Once the data creation steps are complete, you would typically proceed to train your language model using the generated datasets. This project uses VLLM for efficient model serving during inference, and training would likely involve fine-tuning a base model on the data produced by `composite_data.py`.

*(Note: Specific scripts or commands for model training are not detailed in this README. You would adapt your standard training procedures for language models to use the data generated by the previous steps.)*

## Inference

The inference process uses the trained model to answer questions based on the knowledge graph.

**Using the `scripts/inference.sh` script:**

The recommended way to run inference is by using the `scripts/inference.sh` script.

1.  **Configure the script**: Open `scripts/inference.sh` in a text editor.
    *   Set the `CUDA_VISIBLE_DEVICES` environment variable at the top of the script.
    *   Update `MODEL_NAME_OR_PATH`, `INPUT_FILE`, `OUTPUT_FILE`, and other parameters like `WORLD_SIZE` and `MAX_NEW_TOKENS` to match your setup.
    *   Configure the `ENABLE_CACHING` variable (set to `true` or `false`) to control whether embedding caching is used (it's enabled by default in the Python script if Redis is available).
2.  **Make the script executable**:
    ```bash
    chmod +x scripts/inference.sh
    ```
3.  **Run the script from the project root directory**:
    ```bash
    bash scripts/inference.sh
    ```
    The script will display the configuration and the command it will execute. You may need to uncomment the `eval $CMD` line within the script to actually run the inference.

**Understanding `src/inference.py` (called by `scripts/inference.sh`):**

The `scripts/inference.sh` script executes `src/inference.py`, which:
- Loads a pre-trained language model (compatible with VLLM) and its tokenizer.
- Utilizes a FAISS vector database with cached embeddings (e.g., BAAI/bge-large-en-v1.5) for efficient relation and entity retrieval.
- Iteratively builds a prediction tree, retrieving candidate relations and entities, and scoring them.
- Supports beam search by exploring multiple paths.
- Generates answers and logs detailed information.
- Calculates evaluation metrics (F1 score, Hits@1).

**Key Parameters (configurable within `scripts/inference.sh`):**

The shell script allows you to set these parameters, which are then passed to `src/inference.py`:
*   `CUDA_VISIBLE_DEVICES`: Specifies the GPU IDs.
*   `MODEL_NAME_OR_PATH`: Path to your trained model checkpoint.
*   `INPUT_FILE`: Path to the test data.
*   `OUTPUT_FILE`: Path to save results.
*   `WORLD_SIZE`: Number of GPUs for tensor parallelism.
*   `MAX_NEW_TOKENS`: Max new tokens for generation.
*   `ENABLE_CACHING`: Controls the `--cached_embedding` flag passed to the Python script. If `ENABLE_CACHING` is set to `false` in the shell script, the `--cached_embedding` flag is added to the Python command, which *disables* caching. Otherwise, caching is enabled by default in the Python script.

**Interactive Inference with `inference.ipynb`:**

For a more interactive approach to running inference, debugging steps, or visualizing the prediction tree, you can use the `inference.ipynb` Jupyter notebook. It allows for cell-by-cell execution of the inference logic, providing a closer look at intermediate outputs and the model's behavior. This can be particularly useful for development and detailed analysis.