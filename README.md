# ArG: Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection

Official implementation of the paper "ArG: Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection". This project introduces ArG, a framework designed to enhance Knowledge Base Question Answering (KBQA) by actively learning how to retrieve relevant information and reason over it through a process of self-reflection. We warmly welcome discussions and collaborations in the field of KBQA!

![WordFlow](./ArG.png)

## Getting Started

Welcome to ArG! This project enables you to leverage active self-reflection for retrieving and reasoning on knowledge graphs.

1.  **Set up your environment**: Begin by preparing your Python environment and installing all necessary dependencies. Detailed instructions can be found in the **[Setup](#setup)** section.
2.  **Prepare your data**: The next step is to process your knowledge graph data and generate the specific formats required for training. This involves extracting relation paths, using language models to score them, and composing the final training data. Please refer to the **[Data Creation](#data-creation)** subsection under **[Training](#training)** for scripts and commands.
3.  **Train your model**: Once the data is created, you can proceed to train your model using the generated datasets. (Further details on the training script and its parameters would typically be here.)
4.  **Run inference**: After you have a trained model (either one you've trained or a pre-trained one compatible with this project's inference script), you can use it to answer questions. The **[Inference](#inference)** section provides details on how to run the inference scripts and interpret the outputs.

Following these steps will guide you through using the ArG project from initial setup to performing inference on your knowledge graph data.

## Setup

This project requires Python 3.10 or newer.


1.  **Clone the repository & install dependencies:**
    ```bash
    git clone https://github.com/rmanluo/ArG.git
    cd ArG
    ```
2.  **Redis Server (Optional for Caching):**
    The scripts can use `RedisStore` for caching embeddings, which can speed up repeated initializations. If you plan to use this feature:
    *   Ensure you have a Redis server running. You can install Redis via your system's package manager (e.g., `sudo apt-get install redis-server` on Debian/Ubuntu) or run it using Docker.
    *   The default Redis URL used in the scripts is `redis://localhost:6379`.

3.  **Environment Variables (for `generate_data.py`):**
    If using Azure OpenAI services with `src.data_creation.generate_data.py`, you'll need to set environment variables or pass them as arguments:
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
This stage involves processing the knowledge graph, generating scored data using language models, and finally composing it into a training format.

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
    # Ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, 
    # and AZURE_OPENAI_API_VERSION are set as environment variables or passed as arguments.
    
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


### Model Training
Once the data creation steps are complete, you would typically proceed to train your language model using the generated datasets. This project uses VLLM for efficient model serving during inference, and training would likely involve fine-tuning a base model on the data produced by `composite_data.py`.
You can refer to **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file)** for fine-tuning.

## Inference

The inference process uses the trained model to answer questions based on the knowledge graph. The primary script for this is `src/inference.py`.

**Example Command:**

The following command (similar to `start.sh`) shows how to run the inference script. By default, embedding caching is enabled if a Redis server is accessible.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.inference \
  --model_name /path/to/your/trained_model_checkpoint \
  --input_file ./data/merged/WebQSP_test.json \
  --output_file ./output/inference/webqsp_test_results.json \
  --world_size 2 \
  --max_new_tokens 100 
```

To disable embedding caching, add the `--cached_embedding` flag:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.inference \
  --model_name /path/to/your/trained_model_checkpoint \
  --input_file ./data/merged/WebQSP_test.json \
  --output_file ./output/inference/webqsp_test_results.json \
  --world_size 2 \
  --max_new_tokens 100 \
  --cached_embedding
```

**Interactive Inference with `inference.ipynb`:**

For a more interactive approach to running inference, debugging steps, or visualizing the prediction tree, you can use the `inference.ipynb` Jupyter notebook. It allows for cell-by-cell execution of the inference logic, providing a closer look at intermediate outputs and the model's behavior. This can be particularly useful for development and detailed analysis.
