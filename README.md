# Theis LLM Project

This project demonstrates a simple language model using the GPT architecture implemented with PyTorch. It includes functionality for training, evaluating, and generating text based on the transformer-based model.

## Features
- Self-Attention and Multi-Head Attention implementation
- Customizable training parameters through command line arguments
- Parallel processing for text extraction and vocabulary update
- Text generation using the trained model

## Requirements

To run this project, you will need:
- Python 3.8+
- PyTorch
- tqdm
- lzma
- concurrent.futures

## Installation

Clone this repository to your local machine:
```bash
git clone [repository-url]
```

Install the required Python packages:
```bash
pip install torch tqdm
```

## Usage

1. **Setting up the Dataset:**
   - Place your `.xz` compressed text files in the specified data directory.
   - Modify the `data_directory` variable in the script to point to your directory of `.xz` files.

2. **Training the Model:**
   - Run the training script with the required batch size:
   ```bash
   python model_script.py -batch_size [your-batch-size]
   ```

3. **Generating Text:**
   - Use the interactive prompt to generate text:
   ```bash
   python model_script.py -batch_size [your-batch-size]
   ```

## Configuration

Edit the parameters in the script to customize the model and training process:
- `batch_size`
- `block_size`
- `max_steps`
- `learning_rate`
- `num_heads`
- `num_layers`# learn_llm
