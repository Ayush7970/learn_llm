
# 🤖 **Theis LLM Project**

A lightweight implementation of a transformer-based **Language Model (LLM)** using the **GPT architecture** with **PyTorch**. This project covers training, evaluation, and text generation, offering a customizable and extendable codebase for NLP enthusiasts and researchers.

---

## ✨ **Key Features**

🚀 **Transformer Model**
- Implements **self-attention** and **multi-head attention** mechanisms  
- Supports customizable architecture (number of heads, layers, block size)

⚡ **Efficient Training**
- Parallel text extraction & vocabulary update using `concurrent.futures`
- Progress tracking with `tqdm`
- Command-line configurable hyperparameters

📝 **Text Generation**
- Interactive prompt for generating text using the trained model
- Supports custom starting prompts and generation length

🛠 **Customization**
- Easily tweak model hyperparameters and dataset configurations in one place

---

## 📦 **Requirements**

To run this project, you'll need:

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- tqdm
- lzma (standard library)
- concurrent.futures (standard library)

💡 _Install dependencies via pip:_
```bash
pip install torch tqdm
```

---

## 🛠 **Installation**

1️⃣ **Clone the repository**
```bash
git clone [repository-url]
cd Theis-LLM-Project
```

2️⃣ **Install the required Python packages**
```bash
pip install torch tqdm
```

---

## ⚙ **Usage**

### 📂 **Prepare the Dataset**
- Place your `.xz` compressed text files into your dataset directory.
- Update the `data_directory` variable in `model_script.py` to point to this directory.

---

### 🔥 **Train the Model**
```bash
python model_script.py -batch_size [your-batch-size]
```
Example:
```bash
python model_script.py -batch_size 64
```

---

### 💬 **Generate Text**
```bash
python model_script.py -batch_size [your-batch-size]
```
✅ After training, use the interactive prompt to start generating text.

---

## 🖼 **Example Screenshot**

Here’s a sample of the interactive text generation interface:

![LLM Interactive Chat Example](./Tarrifarm.png)

---

## ⚡ **Configurable Parameters**

You can modify these directly in the script or via command-line:

| Parameter       | Description                              |
|-----------------|------------------------------------------|
| `batch_size`     | Number of samples per batch              |
| `block_size`     | Length of context window                 |
| `max_steps`      | Total number of training steps           |
| `learning_rate`  | Optimizer learning rate                  |
| `num_heads`      | Number of attention heads                |
| `num_layers`     | Number of transformer layers             |

---

## 🧠 **Future Enhancements**
🌱 Planned features to take this project further:

- Save & load trained models  
- Add positional encoding visualization  
- Implement beam search for better text generation  
- Multi-GPU training support  

---

## 📝 **License**

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 **Contributors**

Made with ❤️ by Theis LLM Team.

---

### ✅ **Tip:**  
Replace `[repository-url]` with your actual GitHub repository link!
