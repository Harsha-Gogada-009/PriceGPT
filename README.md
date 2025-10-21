# PriceGPT – LLM-based Product Price Prediction

PriceGPT is a **large language model pipeline** for predicting product prices from catalog content. It leverages **Llama 3-8B-Instruct** fine-tuned using **LoRA (Low-Rank Adaptation)** for efficient training and inference. The project demonstrates a full **data preprocessing, model training, and batch inference workflow** for numeric prediction tasks using LLMs.

---

## Features

- **Fine-tuned Llama 3-8B-Instruct** for product price prediction from catalog content.
- **Parameter-efficient LoRA adaptation** with ~4.6M trainable parameters.
- **Data cleaning & preprocessing**: removes bullet points, truncates content, handles missing values.
- **JSONL conversion** for instruction-tuning of numeric regression tasks.
- **Batch-wise inference** with GPU acceleration for large datasets (>75k rows).
- **Automatic numeric extraction** from model outputs with formatted 2-decimal predictions.

---

## Project Structure
.
├── finetuning.ipynb # Notebook for LoRA fine-tuning.
├── testing.ipynb # Notebook for batch inference.
├── train_final.csv # Original training CSV (catalog + price).
├── testCleaned.csv # Cleaned test data for inference.
├── train_llama2_numeric.jsonl # Training JSONL for LLM.
├── validation_llama2_numeric.jsonl.
├── test_llama2_numeric.jsonl.
├── llama3_price_model # Saved LoRA adapter (checkpoint).
└── predictions_25k.csv # Output predictions from test set.

All other files are uploaded on drive 
Link:-
