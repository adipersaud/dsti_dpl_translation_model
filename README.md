# English → French Translation App

An interactive **Streamlit** web application for **English–French translation**, fine-tuned on the OPUS corpus (Full and Distilled) datasets using **Transformer-based encoder–decoder models** from [Hugging Face Transformers](https://huggingface.co/transformers).

The app allows users to **translate sentences or batch files** and **evaluate translation quality** using multiple metrics such as **BLEU**, **SacreBLEU**, **METEOR**, **chrF** and **BERTScore**.

---

## Features

### Model Selection
Choose between three fine-tuned translation models:
- **Full dataset (1.2M pairs)** — highest lexical accuracy.  
- **Distilled dataset (0.2M pairs)** — faster inference with minimal quality loss.  
- **Distilled dataset (COMET metric)** — optimized for human alignment.

---

### Metric Evaluation
Supports automatic computation of:
- **BLEU** — n-gram precision metric.  
- **SacreBLEU** — standardized BLEU for reproducibility.  
- **METEOR** — includes stemming, synonym, and order matching.  
- **chrF** — character-level metric for morphology.  
- **BERTScore** — contextual semantic similarity.  

---

### Interactive UI
- Single-sentence or paragraph translation.  
- Batch translation from CSV/TSV/JSON files.  
- Download results with model predictions and metrics.  
 

---

## Python Environment

- **Python version:** 3.10.4  

---

## Installation Instructions

### 1️ Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate       # (Linux/Mac)
.\.venv\Scripts\activate        # (Windows)

---
## Install requirements
---
pip install -r requirements.txt

## Run streamlit application

streamlit run app.py 

## Good to go!
# dsti_dpl_translation_model
Steamlit Application of my Translation model - English to French
