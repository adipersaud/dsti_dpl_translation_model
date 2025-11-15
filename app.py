import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from typing import List, Dict
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import pandas as pd
from huggingface_hub import InferenceClient

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

# Page config
st.set_page_config(page_title="EN → FR Translator", page_icon=":congratulations:", layout="centered")

# Custom Light Blue Theme
st.markdown("""
    <style>
    .stApp { background-color: #e6f0fa; color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #d4e4f7; color: #0f172a; }
    h1, h2, h3, h4 { color: #1e3a8a; }
    div.stButton > button:first-child {
        background-color: #3b82f6; color: white; border-radius: 10px;
        border: none; font-weight: 600; transition: all 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb; transform: scale(1.03);
    }
    textarea, input, .stTextInput > div > div > input {
        background-color: #f0f7ff !important; border-radius: 8px;
        border: 1px solid #93c5fd;
    }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed; bottom: 0; width: 100%;
        background-color: #3b82f6; color: white; text-align: center;
        padding: 8px; font-size: 14px; border-top: 1px solid #1e3a8a;
    }
    </style>
    <div class="footer-text">© 2025 Aditya Persaud | EN–FR Translation App</div>
""", unsafe_allow_html=True)

# Sidebar: model selection
st.sidebar.title("Model Selection")

model_descriptions = {
    "Fine-tuned (Full dataset, 1.2M rows)": "Trained on the full 1.2M English–French dataset.",
    "Fine-tuned (Distilled dataset)": "Distilled subset (~200k pairs) optimized for speed.",
    "Fine-tuned (Distilled dataset comet)": "Distilled dataset optimized for COMET alignment."
}

model_options = {
    "Fine-tuned (Full dataset, 1.2M rows)": "apersaud/opus-mt-en-fr-finetuned-en-to-fr_multi-metric",
    "Fine-tuned (Distilled dataset)": "apersaud/opus-mt-en-fr-finetuned-en-to-fr_multi-metric-distilled_dataset",
    "Fine-tuned (Distilled dataset comet)": "apersaud/opus-mt-en-fr-finetuned-en-to-fr_multi-metric-distilled_dataset_comet",
}

model_choice = st.sidebar.selectbox("Choose translation model", list(model_options.keys()))
model_path = model_options[model_choice]
st.sidebar.caption(model_descriptions[model_choice])

base_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
use_base_fallback = st.sidebar.checkbox(f"Fallback to base model ({base_checkpoint})", value=True)

# Metrics selection
st.sidebar.markdown("---")
metric_options = st.sidebar.multiselect(
    "Select metrics",
    ["BLEU", "SacreBLEU", "METEOR", "BERTScore", "chrF", "COMET"],
    default=["SacreBLEU", "chrF"]
)

# Load translation model
@st.cache_resource(show_spinner=True)
def load_model_tokenizer(primary_path, fallback_path, allow_fallback):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(primary_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(primary_path).to(device)
        return tokenizer, model, device, primary_path
    except Exception:
        if not allow_fallback:
            raise
        tokenizer = AutoTokenizer.from_pretrained(fallback_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_path).to(device)
        return tokenizer, model, device, fallback_path

tokenizer, model, device, loaded_from = load_model_tokenizer(model_path, base_checkpoint, use_base_fallback)
st.info(f"Loaded model: **{loaded_from}**")

# Load metrics
metric_loaders = {}

def safe_load_metric(name):
    try:
        return evaluate.load(name)
    except Exception as e:
        st.warning(f"Failed to load {name}: {e}")
        return None

if "SacreBLEU" in metric_options:
    metric_loaders["SacreBLEU"] = safe_load_metric("sacrebleu")
if "BLEU" in metric_options:
    metric_loaders["BLEU"] = safe_load_metric("bleu")
if "METEOR" in metric_options:
    metric_loaders["METEOR"] = safe_load_metric("meteor")
if "BERTScore" in metric_options:
    metric_loaders["BERTScore"] = safe_load_metric("bertscore")
if "chrF" in metric_options:
    metric_loaders["chrF"] = safe_load_metric("chrf")

# COMET via HuggingFace API
if "COMET" in metric_options:
    try:
        @st.cache_resource
        def load_comet_api():
            token = st.secrets["HF_TOKEN"]
            return InferenceClient("Unbabel/COMET", token=token)
        comet_api = load_comet_api()
        st.sidebar.success("COMET API ready")
    except Exception as e:
        st.warning(f"COMET API error: {e}")
        comet_api = None
else:
    comet_api = None

# Compute Metrics
def compute_selected_metrics(preds, refs, srcs=None):
    refs_wrapped = [[r] for r in refs]
    out = {}

    for name, metric in metric_loaders.items():
        if metric is None:
            continue
        try:
            if name == "SacreBLEU":
                out[name] = metric.compute(predictions=preds, references=refs_wrapped)["score"]
            elif name == "BLEU":
                out[name] = metric.compute(predictions=preds, references=refs)["bleu"]
            elif name == "METEOR":
                out[name] = metric.compute(predictions=preds, references=refs)["meteor"]
            elif name == "BERTScore":
                b = metric.compute(predictions=preds, references=refs, lang="fr")
                out["BERTScore_F1"] = sum(b["f1"]) / len(b["f1"])
            elif name == "chrF":
                out["chrF"] = metric.compute(predictions=preds, references=refs_wrapped)["score"]
        except Exception as e:
            out[name] = f"Error: {e}"

    # COMET API
    if comet_api is not None and refs and srcs:
        try:
            response = comet_api.post_json({
                "src": srcs[0],
                "mt": preds[0],
                "ref": refs[0]
            })
            out["COMET"] = response.get("score", None)
        except Exception as e:
            out["COMET"] = f"Error: {e}"

# Translation function
def translate_batch(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

# UI
st.title("English → French Translation")

src = st.text_area("English text", height=120)
ref = st.text_input("Optional French reference")

col1, col2 = st.columns(2)
run_btn = col1.button("Translate")
score_btn = col2.button("Translate + Score")

if run_btn or score_btn:
    if not src.strip():
        st.warning("Enter text")
    else:
        pred = translate_batch([src])[0]
        st.success(pred)

        if score_btn and ref.strip():
            scores = compute_selected_metrics([pred], [ref], [src])
            st.json(scores)

# Batch mode
uploaded = st.file_uploader("Upload CSV/TSV/JSON with `en` and `fr`")

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif uploaded.name.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t")
    else:
        df = pd.read_json(uploaded)

    if not {"en", "fr"}.issubset(df.columns):
        st.error("Missing `en` / `fr` columns")
    else:
        batch_size = st.number_input("Batch size", 1, 128, 32)

        if st.button("Run batch translate + evaluate"):
            texts = df["en"].astype(str).tolist()
            refs = df["fr"].astype(str).tolist()
            preds = []

            for i in range(0, len(texts), batch_size):
                preds.extend(translate_batch(texts[i:i+batch_size]))

            df["prediction_fr"] = preds
            st.dataframe(df.head())

            scores = compute_selected_metrics(preds, refs, texts)
            st.json(scores)

            st.download_button(
                "Download CSV", df.to_csv(index=False).encode("utf-8"),
                "predictions_en_fr.csv", "text/csv"
            )

# Notes
with st.expander("Notes"):
    st.markdown("""
    - Evaluate with BLEU, METEOR, chrF, BERTScore, or COMET.
    - COMET is computed using the official Hugging Face API.
    """)
