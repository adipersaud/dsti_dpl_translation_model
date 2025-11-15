import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from typing import List, Dict
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import pandas as pd


# Page config FIRST
st.set_page_config(page_title="EN → FR Translator", page_icon=":congratulations:", layout="centered")

# Custom Light Blue Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #e6f0fa;
        color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        background-color: #d4e4f7;
        color: #0f172a;
    }
    h1, h2, h3, h4 {
        color: #1e3a8a;
    }
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
        transform: scale(1.03);
    }
    textarea, input, .stTextInput > div > div > input {
        background-color: #f0f7ff !important;
        border-radius: 8px;
        border: 1px solid #93c5fd;
    }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #3b82f6;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
        border-top: 1px solid #1e3a8a;
    }
    </style>
    <div class="footer-text">
        © 2025 Aditya Persaud | EN–FR Translation App
    </div>
""", unsafe_allow_html=True)


# Sidebar: Model selection
st.sidebar.title("Model Selection")

model_descriptions = {
    "Fine-tuned (Full dataset, 1.2M rows)": "Trained on the full 1.2M English–French dataset for maximum accuracy and coverage.",
    "Fine-tuned (Distilled dataset)": "Trained on a distilled subset (~200k pairs) optimized for faster inference with minimal quality loss and SacreBLEU alignment",
    "Fine-tuned (Distilled dataset comet)": "Trained on a distilled subset (~200k pairs) optimized for faster inference with minimal quality loss and COMET alignment.",
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
use_base_fallback = st.sidebar.checkbox(f"Fallback to base ({base_checkpoint}) if loading fails", value=True)

# Sidebar: Metric selection
st.sidebar.markdown("---")
metric_options = st.sidebar.multiselect(
    "Select metrics for evaluation",
    ["BLEU", "SacreBLEU", "METEOR", "BERTScore", "chrF", "COMET"],
    default=["SacreBLEU", "chrF"]
)

# Load model & tokenizer
@st.cache_resource(show_spinner=True)
def load_model_tokenizer(primary_path: str, base_ckpt: str, allow_fallback: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(primary_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(primary_path).to(device)
        return tokenizer, model, device, primary_path
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(f"Failed to load model from '{primary_path}'. {e}")
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_ckpt).to(device)
        return tokenizer, model, device, base_ckpt

tokenizer, model, device, loaded_from = load_model_tokenizer(model_path, base_checkpoint, use_base_fallback)
st.info(f"Loaded: **{loaded_from}** ({'fine-tuned' if loaded_from == model_path else 'base'})")


# Load selected metrics
metric_loaders = {}

def safe_load_metric(name, *args, **kwargs):
    try:
        return evaluate.load(name, *args, **kwargs)
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

# COMET integration
if "COMET" in metric_options:
    try:
        from comet import download_model, load_from_checkpoint

        @st.cache_resource(show_spinner=True)
        def load_comet_model():
            model_path = download_model("Unbabel/wmt22-comet-da")
            return load_from_checkpoint(model_path)

        comet_model = load_comet_model()
        st.sidebar.success("COMET model loaded successfully")
    except Exception as e:
        st.warning(f"Failed to load COMET: {e}")
        comet_model = None
else:
    comet_model = None


# Metric computation
def compute_selected_metrics(preds: List[str], refs: List[str], srcs: List[str] = None) -> Dict[str, float]:
    refs_wrapped = [[r] for r in refs]
    out = {}
    for name, metric in metric_loaders.items():
        try:
            if name == "SacreBLEU":
                out[name] = metric.compute(predictions=preds, references=refs_wrapped)["score"]
            elif name == "BLEU":
                out[name] = metric.compute(predictions=preds, references=refs)["bleu"]
            elif name == "METEOR":
                out[name] = metric.compute(predictions=preds, references=refs)["meteor"]
            elif name == "BERTScore":
                bs = metric.compute(predictions=preds, references=refs, lang="fr")
                out["BERTScore_F1"] = float(sum(bs["f1"]) / len(bs["f1"])) if bs["f1"] else 0.0
            elif name == "chrF":
                out["chrF"] = metric.compute(predictions=preds, references=refs_wrapped)["score"]
        except Exception as e:
            out[name] = f"Error: {e}"

    if comet_model is not None and srcs is not None:
        try:
            data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
            result = comet_model.predict(data, batch_size=8, gpus=0)
            sys_score = result["system_score"]
            out["COMET"] = float(sys_score)
        except Exception as e:
            out["COMET"] = f"Error: {e}"

    return out


# Translation function
def translate_batch(sentences_en: List[str]) -> List[str]:
    if not sentences_en:
        return []
    inputs = tokenizer(sentences_en, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in outputs]


# UI: Single sentence
st.title("English → French Translation")
st.caption("Choose between fine-tuned models and evaluate using selected metrics (BLEU, SacreBLEU, chrF, COMET, etc.).")

st.subheader("Try a single sentence or paragraph")
src = st.text_area("English text", height=120)
ref = st.text_input("Optional French reference (for metrics)", "")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Translate")
with col2:
    score_btn = st.button("Translate + Score")

if run_btn or score_btn:
    if not src.strip():
        st.warning("Please enter an English sentence.")
    else:
        pred = translate_batch([src])[0]
        st.markdown("**Translation:**")
        st.success(pred)

        if score_btn and ref.strip():
            scores = compute_selected_metrics([pred], [ref], [src])
            st.markdown("**Metrics:**")
            st.json(scores)
        elif score_btn and not ref.strip():
            st.info("No reference provided — metrics skipped.")


# Batch mode
st.subheader("Batch evaluation from file")
uploaded = st.file_uploader(
    "Upload CSV/TSV/JSON with columns `en` (source) and `fr` (reference)",
    type=["csv", "tsv", "json"],
)

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif uploaded.name.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t")
    else:
        df = pd.read_json(uploaded)

    if not {"en", "fr"}.issubset(df.columns):
        st.error("Missing required columns: 'en' and 'fr'")
    else:
        st.write(f"Loaded {len(df)} rows.")
        batch_size = st.number_input("Batch size", 1, 128, 32)
        if st.button("Run batch translate + evaluate"):
            preds = []
            texts = df["en"].astype(str).tolist()
            refs = df["fr"].astype(str).tolist()

            for i in range(0, len(texts), batch_size):
                preds.extend(translate_batch(texts[i:i + batch_size]))

            df["prediction_fr"] = preds
            st.dataframe(df.head(10))
            scores = compute_selected_metrics(preds, refs, texts)
            st.markdown("### Aggregate metrics")
            st.json(scores)
            st.download_button(
                "Download results CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_en_fr.csv",
                mime="text/csv"
            )


# Notes
with st.expander("Notes"):
    st.markdown(
        "- You can switch between fine-tuned models in the sidebar.\n"
        "- Choose evaluation metrics such as BLEU, METEOR, chrF, BERTScore, or COMET.\n"
        "- COMET provides semantic evaluation aligned with human judgment.\n"
        "- chrF captures character-level morphological accuracy.\n"
        "- BERTScore assesses contextual and semantic similarity."
    )
