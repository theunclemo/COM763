import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from scipy.sparse import issparse, diags
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.append(".")

# ── regex patterns ────────────────────────────────────────────────────────────
LOG_PATTERN = re.compile(
    r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w.$]+):\s+(.+)$"
)
BLOCK_PATTERN = re.compile(r"(blk_-?\d+)")
RE_BLOCK      = re.compile(r"blk_-?\d+")
RE_IP         = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?")
RE_NUMBER     = re.compile(r"\b\d+\b")


def clean_content(content: str) -> str:
    content = RE_BLOCK.sub("<BLOCK>", content)
    content = RE_IP.sub("<IP>", content)
    content = RE_NUMBER.sub("<NUM>", content)
    return content


@st.cache_resource
def load_model_and_vectorizer():
    model      = joblib.load("data/processed/isolation_forest.joblib")
    vectorizer = joblib.load("data/processed/tfidf_vectorizer.joblib")
    return model, vectorizer


@st.cache_resource
def load_drain3_miner():
    config = TemplateMinerConfig()
    config.drain_sim_th      = 0.5
    config.drain_depth       = 4
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_uploaded_logs(raw_lines: list[str], miner: TemplateMiner) -> pd.DataFrame:
    records = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        match = LOG_PATTERN.match(line)
        if not match:
            continue
        date, time, pid, level, component, content = match.groups()
        block_ids = BLOCK_PATTERN.findall(content)
        cleaned   = clean_content(content)
        result    = miner.add_log_message(cleaned)
        template  = result["template_mined"]
        records.append({
            "date"      : date,
            "time"      : time,
            "level"     : level,
            "component" : component,
            "content"   : content,
            "block_ids" : block_ids,
            "template"  : template
        })
    return pd.DataFrame(records)


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["block_ids"].apply(lambda x: len(x) > 0)].copy()
    df = df.explode("block_ids").rename(columns={"block_ids": "block_id"})
    sequences = (
        df.groupby("block_id")["template"]
        .apply(list)
        .reset_index()
    )
    sequences.columns         = ["block_id", "event_sequence"]
    sequences["sequence_str"] = sequences["event_sequence"].apply(
        lambda e: " ".join(e)
    )
    return sequences


def predict_sequences(sequences: pd.DataFrame, model, vectorizer):
    X = vectorizer.transform(sequences["sequence_str"])

    # apply MaxAbsScaler manually to avoid sklearn version
    # mismatch on the saved scaler object
    if issparse(X):
        max_abs = np.asarray(np.abs(X).max(axis=0).todense()).flatten()
        max_abs[max_abs == 0] = 1
        X_scaled = X.dot(diags(1.0 / max_abs))
    else:
        max_abs = np.abs(X).max(axis=0)
        max_abs[max_abs == 0] = 1
        X_scaled = X / max_abs

    raw_scores  = model.decision_function(X_scaled)
    min_s       = raw_scores.min()
    max_s       = raw_scores.max()
    risk_scores = 100 * (1 - (raw_scores - min_s) / (max_s - min_s))

    raw_preds    = model.predict(X_scaled)
    preds_binary = np.where(raw_preds == -1, 1, 0)

    sequences               = sequences.copy()
    sequences["risk_score"] = np.round(risk_scores, 1)
    sequences["prediction"] = preds_binary
    sequences["status"]     = np.where(
        preds_binary == 1, "🔴 Anomaly", "🟢 Normal"
    )
    return sequences


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Log Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Intelligent Log Anomaly Detection System")
st.markdown(
    "Upload a raw HDFS log file. The system will parse it, extract event "
    "sequences and flag anomalous block behaviours using a trained "
    "Isolation Forest model."
)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Isolation Forest  
        **Features:** TF-IDF on Drain3 event templates  
        **Dataset:** HDFS v1 (Loghub)  
        **F1 Score:** 0.69  
        **AUC:** 0.99  
        """
    )
    st.divider()
    st.header("⚙️ Settings")
    show_normal = st.checkbox("Show normal sequences", value=False)

# ── Load model ────────────────────────────────────────────────────────────────
model, vectorizer = load_model_and_vectorizer()
miner             = load_drain3_miner()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload raw HDFS log file (.log or .txt)",
    type=["log", "txt"]
)

if uploaded_file is not None:
    with st.spinner("Reading log file..."):
        raw_content = uploaded_file.read().decode("utf-8", errors="replace")
        raw_lines   = raw_content.splitlines()

    st.info(f"📄 {len(raw_lines):,} lines loaded")

    with st.spinner("Parsing logs and extracting templates..."):
        df = parse_uploaded_logs(raw_lines, miner)

    if df.empty:
        st.error("No valid log lines could be parsed. Check the file format.")
        st.stop()

    with st.spinner("Building block sequences..."):
        sequences = build_sequences(df)

    with st.spinner("Running anomaly detection..."):
        results = predict_sequences(sequences, model, vectorizer)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Detection Summary")

    total     = len(results)
    anomalies = (results["prediction"] == 1).sum()
    normal    = total - anomalies

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sequences",   f"{total:,}")
    col2.metric("Anomalies Flagged", f"{anomalies:,}")
    col3.metric("Normal Sequences",  f"{normal:,}")

    # ── Score distribution plot ───────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Risk Score Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        results.loc[results["prediction"] == 0, "risk_score"],
        bins=60,
        alpha=0.6,
        label="Normal",
        color="steelblue"
    )
    ax.hist(
        results.loc[results["prediction"] == 1, "risk_score"],
        bins=60,
        alpha=0.6,
        label="Anomaly",
        color="crimson"
    )
    ax.set_xlabel("Risk Score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution by Class")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # ── Results table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Sequence Results")

    display_df = results[["block_id", "status", "risk_score"]].copy()

    if not show_normal:
        display_df = display_df[results["prediction"] == 1]
        st.caption(
            f"Showing {len(display_df):,} anomalous sequences only. "
            "Toggle in sidebar to show all."
        )
    else:
        st.caption(f"Showing all {len(display_df):,} sequences.")

    st.dataframe(
        display_df.sort_values(
            "risk_score", ascending=False
        ).reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    csv = results[
        ["block_id", "status", "risk_score", "sequence_str"]
    ].to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a log file above to get started.")
    st.markdown(
        """
        **Expected format:** Raw HDFS log lines e.g.
        ```
        081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
        ```
        """
    )
