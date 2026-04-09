import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> dict:
    """
    Compute precision, recall, F1 for a given model's predictions.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"\n--- {model_name} ---")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])}")

    return {
        "model"     : model_name,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")


def plot_score_distribution(
    scores_if: np.ndarray,
    scores_svm: np.ndarray,
    y_true: np.ndarray,
    output_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores, title in zip(
        axes,
        [scores_if, scores_svm],
        ["Isolation Forest", "One-Class SVM"]
    ):
        ax.hist(
            scores[y_true == 0],
            bins=100,
            alpha=0.6,
            label="Normal",
            color="steelblue"
        )
        ax.hist(
            scores[y_true == 1],
            bins=100,
            alpha=0.6,
            label="Anomaly",
            color="crimson"
        )
        ax.set_title(f"Anomaly Score Distribution — {title}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Score distribution plot saved: {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    scores_if: np.ndarray,
    scores_svm: np.ndarray,
    output_path: str
) -> None:
    """
    ROC curves using anomaly scores.
    For Isolation Forest: lower score = more anomalous, so we negate.
    For One-Class SVM: lower score = more anomalous, so we negate.
    """
    fpr_if,  tpr_if,  _ = roc_curve(y_true, -scores_if)
    fpr_svm, tpr_svm, _ = roc_curve(y_true, -scores_svm)

    auc_if  = auc(fpr_if,  tpr_if)
    auc_svm = auc(fpr_svm, tpr_svm)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_if,  tpr_if,  label=f"Isolation Forest (AUC = {auc_if:.4f})",  color="steelblue")
    plt.plot(fpr_svm, tpr_svm, label=f"One-Class SVM   (AUC = {auc_svm:.4f})",  color="crimson")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {output_path}")

    logger.info(f"AUC — Isolation Forest : {auc_if:.4f}")
    logger.info(f"AUC — One-Class SVM    : {auc_svm:.4f}")

    return auc_if, auc_svm


def plot_metrics_comparison(
    results: list[dict],
    output_path: str
) -> None:
    df = pd.DataFrame(results)
    df_melted = df.melt(
        id_vars="model",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melted, x="metric", y="score", hue="model")
    plt.title("Model Comparison — Precision, Recall, F1")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Metrics comparison plot saved: {output_path}")


if __name__ == "__main__":

    PREDS_PATH      = "data/processed/predictions.csv"
    CM_IF_PATH      = "outputs/plots/confusion_matrix_if.png"
    CM_SVM_PATH     = "outputs/plots/confusion_matrix_svm.png"
    SCORE_DIST_PATH = "outputs/plots/score_distributions.png"
    ROC_PATH        = "outputs/plots/roc_curves.png"
    METRICS_PATH    = "outputs/plots/metrics_comparison.png"
    RESULTS_PATH    = "outputs/results/evaluation_results.csv"

    logger.info("Loading predictions...")
    df = pd.read_csv(PREDS_PATH)

    y_true    = df["label"].values
    if_preds  = df["if_pred"].values
    svm_preds = df["svm_pred"].values
    if_scores  = df["if_score"].values
    svm_scores = df["svm_score"].values

    # evaluate both models
    if_results  = evaluate_model(y_true, if_preds,  "Isolation Forest")
    svm_results = evaluate_model(y_true, svm_preds, "One-Class SVM")

    results = [if_results, svm_results]

    # save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"\nResults table saved: {RESULTS_PATH}")

    # plots
    plot_confusion_matrix(y_true, if_preds,  "Isolation Forest", CM_IF_PATH)
    plot_confusion_matrix(y_true, svm_preds, "One-Class SVM",    CM_SVM_PATH)
    plot_score_distribution(if_scores, svm_scores, y_true, SCORE_DIST_PATH)
    plot_roc_curves(y_true, if_scores, svm_scores, ROC_PATH)
    plot_metrics_comparison(results, METRICS_PATH)

    print("\n--- Final Results Table ---")
    print(results_df.to_string(index=False))