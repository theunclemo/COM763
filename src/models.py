import sys
import logging
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MaxAbsScaler

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def train_isolation_forest(X) -> tuple:
    """
    Train Isolation Forest on the full feature matrix.
    contamination is set to match the known anomaly rate (2.93%)
    Returns model and raw anomaly scores.
    """
    logger.info("Training Isolation Forest...")
    logger.info(f"  contamination = 0.0293")
    logger.info(f"  n_estimators  = 100")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.0293,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    # decision_function returns anomaly score
    # more negative = more anomalous
    scores = model.decision_function(X)

    # predict returns 1 (normal) or -1 (anomaly)
    # convert to 0/1 to match our label format
    raw_preds = model.predict(X)
    preds = np.where(raw_preds == -1, 1, 0)

    anomalies_detected = preds.sum()
    logger.info(f"  Anomalies detected : {anomalies_detected:,}")

    return model, scores, preds


def train_one_class_svm(X) -> tuple:
    """
    Train One-Class SVM on the full feature matrix.
    nu parameter approximates the expected anomaly rate.
    Note: One-Class SVM does not scale well to very large datasets.
    We subsample for training but predict on all data.
    """
    logger.info("Training One-Class SVM...")
    logger.info(f"  nu     = 0.0293")
    logger.info(f"  kernel = rbf")

    # One-Class SVM is slow on 575k rows
    # subsample 50k rows for training — standard practice
    rng = np.random.RandomState(42)
    n_train = 50_000
    idx = rng.choice(X.shape[0], size=n_train, replace=False)
    X_train = X[idx]

    logger.info(f"  Training on {n_train:,} subsampled rows")

    model = OneClassSVM(
        kernel="rbf",
        nu=0.0293,
        gamma="scale"
    )

    # One-Class SVM requires dense input
    model.fit(X_train.toarray())

    logger.info("  Predicting on full dataset (this may take a while)...")
    scores = model.decision_function(X.toarray())
    raw_preds = model.predict(X.toarray())
    preds = np.where(raw_preds == -1, 1, 0)

    anomalies_detected = preds.sum()
    logger.info(f"  Anomalies detected : {anomalies_detected:,}")

    return model, scores, preds


def save_model(model, path: str) -> None:
    joblib.dump(model, path)
    logger.info(f"Model saved to: {path}")


if __name__ == "__main__":
    import pandas as pd

    MATRIX_PATH    = "data/processed/tfidf_matrix.npz"
    LABELS_PATH    = "data/processed/labels.npy"
    SEQUENCES_PATH = "data/processed/sequences.csv"

    IF_MODEL_PATH  = "data/processed/isolation_forest.joblib"
    SVM_MODEL_PATH = "data/processed/one_class_svm.joblib"
    SCALER_PATH    = "data/processed/scaler.joblib"

    PREDS_PATH     = "data/processed/predictions.csv"

    logger.info("Loading features and labels...")
    X = load_npz(MATRIX_PATH)
    y = np.load(LABELS_PATH)
    sequences = pd.read_csv(SEQUENCES_PATH)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Labels        : {y.shape}")

    # scale features — important for One-Class SVM
    logger.info("Scaling features with MaxAbsScaler...")
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaling complete")

    # train both models
    if_model,  if_scores,  if_preds  = train_isolation_forest(X_scaled)
    svm_model, svm_scores, svm_preds = train_one_class_svm(X_scaled)

    # save models
    save_model(if_model,  IF_MODEL_PATH)
    save_model(svm_model, SVM_MODEL_PATH)

    # save predictions for evaluation
    sequences["if_pred"]    = if_preds
    sequences["if_score"]   = if_scores
    sequences["svm_pred"]   = svm_preds
    sequences["svm_score"]  = svm_scores
    sequences["label"]      = y

    sequences.to_csv(PREDS_PATH, index=False)
    logger.info(f"Predictions saved to: {PREDS_PATH}")

    print("\n--- Isolation Forest ---")
    print(f"Anomalies detected : {if_preds.sum():,}")
    print(f"Score range        : {if_scores.min():.4f} to {if_scores.max():.4f}")

    print("\n--- One-Class SVM ---")
    print(f"Anomalies detected : {svm_preds.sum():,}")
    print(f"Score range        : {svm_scores.min():.4f} to {svm_scores.max():.4f}")