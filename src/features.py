import sys
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def build_tfidf_features(
    sequences: pd.DataFrame,
    max_features: int = 100
) -> tuple:
    """
    Convert sequence_str column into a TF-IDF feature matrix.
    Each row = one block sequence.
    Each column = one TF-IDF weighted template token.
    Returns:
        X      : sparse feature matrix
        y      : label array
        vectorizer : fitted TfidfVectorizer (saved for app use)
    """

    logger.info(f"Building TF-IDF features with max_features={max_features}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=False,      # templates are already normalised
        token_pattern=r"\S+", # treat each whitespace-separated token as a term
    )

    X = vectorizer.fit_transform(sequences["sequence_str"])
    y = sequences["label"].values

    logger.info(f"Feature matrix shape : {X.shape}")
    logger.info(f"Label array shape    : {y.shape}")
    logger.info(f"Vocabulary size      : {len(vectorizer.vocabulary_)}")

    # show top 20 terms by document frequency
    feature_names = vectorizer.get_feature_names_out()
    logger.info(f"Sample features      : {list(feature_names[:20])}")

    return X, y, vectorizer


def save_features(
    X,
    y: np.ndarray,
    vectorizer,
    matrix_path: str,
    labels_path: str,
    vectorizer_path: str
) -> None:
    save_npz(matrix_path, X)
    np.save(labels_path, y)
    joblib.dump(vectorizer, vectorizer_path)

    logger.info(f"Feature matrix saved : {matrix_path}")
    logger.info(f"Labels saved         : {labels_path}")
    logger.info(f"Vectorizer saved     : {vectorizer_path}")


if __name__ == "__main__":

    SEQUENCES_PATH   = "data/processed/sequences.csv"
    MATRIX_PATH      = "data/processed/tfidf_matrix.npz"
    LABELS_PATH      = "data/processed/labels.npy"
    VECTORIZER_PATH  = "data/processed/tfidf_vectorizer.joblib"

    logger.info("Loading sequences...")
    sequences = pd.read_csv(SEQUENCES_PATH)
    logger.info(f"Loaded {len(sequences):,} sequences")

    X, y, vectorizer = build_tfidf_features(sequences, max_features=100)
    save_features(X, y, vectorizer, MATRIX_PATH, LABELS_PATH, VECTORIZER_PATH)

    print("\n--- Feature matrix ---")
    print(f"Shape  : {X.shape}")
    print(f"Density: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%")

    print("\n--- Label distribution ---")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label} : {count:,}")