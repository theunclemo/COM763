import ast
import sys
import logging
import pandas as pd

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def build_sequences(
    df: pd.DataFrame,
    label_map: dict[str, int]
) -> pd.DataFrame:
    """
    Group parsed log lines by block_id.
    Each block becomes one row containing:
        - block_id
        - ordered list of event templates
        - template sequence as a single string (for TF-IDF)
        - ground truth label (0=normal, 1=anomaly)
    Blocks with no label are dropped.
    """

    logger.info("Exploding block_ids to individual rows...")

    # block_ids column is stored as string representation of a list
    # we need to convert it back to actual lists
    df["block_ids"] = df["block_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # drop rows with no block_id
    df = df[df["block_ids"].apply(lambda x: len(x) > 0)].copy()
    logger.info(f"Rows with at least one block_id: {len(df):,}")

    # explode so each row has one block_id
    df = df.explode("block_ids").rename(columns={"block_ids": "block_id"})
    logger.info(f"Rows after explode: {len(df):,}")

    # group by block_id — collect ordered template sequence
    logger.info("Grouping by block_id to build sequences...")
    sequences = (
        df.groupby("block_id")["template"]
        .apply(list)
        .reset_index()
    )
    sequences.columns = ["block_id", "event_sequence"]

    # join template list into single string for TF-IDF
    sequences["sequence_str"] = sequences["event_sequence"].apply(
        lambda events: " ".join(events)
    )

    # attach ground truth labels
    sequences["label"] = sequences["block_id"].map(label_map)

    # drop blocks that have no label entry
    before = len(sequences)
    sequences = sequences.dropna(subset=["label"])
    sequences["label"] = sequences["label"].astype(int)
    after = len(sequences)

    logger.info(f"Blocks before label join : {before:,}")
    logger.info(f"Blocks after label join  : {after:,}")
    logger.info(f"Blocks dropped (no label): {before - after:,}")

    # summary stats
    normal    = (sequences["label"] == 0).sum()
    anomalous = (sequences["label"] == 1).sum()
    total     = len(sequences)

    logger.info(f"Total sequences  : {total:,}")
    logger.info(f"Normal           : {normal:,}  ({normal/total*100:.2f}%)")
    logger.info(f"Anomalous        : {anomalous:,}  ({anomalous/total*100:.2f}%)")

    return sequences


def save_sequences(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    logger.info(f"Sequences saved to: {output_path}")


if __name__ == "__main__":
    from src.ingestion import load_labels

    PARSED_PATH   = "data/processed/parsed_logs.csv"
    LABEL_PATH    = "data/raw/anomaly_label.csv"
    OUTPUT_PATH   = "data/processed/sequences.csv"

    logger.info("Loading parsed logs...")
    df = pd.read_csv(PARSED_PATH)
    logger.info(f"Loaded {len(df):,} parsed log rows")

    label_map = load_labels(LABEL_PATH)
    sequences = build_sequences(df, label_map)
    save_sequences(sequences, OUTPUT_PATH)

    print("\n--- Shape ---")
    print(sequences.shape)

    print("\n--- Sample row ---")
    row = sequences.iloc[0]
    print(f"block_id       : {row['block_id']}")
    print(f"label          : {row['label']}")
    print(f"num events     : {len(row['event_sequence'])}")
    print(f"sequence_str   : {row['sequence_str'][:120]}...")

    print("\n--- Label distribution ---")
    print(sequences["label"].value_counts())