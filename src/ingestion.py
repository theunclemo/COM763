import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MIN_FIELDS = 5

def load_raw_logs(log_path: str) -> list[str]:
    """
    Read the raw HDFS log file line by line.
    Skips empty lines and lines that do not meet the minimum
    field requirement.
    Returns a list of clean raw log strings.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at: {log_path}")

    raw_lines = []
    skipped = 0
    total = 0

    logger.info(f"Loading raw logs from: {log_path}")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            line = line.strip()

            # skip empty lines
            if not line:
                skipped += 1
                continue

            # skip lines that are too short to be valid log entries
            if len(line.split()) < MIN_FIELDS:
                skipped += 1
                continue

            raw_lines.append(line)

    logger.info(f"Total lines read     : {total}")
    logger.info(f"Valid lines kept     : {len(raw_lines)}")
    logger.info(f"Lines skipped        : {skipped}")

    return raw_lines


def load_labels(label_path: str) -> dict[str, int]:
    """
    Load the anomaly_label.csv file.
    Returns a dictionary mapping block_id to binary label.
    Normal  -> 0
    Anomaly -> 1
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")

    logger.info(f"Loading labels from: {label_path}")

    df = pd.read_csv(label_path)

    # normalise column names
    df.columns = [c.strip() for c in df.columns]

    label_map = {
        row["BlockId"]: 1 if row["Label"].strip() == "Anomaly" else 0
        for _, row in df.iterrows()
    }

    total_blocks = len(label_map)
    anomalous = sum(label_map.values())
    normal = total_blocks - anomalous

    logger.info(f"Total blocks         : {total_blocks}")
    logger.info(f"Normal blocks        : {normal}")
    logger.info(f"Anomalous blocks     : {anomalous}")
    logger.info(
        f"Anomaly rate         : {anomalous / total_blocks * 100:.2f}%"
    )

    return label_map


if __name__ == "__main__":
    LOG_PATH   = "data/raw/HDFS.log"
    LABEL_PATH = "data/raw/anomaly_label.csv"

    raw_lines = load_raw_logs(LOG_PATH)
    labels    = load_labels(LABEL_PATH)

    print("\n--- First 3 raw log lines ---")
    for line in raw_lines[:3]:
        print(line)

    print("\n--- First 3 label entries ---")
    for block_id, label in list(labels.items())[:3]:
        print(f"{block_id} -> {label}")