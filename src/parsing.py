import re
import sys
import logging
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

LOG_PATTERN = re.compile(
    r"^(\d{6})\s+"
    r"(\d{6})\s+"
    r"(\d+)\s+"
    r"(\w+)\s+"
    r"([\w.$]+):\s+"
    r"(.+)$"
)

BLOCK_PATTERN = re.compile(r"(blk_-?\d+)")

# pre-cleaning patterns — applied before Drain3
RE_BLOCK  = re.compile(r"blk_-?\d+")
RE_IP     = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?")
RE_NUMBER = re.compile(r"\b\d+\b")


def clean_content(content: str) -> str:
    """
    Replace dynamic tokens before Drain3 sees the content.
    Reduces variation so Drain3 clusters more aggressively
    without needing a high similarity threshold.
    """
    content = RE_BLOCK.sub("<BLOCK>", content)
    content = RE_IP.sub("<IP>", content)
    content = RE_NUMBER.sub("<NUM>", content)
    return content


def build_drain3_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_sim_th      = 0.5
    config.drain_depth       = 4
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    return miner


def parse_logs(raw_lines: list[str]) -> pd.DataFrame:
    miner     = build_drain3_miner()
    records   = []
    malformed = 0

    logger.info(f"Parsing {len(raw_lines):,} log lines with Drain3...")

    for i, line in enumerate(raw_lines):
        match = LOG_PATTERN.match(line)

        if not match:
            malformed += 1
            continue

        date, time, pid, level, component, content = match.groups()
        block_ids = BLOCK_PATTERN.findall(content)

        # pre-clean then feed to Drain3
        cleaned = clean_content(content)
        miner.add_log_message(cleaned)

        records.append({
            "date"      : date,
            "time"      : time,
            "pid"       : int(pid),
            "level"     : level,
            "component" : component,
            "content"   : content,
            "block_ids" : block_ids,
        })

        if (i + 1) % 1_000_000 == 0:
            logger.info(f"  Processed {i + 1:,} lines...")

    logger.info(f"Parsing complete")
    logger.info(f"Records parsed       : {len(records):,}")
    logger.info(f"Malformed lines      : {malformed:,}")
    logger.info(f"Unique templates     : {len(miner.drain.id_to_cluster):,}")

    df = pd.DataFrame(records)

    # second pass — remap using final cluster state
    logger.info("Remapping templates using final cluster state...")
    final_templates = []
    for content in df["content"]:
        cleaned = clean_content(content)
        result  = miner.match(cleaned)
        if result:
            final_templates.append(result.get_template())
        else:
            final_templates.append(cleaned)

    df["template"] = final_templates
    logger.info("Remapping complete")

    return df


def save_parsed(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    logger.info(f"Parsed logs saved to : {output_path}")


if __name__ == "__main__":
    from src.ingestion import load_raw_logs

    LOG_PATH    = "data/raw/HDFS.log"
    OUTPUT_PATH = "data/processed/parsed_logs.csv"

    raw_lines = load_raw_logs(LOG_PATH)
    df        = parse_logs(raw_lines)
    save_parsed(df, OUTPUT_PATH)

    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Sample row ---")
    print(df.iloc[0])

    print("\n--- Unique templates ---")
    templates = df["template"].unique()
    print(f"Total unique templates: {len(templates)}")
    for t in templates[:15]:
        print(t)