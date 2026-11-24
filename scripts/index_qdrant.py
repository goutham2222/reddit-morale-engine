"""CLI script to batch index enriched records into persistent local Qdrant."""

import argparse
from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.rag.qdrant_indexer import QdrantIndexer

logger = get_logger("index_qdrant")


def index_enriched_dataset(recreate: bool = False, limit: int = None):
    """Loads enriched_records.parquet and indexes unique records into persistent Qdrant."""
    config = EngineConfig.load()
    storage = StorageManager(config)

    logger.info("Loading enriched records for Qdrant indexing...")
    df = storage.load_parquet("enriched/enriched_records.parquet")

    if limit:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit:,} records.")

    indexer = QdrantIndexer(config=config)
    if recreate:
        indexer.init_collection(recreate=True)

    count = indexer.index_records(df)
    logger.info(f"Successfully indexed {count:,} records into Qdrant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index enriched records into Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection from scratch")
    parser.add_argument("--limit", type=int, default=None, help="Max records to index")
    args = parser.parse_args()

    index_enriched_dataset(recreate=args.recreate, limit=args.limit)
