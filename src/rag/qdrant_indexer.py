"""Persistent local Qdrant vector indexer with multi-field payload indexing."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.common.azure_storage import StorageManager
from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.rag.embeddings import EmbeddingModel

logger = get_logger("qdrant_indexer")


class QdrantIndexer:
    """Indexes deduplicated, culture-relevant records into persistent local Qdrant storage."""

    COLLECTION_NAME = "reddit_culture_docs"

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        storage_path: Optional[Union[str, Path]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.config = config or EngineConfig.load()
        self.storage = StorageManager(self.config)
        self.embedder = embedding_model or EmbeddingModel()

        # Path to persistent local Qdrant database
        db_path = storage_path or (self.config.local_data_dir / "qdrant_db")
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Connecting to persistent local Qdrant at: {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))

        self.init_collection(recreate=False)

    def init_collection(self, recreate: bool = False):
        """Initializes the Qdrant collection and creates payload indices for fast metadata filtering."""
        collections = [c.name for c in self.client.get_collections().collections]

        if recreate and self.COLLECTION_NAME in collections:
            logger.info(f"Recreating collection: {self.COLLECTION_NAME}")
            self.client.delete_collection(self.COLLECTION_NAME)
            collections.remove(self.COLLECTION_NAME)

        if self.COLLECTION_NAME not in collections:
            logger.info(f"Creating Qdrant collection '{self.COLLECTION_NAME}' (dim={self.embedder.DIMENSIONS})...")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.embedder.DIMENSIONS,
                    distance=models.Distance.COSINE,
                ),
            )

            # Create payload indices for fast metadata filtering
            payload_fields = [
                ("companies", models.PayloadSchemaType.KEYWORD),
                ("is_company_tagged", models.PayloadSchemaType.BOOL),
                ("subreddit", models.PayloadSchemaType.KEYWORD),
                ("year_month", models.PayloadSchemaType.KEYWORD),
                ("themes", models.PayloadSchemaType.KEYWORD),
                ("actor_role", models.PayloadSchemaType.KEYWORD),
                ("depth", models.PayloadSchemaType.INTEGER),
                ("culture_intensity", models.PayloadSchemaType.INTEGER),
            ]
            for field_name, schema_type in payload_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as e:
                    logger.debug(f"Payload index creation note ({field_name}): {e}")

            logger.info(f"Collection '{self.COLLECTION_NAME}' initialized with payload indices.")

    def index_records(self, df_records: pd.DataFrame, batch_size: int = 64) -> int:
        """Embeds and upserts deduplicated records into Qdrant."""
        if len(df_records) == 0:
            logger.warning("No records provided to index.")
            return 0

        logger.info(f"Indexing {len(df_records):,} records into Qdrant collection '{self.COLLECTION_NAME}'...")
        points = []
        indexed_count = 0

        for idx, row in df_records.iterrows():
            rec_id = str(row.get("record_id", ""))
            text = str(row.get("clean_text", ""))
            if not text or not rec_id:
                continue

            # Generate unique deterministic UUID for Qdrant point
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rec_id))

            # Vector embedding
            vector = self.embedder.embed_text(text)

            # Process themes
            themes_val = row.get("themes", [])
            if isinstance(themes_val, str):
                try:
                    themes_list = json.loads(themes_val)
                except Exception:
                    themes_list = [themes_val]
            elif isinstance(themes_val, list):
                themes_list = themes_val
            else:
                themes_list = []

            # Process companies
            comps_val = row.get("companies", [])
            if isinstance(comps_val, str):
                try:
                    comps_list = json.loads(comps_val)
                except Exception:
                    comps_list = [comps_val] if comps_val else []
            elif isinstance(comps_val, list):
                comps_list = comps_val
            else:
                comps_list = []

            # Build metadata payload
            payload = {
                "record_id": rec_id,
                "post_id": str(row.get("post_id", "")),
                "comment_id": str(row.get("comment_id", "")),
                "depth": int(row.get("depth", 0)),
                "record_type": str(row.get("record_type", "post")),
                "subreddit": str(row.get("subreddit", "")),
                "author": str(row.get("author", "[deleted]")),
                "year": int(row.get("year", 2023)),
                "year_month": str(row.get("year_month", "")),
                "title": str(row.get("title", "")),
                "clean_text": text,
                "score": int(row.get("score", 0)),
                "permalink": str(row.get("permalink", "")),
                "duplicate_count": int(row.get("duplicate_count", 1)),
                "companies": comps_list,
                "is_company_tagged": bool(len(comps_list) > 0),
                "themes": themes_list,
                "actor_role": str(row.get("actor_role", "unknown")),
                "recommendation": str(row.get("recommendation", "unknown")),
                "culture_intensity": int(row.get("culture_intensity", 1)),
                "key_quote": str(row.get("key_quote", "") or ""),
                "vader_score": float(row.get("vader_score", 0.0)),
                "roberta_score": float(row.get("roberta_score", 0.0)),
                "llm_sentiment_score": float(row.get("llm_sentiment_score", 0.0)),
            }

            point = models.PointStruct(id=point_id, vector=vector, payload=payload)
            points.append(point)

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
                indexed_count += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
            indexed_count += len(points)

        logger.info(f"Qdrant indexing complete: {indexed_count:,} points stored in '{self.COLLECTION_NAME}'.")
        return indexed_count
