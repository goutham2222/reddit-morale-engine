"""Payload-filtered Maximal Marginal Relevance (MMR) retriever in persistent Qdrant."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.common.config import EngineConfig
from src.common.logger import get_logger
from src.rag.embeddings import EmbeddingModel
from src.rag.qdrant_indexer import QdrantIndexer

logger = get_logger("retriever")


class QdrantMMRRetriever:
    """Retrieves diverse, payload-filtered employee evidence using MMR in Qdrant."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        indexer: Optional[QdrantIndexer] = None,
    ):
        self.config = config or EngineConfig.load()
        self.indexer = indexer or QdrantIndexer(self.config)
        self.client = self.indexer.client
        self.embedder = self.indexer.embedder
        self.collection_name = self.indexer.COLLECTION_NAME

    def _build_filter(
        self,
        companies: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        actor_roles: Optional[List[str]] = None,
        depth: Optional[int] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Optional[models.Filter]:
        """Constructs a boolean metadata filter for Qdrant payload search."""
        must_conditions = []

        # Filter by companies (Matches if any specified company is present in payload list)
        if companies:
            must_conditions.append(
                models.FieldCondition(
                    key="companies",
                    match=models.MatchAny(any=companies),
                )
            )

        # Filter by subreddits
        if subreddits:
            must_conditions.append(
                models.FieldCondition(
                    key="subreddit",
                    match=models.MatchAny(any=subreddits),
                )
            )

        # Filter by cultural themes
        if themes:
            must_conditions.append(
                models.FieldCondition(
                    key="themes",
                    match=models.MatchAny(any=themes),
                )
            )

        # Filter by actor roles (e.g. ['current_employee', 'former_employee'])
        if actor_roles:
            must_conditions.append(
                models.FieldCondition(
                    key="actor_role",
                    match=models.MatchAny(any=actor_roles),
                )
            )

        # Filter by depth (e.g. depth=0 for posts only)
        if depth is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="depth",
                    match=models.MatchValue(value=depth),
                )
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def retrieve(
        self,
        query: str,
        companies: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        actor_roles: Optional[List[str]] = None,
        depth: Optional[int] = None,
        top_k: int = 8,
        fetch_k: int = 40,
        lambda_mult: float = 0.7,
    ) -> List[Dict]:
        """Executes payload-filtered MMR retrieval to return diverse evidence without duplicate spam."""
        query_vec = self.embedder.embed_text(query)
        qdrant_filter = self._build_filter(
            companies=companies,
            subreddits=subreddits,
            themes=themes,
            actor_roles=actor_roles,
            depth=depth,
        )

        try:
            # Fetch candidates from Qdrant (supporting modern query_points and legacy search)
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    query_filter=qdrant_filter,
                    limit=fetch_k,
                    with_payload=True,
                    with_vectors=True,
                )
                candidates = response.points
            else:
                candidates = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    query_filter=qdrant_filter,
                    limit=fetch_k,
                    with_payload=True,
                    with_vectors=True,
                )
        except Exception as e:
            logger.error(f"Qdrant search error ({e}). Returning empty results.")
            return []

        if not candidates:
            # If no candidates matched strict company filter, log and return empty
            logger.info(f"No records matched query '{query[:40]}' with filters: companies={companies}")
            return []

        # If we have fewer candidates than top_k, return all directly
        if len(candidates) <= top_k:
            return [self._format_hit(c) for c in candidates]

        # ---------------------------------------------------------------------
        # Maximal Marginal Relevance (MMR) Reranking
        # ---------------------------------------------------------------------
        cand_vectors = np.array([c.vector for c in candidates])
        query_vector = np.array(query_vec)

        # Cosine similarity to query
        query_sims = np.dot(cand_vectors, query_vector)

        # Pairwise similarity matrix among candidates
        norm_cand = cand_vectors / (np.linalg.norm(cand_vectors, axis=1, keepdims=True) + 1e-9)
        pairwise_sims = np.dot(norm_cand, norm_cand.T)

        selected_indices = [int(np.argmax(query_sims))]
        unselected_indices = list(set(range(len(candidates))) - set(selected_indices))

        while len(selected_indices) < top_k and unselected_indices:
            best_idx = None
            best_mmr_score = -float("inf")

            for idx in unselected_indices:
                sim_to_query = query_sims[idx]
                max_sim_to_selected = max(pairwise_sims[idx, s] for s in selected_indices)
                mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
            else:
                break

        mmr_results = [self._format_hit(candidates[i]) for i in selected_indices]
        return mmr_results

    def _format_hit(self, hit) -> Dict:
        """Formats a Qdrant search hit into a research citation item."""
        p = hit.payload or {}
        rec_id = p.get("record_id", "")
        sub = p.get("subreddit", "")
        ym = p.get("year_month", "")
        role = p.get("actor_role", "unknown")
        quote = p.get("key_quote") or (p.get("clean_text", "")[:160] + "...")

        return {
            "record_id": rec_id,
            "score": round(hit.score, 4),
            "subreddit": sub,
            "year_month": ym,
            "author": p.get("author", "[deleted]"),
            "role": role,
            "companies": p.get("companies", []),
            "themes": p.get("themes", []),
            "roberta_score": p.get("roberta_score", 0.0),
            "llm_sentiment_score": p.get("llm_sentiment_score", 0.0),
            "key_quote": quote,
            "text": p.get("clean_text", ""),
            "permalink": p.get("permalink", ""),
            "duplicate_count": p.get("duplicate_count", 1),
            "citation": f"[r/{sub}, {ym}] (Role: {role})",
        }
