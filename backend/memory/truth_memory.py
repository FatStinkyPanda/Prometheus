# backend/memory/truth_memory.py

import json
from typing import Any, Dict, List, Optional, TypeVar, TYPE_CHECKING, Tuple
from enum import Enum
from datetime import datetime
import asyncpg
import numpy as np

from backend.memory.base_memory import BaseMemory

if TYPE_CHECKING:
    from backend.core.minds.base_mind import BaseMind

T_TruthMemory = TypeVar('T_TruthMemory', bound='TruthMemory')

class TruthEvaluation(Enum):
    """Represents the outcome of evaluating a new claim against existing truths."""
    CONSISTENT = "consistent"
    CONTRADICTORY = "contradictory"
    NOVEL = "novel"
    UNCERTAIN = "uncertain"

class TruthMemory(BaseMemory):
    """Manages the persistent, factual knowledge base of the consciousness system."""

    def __init__(self, config: Dict[str, Any], embedding_mind: 'BaseMind'):
        super().__init__(config)
        if embedding_mind is None:
            raise ValueError("TruthMemory requires a valid embedding_mind instance.")
        self.embedding_mind = embedding_mind
        # Ensure embedding_mind.model and its config are loaded before accessing hidden_size
        if not (hasattr(embedding_mind, 'model') and embedding_mind.model and hasattr(embedding_mind.model, 'config') and embedding_mind.model.config):
             raise ValueError("TruthMemory: embedding_mind's model or model.config is not initialized.")
        self.vector_dimension = self.embedding_mind.model.config.hidden_size
        tm_config = self.config.get('truth_memory', {})
        self.conflict_resolution_threshold = tm_config.get('conflict_resolution_threshold', 0.8)
        self.logger.info("TruthMemory initialized with vector dimension %d.", self.vector_dimension)

    @classmethod
    async def create(cls: type[T_TruthMemory], config: Dict[str, Any], embedding_mind: 'BaseMind') -> T_TruthMemory:
        instance = cls(config, embedding_mind)
        try:
            instance.pool = instance.db_manager.get_pool()
            instance._initialized = True
        except ConnectionError as e:
            instance.logger.critical("Failed to get database pool for TruthMemory: %s", e, exc_info=True)
            raise
        return instance

    async def _generate_embedding_str(self, text: str) -> Optional[str]:
        """
        Generates a vector embedding for text content and returns it as a PostgreSQL-compatible string.
        e.g., '[0.1,0.2,0.3]'
        """
        self.logger.debug("Generating embedding for claim: '%s...'", text[:50])
        try:
            result = await self.embedding_mind.process({'text': text})
            embedding_list: Optional[List[float]] = result.get('state')
            
            if embedding_list is not None and isinstance(embedding_list, list) and len(embedding_list) == self.vector_dimension:
                # Convert list of floats to string format '[f1,f2,f3]'
                return '[' + ','.join(map(str, embedding_list)) + ']'
            elif embedding_list is not None:
                 self.logger.error(f"Embedding dimension mismatch or wrong type for TruthMemory. Expected list of {self.vector_dimension} floats, got {type(embedding_list)} of length {len(embedding_list) if isinstance(embedding_list, list) else 'N/A'}.")
                 return None
            return None
        except Exception as e:
            self.logger.error("Failed to generate embedding for text '%s'. Error: %s", text, e, exc_info=True)
            return None

    async def add(self, claim: str, value: str, confidence: float, evidence: List[Dict], source: str) -> Optional[str]:
        self._check_initialized()
        if value.upper() not in ('TRUE', 'FALSE', 'UNDETERMINED'): 
            self.logger.error(f"Invalid truth value '{value}' for claim '{claim}'. Must be TRUE, FALSE, or UNDETERMINED.")
            return None
            
        embedding_str = await self._generate_embedding_str(claim)
        if embedding_str is None: 
            self.logger.warning(f"Could not generate embedding for claim '{claim}'. Truth not added.")
            return None
            
        try:
            evidence_json = json.dumps(evidence)
        except TypeError as e:
            self.logger.error("Evidence for claim '%s' is not JSON-serializable. Error: %s", claim, e)
            return None
            
        query = """INSERT INTO truths (claim, value, confidence, evidence, source, claim_embedding, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6::vector, CURRENT_TIMESTAMP)
                   ON CONFLICT (claim) DO UPDATE SET 
                        value = EXCLUDED.value, 
                        confidence = EXCLUDED.confidence, 
                        evidence = EXCLUDED.evidence,
                        source = EXCLUDED.source, 
                        claim_embedding = EXCLUDED.claim_embedding, 
                        updated_at = CURRENT_TIMESTAMP
                   RETURNING id;"""
        try:
            record_id = await self._fetchval(query, claim, value.upper(), confidence, evidence_json, source, embedding_str)
            self.logger.info(f"Successfully added/updated truth for claim: '{claim[:50]}...'")
            return str(record_id) if record_id else None
        except Exception as e:
            self.logger.error("DB error adding truth claim '%s'. Error: %s", claim, e, exc_info=True)
            return None

    async def get(self, claim: str) -> Optional[Dict[str, Any]]:
        self._check_initialized()
        record = await self._fetchrow("SELECT * FROM truths WHERE claim = $1;", claim)
        if record:
            await self._execute("UPDATE truths SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = $1;", record['id'])
            return self._format_record(record)
        return None

    async def search(self, query_text: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        self._check_initialized()
        query_embedding_str = await self._generate_embedding_str(query_text)
        if query_embedding_str is None: 
            self.logger.warning(f"Could not generate embedding for search query '{query_text}'. Returning no results.")
            return []
            
        # Query for vector similarity search.
        # `<=>` is the L2 distance operator for pgvector. Smaller is more similar.
        # `1 - (claim_embedding <=> $1::vector)` converts distance to similarity score (0 to 1).
        query = """SELECT *, 1 - (claim_embedding <=> $1::vector) AS similarity FROM truths
                   WHERE 1 - (claim_embedding <=> $1::vector) > $2
                   ORDER BY similarity DESC LIMIT $3;"""
        try:
            records = await self._fetch(query, query_embedding_str, similarity_threshold, limit)
            return [self._format_record(rec) for rec in records]
        except asyncpg.PostgresError as e:
            self.logger.error("DB error during semantic search for truths. Error: %s", e, exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during truth search: {e}", exc_info=True)
            return []


    async def evaluate_claim(self, claim: str, claim_truth_value: str = 'TRUE') -> Tuple[TruthEvaluation, Optional[Dict[str, Any]]]:
        self._check_initialized()
        self.logger.info(f"Evaluating claim: '{claim}' with asserted value '{claim_truth_value}'")
        claim_truth_value = claim_truth_value.upper()
        
        related_truths = await self.search(claim, limit=1, similarity_threshold=self.conflict_resolution_threshold)

        if not related_truths:
            self.logger.info(f"Claim '{claim[:50]}...' is NOVEL. No similar truths found above threshold {self.conflict_resolution_threshold}.")
            return TruthEvaluation.NOVEL, None

        most_relevant_truth = related_truths[0]
        existing_value = most_relevant_truth['value'].upper() # Ensure comparison is case-insensitive
        
        similarity_score = most_relevant_truth.get('similarity', 0.0)
        self.logger.debug(f"Found related truth: '{most_relevant_truth['claim'][:50]}...' (Value: {existing_value}, Similarity: {similarity_score:.3f})")

        is_contradictory = (claim_truth_value == 'TRUE' and existing_value == 'FALSE') or \
                          (claim_truth_value == 'FALSE' and existing_value == 'TRUE')
        
        if is_contradictory:
            self.logger.warning(f"Claim '{claim[:50]}...' is CONTRADICTORY with existing truth ID {most_relevant_truth['id']} ('{most_relevant_truth['claim'][:50]}...').")
            return TruthEvaluation.CONTRADICTORY, most_relevant_truth

        if claim_truth_value == existing_value:
            self.logger.info(f"Claim '{claim[:50]}...' is CONSISTENT with existing truth ID {most_relevant_truth['id']}.")
            return TruthEvaluation.CONSISTENT, most_relevant_truth
            
        self.logger.info(f"Claim '{claim[:50]}...' is UNCERTAIN relative to existing truth ID {most_relevant_truth['id']} (Existing: {existing_value}, New: {claim_truth_value}).")
        return TruthEvaluation.UNCERTAIN, most_relevant_truth

    async def delete(self, claim: str) -> bool:
        self._check_initialized()
        status = await self._execute("DELETE FROM truths WHERE claim = $1;", claim)
        return status is not None and status.endswith('1')


    async def clear(self) -> None:
        self._check_initialized()
        await self._execute("TRUNCATE TABLE truths RESTART IDENTITY;")
        self.logger.info("TruthMemory cleared (TRUNCATE TABLE truths).")


    def _format_record(self, record: asyncpg.Record) -> Dict[str, Any]:
        """Helper to convert a database record into a dictionary, handling JSON fields."""
        data = dict(record)
        if data.get("evidence") and isinstance(data["evidence"], str):
            try:
                data["evidence"] = json.loads(data["evidence"])
            except json.JSONDecodeError:
                self.logger.warning(f"Could not decode JSON evidence for truth ID {data.get('id')}: {data['evidence']}")
                data["evidence"] = [] # Default to empty list on error
        elif data.get("evidence") is None:
             data["evidence"] = []

        # Ensure 'id' is always a string
        if "id" in data and data["id"] is not None:
            data["id"] = str(data["id"])
            
        # Format datetime objects to ISO strings if they exist
        for dt_field in ["created_at", "updated_at", "last_accessed"]:
            if dt_field in data and isinstance(data[dt_field], datetime):
                data[dt_field] = data[dt_field].isoformat()
        
        return data