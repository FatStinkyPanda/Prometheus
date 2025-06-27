# backend/memory/contextual_memory.py

import json
from typing import Any, Dict, List, Optional, TypeVar, TYPE_CHECKING

import asyncpg
import numpy as np

from backend.memory.base_memory import BaseMemory

# Use TYPE_CHECKING to prevent circular import errors at runtime
if TYPE_CHECKING:
    from backend.core.minds.base_mind import BaseMind
    from backend.memory.truth_memory import TruthMemory
    from backend.io_systems.natural_language_processor import NaturalLanguageProcessor

# Generic TypeVar for the class instance, used for the factory method's return type.
T_ContextualMemory = TypeVar('T_ContextualMemory', bound='ContextualMemory')


class ContextualMemory(BaseMemory):
    """
    Manages the long-term, episodic memory of all system interactions.

    This memory component stores a complete history of conversations and actions.
    It uses vector embeddings for semantic search and includes a sophisticated
    method to build a rich context prompt for the AI's awareness.
    """

    def __init__(self, config: Dict[str, Any], embedding_model: 'BaseMind', truth_memory: 'TruthMemory', nlp: 'NaturalLanguageProcessor'):
        """
        Initializes the ContextualMemory instance.
        """
        super().__init__(config)
        if any(arg is None for arg in [embedding_model, truth_memory, nlp]):
            raise ValueError("ContextualMemory requires valid instances of embedding_model, truth_memory, and nlp.")
            
        self.embedding_model = embedding_model
        self.truth_memory = truth_memory
        self.nlp = nlp
        
        mem_config = self.config.get('contextual_memory', {})
        self.vector_dimension = self.config.get("vector_dimension", 768)
        self.context_window = mem_config.get('context_window', 100)
        self.relevance_threshold = mem_config.get('relevance_threshold', 0.75)
        
        self.logger.info("ContextualMemory initialized with vector dimension %d and context window of %d.",
                         self.vector_dimension, self.context_window)

    @classmethod
    async def create(cls: type[T_ContextualMemory], config: Dict[str, Any], embedding_model: 'BaseMind', truth_memory: 'TruthMemory', nlp: 'NaturalLanguageProcessor') -> T_ContextualMemory:
        """
        Asynchronous factory for creating and initializing a ContextualMemory instance.
        """
        instance = cls(config, embedding_model, truth_memory, nlp)
        try:
            instance.pool = instance.db_manager.get_pool()
            instance._initialized = True
            instance.logger.info("Successfully initialized and connected to the database pool.")
        except ConnectionError as e:
            instance.logger.critical("Failed to get database pool during ContextualMemory initialization. %s", e, exc_info=True)
            raise
        return instance

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generates a vector embedding for text content."""
        if not text:
            return None
        self.logger.debug("Generating context embedding for text: '%s...'", text[:70])
        try:
            # We must pass the logical_mind instance to the other minds for state generation
            result = await self.embedding_model.process({'text': text})
            embedding = result.get('state')
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            if len(embedding) != self.vector_dimension:
                self.logger.error("Embedding dimension mismatch! Expected %d, got %d. Check your model.",
                                  self.vector_dimension, len(embedding))
                return None
            return embedding
        except Exception as e:
            self.logger.error("Failed to generate context embedding. Error: %s", e, exc_info=True)
            return None

    async def add(self, session_id: str, interaction_type: str, input_data: Dict,
                  output_data: Dict, unified_state: List[float], text_content: str,
                  mind_states: Dict, emotional_context: Dict) -> Optional[str]:
        """Adds a new interaction entry to the memory."""
        self.logger.info("Adding new '%s' interaction for session '%s'.", interaction_type, session_id)
        text_embedding = await self._generate_embedding(text_content)
        try:
            query = """
                INSERT INTO contextual_interactions (session_id, interaction_type, input_data, output_data,
                                                     unified_state, text_content, text_embedding, mind_states,
                                                     emotional_context)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id;
            """
            record_id = await self._fetchval(
                query, session_id, interaction_type, json.dumps(input_data), json.dumps(output_data),
                unified_state, text_content, text_embedding, json.dumps(mind_states), json.dumps(emotional_context)
            )
            return str(record_id)
        except Exception as e:
            self.logger.error("Database error while adding contextual interaction. Error: %s", e, exc_info=True)
            return None

    async def get(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific interaction by its UUID."""
        record = await self._fetchrow("SELECT * FROM contextual_interactions WHERE id = $1;", interaction_id)
        return self._format_record(record) if record else None

    async def search(self, session_id: str, query_text: str, semantic_limit: int = 5, recent_limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves relevant context by combining recent and semantically similar interactions."""
        self.logger.debug("Searching for context for session '%s' with query: '%s'", session_id, query_text)
        query_embedding = await self._generate_embedding(query_text)
        
        # Combine recent and semantic searches into one query for efficiency
        # This uses a UNION to merge results and then sorts them.
        sql_query = """
            WITH recent AS (
                SELECT *, 0.0 as similarity FROM contextual_interactions
                WHERE session_id = $1 ORDER BY timestamp DESC LIMIT $2
            ),
            semantic AS (
                SELECT *, (1 - (text_embedding <=> $3)) as similarity FROM contextual_interactions
                WHERE session_id = $1 AND text_embedding IS NOT NULL
                ORDER BY similarity DESC LIMIT $4
            )
            SELECT * FROM recent
            UNION
            SELECT * FROM semantic;
        """
        
        records = await self._fetch(sql_query, session_id, recent_limit, query_embedding, semantic_limit)
        
        # De-duplicate results, keeping the one with higher similarity if available
        unique_interactions = {}
        for r in records:
            if r['id'] not in unique_interactions or r['similarity'] > unique_interactions[r['id']].get('similarity', 0.0):
                unique_interactions[r['id']] = self._format_record(r)
                
        sorted_interactions = sorted(unique_interactions.values(), key=lambda x: x['timestamp'])
        self.logger.info("Found %d relevant context entries for session '%s'.", len(sorted_interactions), session_id)
        return sorted_interactions

    async def build_context_prompt(self, session_id: str, current_input: str) -> str:
        """
        Constructs a rich, summarized context prompt for the AI's awareness.
        """
        self.logger.info(f"Building context prompt for session '{session_id}'.")
        
        # 1. Get recent interactions
        recent_interactions = await self.search(session_id, current_input, semantic_limit=0, recent_limit=5)
        
        # 2. Get semantically relevant interactions
        semantic_interactions = await self.search(session_id, current_input, semantic_limit=3, recent_limit=0)
        
        # 3. Get relevant truths from Truth Memory
        relevant_truths = await self.truth_memory.search(current_input, limit=3, similarity_threshold=self.relevance_threshold)

        # 4. Combine and summarize
        prompt_parts = []
        
        if relevant_truths:
            truths_summary = " ".join([f"It is known that '{t['claim']}' is {t['value'].lower()}." for t in relevant_truths])
            prompt_parts.append(f"[Established Facts]\n{truths_summary}")

        combined_interactions = {item['id']: item for item in recent_interactions + semantic_interactions}
        if combined_interactions:
            # We only want to summarize older, relevant interactions, not the very last one.
            context_summary_text = "\n".join([item['text_content'] for item in sorted(combined_interactions.values(), key=lambda x: x['timestamp'])[:-1]])
            if context_summary_text:
                summary = self.nlp.summarize(context_summary_text, ratio=0.5)
                if summary:
                    prompt_parts.append(f"[Summary of Relevant Past Conversation]\n{summary}")
        
        if not prompt_parts:
            return "" # Return empty string if no relevant context was found
            
        final_prompt = "--- CONTEXT FOR CURRENT INPUT ---\n" + "\n\n".join(prompt_parts) + "\n--- END CONTEXT ---\n\n"
        self.logger.info(f"Generated context prompt of length {len(final_prompt)} for session '{session_id}'.")
        return final_prompt

    async def delete(self, interaction_id: str) -> bool:
        """Deletes a specific interaction by its UUID."""
        status = await self._execute("DELETE FROM contextual_interactions WHERE id = $1;", interaction_id)
        return status.endswith('1')

    async def clear(self, session_id: str) -> None:
        """Clears all interaction history for a specific session."""
        await self._execute("DELETE FROM contextual_interactions WHERE session_id = $1;", session_id)

    def _format_record(self, record: asyncpg.Record) -> Dict[str, Any]:
        """Helper to convert a database record into a dictionary."""
        data = dict(record)
        for field in ['input_data', 'output_data', 'mind_states', 'emotional_context']:
            if data.get(field) and isinstance(data[field], str):
                try: data[field] = json.loads(data[field])
                except json.JSONDecodeError: data[field] = {}
        data['id'] = str(data['id'])
        return data