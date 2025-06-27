# backend/memory/hierarchical_memory.py

import asyncio
import json
import hashlib
import pickle
import lz4.frame
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import asyncpg
from collections import defaultdict

from backend.memory.base_memory import BaseMemory
from backend.utils.logger import Logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.core.minds.base_mind import BaseMind
    from backend.io_systems.natural_language_processor import NaturalLanguageProcessor


class MemoryTier(Enum):
    ACTIVE = "active"
    RECENT = "recent"
    LONG_TERM = "long_term"
    ARCHIVE = "archive"

@dataclass
class MemoryNode:
    node_id: str
    content: str
    embedding: Optional[List[float]]
    timestamp: datetime
    token_count: int
    importance_score: float
    access_frequency: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: MemoryTier = MemoryTier.ACTIVE
    chunk_id: Optional[str] = None
    # Add a similarity attribute for sorting search results, with a default
    similarity: float = 0.0

class HierarchicalMemory(BaseMemory):
    def __init__(self, config: Dict[str, Any], embedding_model: 'BaseMind', nlp_processor: 'NaturalLanguageProcessor'):
        super().__init__(config)
        if not all([embedding_model, nlp_processor]):
            raise ValueError("HierarchicalMemory requires valid embedding_model and nlp_processor instances.")
        self.embedding_model = embedding_model
        self.nlp_processor = nlp_processor
        
        mem_config = self.config.get('hierarchical_memory', self.config.get('memory', {}))
        self.vector_dimension = self.config.get("vector_dimension", 768)
        self.active_memory_threshold_nodes = mem_config.get("active_threshold_nodes", 150) # Increased default
        self.recent_tier_threshold_nodes = mem_config.get("recent_threshold_nodes", 1500) # Increased default
        self.active_memories: Dict[str, MemoryNode] = {}
        self.logger.info(f"HierarchicalMemory initialized. Active threshold: {self.active_memory_threshold_nodes} nodes.")

    @classmethod
    async def create(cls, config: Dict[str, Any], embedding_model: 'BaseMind', nlp_processor: 'NaturalLanguageProcessor'):
        instance = cls(config, embedding_model, nlp_processor)
        try:
            instance.pool = instance.db_manager.get_pool()
            instance._initialized = True
        except ConnectionError as e:
            instance.logger.critical("Failed to get database pool for HierarchicalMemory: %s", e, exc_info=True)
            raise
        return instance

    async def add(self, content: str, importance_score: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        self._check_initialized()
        embedding = await self._generate_embedding(content)
        if not embedding: 
            self.logger.warning(f"Could not generate embedding for content, memory not added: '{content[:50]}...'")
            return None

        if not self.nlp_processor or not hasattr(self.nlp_processor, 'nlp'):
            self.logger.error("NLP processor not available to count tokens.")
            token_count = len(content.split()) # Fallback
        else:
            token_count = len(self.nlp_processor.nlp.tokenizer(content))
        
        node_id = hashlib.sha256(f"{content}{datetime.utcnow().isoformat()}".encode('utf-8')).hexdigest()[:24]
        node = MemoryNode(node_id=node_id, content=content, embedding=embedding, timestamp=datetime.utcnow(), token_count=token_count, importance_score=importance_score, metadata=metadata or {})
        self.active_memories[node_id] = node
        self.logger.debug(f"Added memory {node_id} to ACTIVE tier. Active size: {len(self.active_memories)} nodes.")
        if len(self.active_memories) > self.active_memory_threshold_nodes:
            # Run promotion in a non-blocking way
            asyncio.create_task(self._promote_active_to_recent())
        return node_id

    async def _promote_active_to_recent(self):
        if not self.active_memories: return
        # Promote if over threshold, down to 80% of threshold to prevent rapid re-triggering
        if len(self.active_memories) <= self.active_memory_threshold_nodes: return

        num_to_promote = len(self.active_memories) - int(self.active_memory_threshold_nodes * 0.8)
        # Sort by timestamp first, then by importance (least important of the oldest get promoted)
        sorted_nodes = sorted(self.active_memories.values(), key=lambda n: (n.timestamp, n.importance_score))
        nodes_to_promote = sorted_nodes[:num_to_promote]
        
        self.logger.info(f"Promoting {len(nodes_to_promote)} nodes from ACTIVE to RECENT tier.")
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for node in nodes_to_promote:
                        await self._store_node_in_db(node, MemoryTier.RECENT, conn)
                        if node.node_id in self.active_memories:
                             del self.active_memories[node.node_id]
        except Exception as e:
            self.logger.error(f"Error promoting active memories to recent tier: {e}", exc_info=True)


    async def maintain_tiers(self):
        self._check_initialized()
        self.logger.info("Running memory tier maintenance...")
        try:
            recent_node_count = await self._fetchval("SELECT COUNT(*) FROM memory_nodes WHERE tier = $1", MemoryTier.RECENT.value) or 0
            if recent_node_count > self.recent_tier_threshold_nodes:
                self.logger.info(f"RECENT tier size ({recent_node_count}) exceeds threshold. Compressing oldest nodes...")
                # Fetch oldest nodes to compress into a chunk
                nodes_to_compress_records = await self._fetch("SELECT * FROM memory_nodes WHERE tier = $1 ORDER BY timestamp ASC LIMIT $2", MemoryTier.RECENT.value, self.recent_tier_threshold_nodes)
                if nodes_to_compress_records:
                    await self._compress_and_store_chunk(nodes_to_compress_records)
            else:
                self.logger.info("RECENT tier is within size limits. No compression needed.")
        except Exception as e:
            self.logger.error(f"Error during memory tier maintenance: {e}", exc_info=True)

    async def _compress_and_store_chunk(self, records: List[asyncpg.Record]):
        if not records: return
        self.logger.info(f"Compressing {len(records)} nodes into a LONG_TERM chunk.")
        nodes = [self._record_to_node(rec) for rec in records]
        combined_text = "\n\n---\n\n".join([node.content for node in nodes])
        summary = self.nlp_processor.summarize(combined_text, ratio=0.05) if self.nlp_processor else (combined_text[:200] + "...")
        keywords = set()
        if self.nlp_processor:
            entities = self.nlp_processor.extract_entities(combined_text)
            keywords = {entity[0].lower() for entity in entities[:30]}

        valid_embeddings = [node.embedding for node in nodes if node.embedding is not None and len(node.embedding) == self.vector_dimension]
        chunk_embedding = np.mean(np.array(valid_embeddings, dtype=np.float32), axis=0).tolist() if valid_embeddings else None
        chunk_node_data = [{"id": n.node_id, "ts": n.timestamp.isoformat(), "imp": n.importance_score, "meta": n.metadata, "emb": n.embedding} for n in nodes]
        serialized_data = pickle.dumps(chunk_node_data)
        compressed_data = lz4.frame.compress(serialized_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
        compression_ratio = len(compressed_data) / len(serialized_data) if serialized_data else 0
        chunk_id = hashlib.sha256(f"{nodes[0].timestamp.isoformat()}_{nodes[-1].timestamp.isoformat()}".encode('utf-8')).hexdigest()[:24]
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("INSERT INTO memory_chunks (chunk_id, tier, start_time, end_time, token_count, compressed_data, summary, keywords, embedding, compression_ratio) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)", chunk_id, MemoryTier.LONG_TERM.value, nodes[0].timestamp, nodes[-1].timestamp, sum(n.token_count for n in nodes), compressed_data, summary, json.dumps(list(keywords)), chunk_embedding, compression_ratio)
                node_ids_to_update = [node.node_id for node in nodes]
                await conn.execute("UPDATE memory_nodes SET tier = $1, chunk_id = $2 WHERE node_id = ANY($3::TEXT[])", MemoryTier.LONG_TERM.value, chunk_id, node_ids_to_update)
        self.logger.info(f"Successfully created LONG_TERM chunk {chunk_id}. Compression ratio: {compression_ratio:.2f}.")

    async def search(self, query: str, limit: int = 10, session_id: Optional[str] = None, min_importance: float = 0.0) -> List[Dict[str, Any]]:
        self.logger.info(f"Initiating multi-tier search for: '{query[:50]}...' in session '{session_id}'")
        is_history_retrieval = not query and session_id
        
        query_embedding_list = None
        if not is_history_retrieval:
            query_embedding_list = await self._generate_embedding(query)
            if not query_embedding_list: return []
        
        # Sequentially search tiers
        active_results = await self._search_active_memories(query_embedding_list, limit, session_id, is_history_retrieval)
        recent_results = await self._search_recent_tier_db(query_embedding_list, limit, session_id, min_importance, is_history_retrieval)
        long_term_results = []
        if not is_history_retrieval:
            long_term_results = await self._search_compressed_tier(query_embedding_list, limit, session_id)
        
        unique_results = {}
        for tier_results in [active_results, recent_results, long_term_results]:
            for res in tier_results:
                node_id = res.get('node_id')
                if not node_id: continue
                if node_id not in unique_results or res.get('similarity', 0) > unique_results[node_id].get('similarity', 0):
                    unique_results[node_id] = res

        sort_key = 'timestamp' if is_history_retrieval else 'similarity'
        reverse_order = True # Both history (newest first) and similarity (highest first) are descending
        final_results = sorted(unique_results.values(), key=lambda x: x.get(sort_key, 0 if sort_key=='similarity' else datetime.min), reverse=reverse_order)
        return final_results[:limit]

    async def _search_compressed_tier(self, query_embedding_list: Optional[List[float]], limit: int, session_id: Optional[str]) -> List[Dict[str, Any]]:
        if not query_embedding_list: return []
        query_embedding_str = '[' + ','.join(map(str, query_embedding_list)) + ']'
        chunk_query = "SELECT *, 1 - (embedding <=> $1) AS similarity FROM memory_chunks WHERE tier = $2 ORDER BY similarity DESC LIMIT 5"
        relevant_chunks_records = await self._fetch(chunk_query, query_embedding_str, MemoryTier.LONG_TERM.value)
        if not relevant_chunks_records: return []
        
        self.logger.debug(f"Found {len(relevant_chunks_records)} potentially relevant long-term chunks.")
        
        tasks = [self._search_in_chunk(chunk['chunk_id'], query_embedding_list, limit, session_id) for chunk in relevant_chunks_records]
        results_from_chunks = await asyncio.gather(*tasks)
        
        return [item for sublist in results_from_chunks for item in sublist]

    async def _search_in_chunk(self, chunk_id: str, query_embedding_list: List[float], limit: int, session_id: Optional[str]) -> List[Dict[str, Any]]:
        await self._execute("UPDATE memory_chunks SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE chunk_id = $1", chunk_id)
        compressed_data = await self._fetchval("SELECT compressed_data FROM memory_chunks WHERE chunk_id = $1", chunk_id)
        if not compressed_data: return []

        try:
            decompressed = lz4.frame.decompress(compressed_data)
            chunk_data: List[Dict] = pickle.loads(decompressed)
        except Exception as e:
            self.logger.error(f"Failed to decompress or unpickle chunk {chunk_id}: {e}", exc_info=True)
            return []
            
        candidate_nodes = []
        q_emb = np.array(query_embedding_list, dtype=np.float32)
        for node_data in chunk_data:
            if session_id and node_data.get('meta', {}).get('session_id') != session_id:
                continue
            
            node_emb_list = node_data.get('emb')
            if node_emb_list and isinstance(node_emb_list, list) and len(node_emb_list) == self.vector_dimension:
                n_emb = np.array(node_emb_list, dtype=np.float32)
                norm_q = np.linalg.norm(q_emb)
                norm_n = np.linalg.norm(n_emb)
                if norm_q > 0 and norm_n > 0:
                    similarity = np.dot(q_emb, n_emb) / (norm_q * norm_n)
                    if similarity > 0.6: 
                        node_data['similarity'] = float(similarity)
                        candidate_nodes.append(node_data)
        
        candidate_nodes.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        return [{
            'node_id': n['id'], 'content': "Retrieved from compressed chunk.", 
            'timestamp': datetime.fromisoformat(n['ts']), 'importance': n['imp'], 
            'similarity': n.get('similarity', 0.0), 'tier': MemoryTier.LONG_TERM.value,
            'metadata': n.get('meta', {}), 'chunk_id': chunk_id
        } for n in candidate_nodes[:limit]]

    async def get_context_window(self, session_id: str, current_query: str, max_tokens: int = 4096) -> List[Dict[str, str]]:
        history_nodes = await self.search(query="", session_id=session_id, limit=50) # Gets history newest first
        chat_history = []
        current_token_count = 0
        
        if not self.nlp_processor or not hasattr(self.nlp_processor, 'nlp'):
            self.logger.error("NLP processor not available to count tokens in get_context_window.")
            def count_tokens(text: str) -> int: return len(text.split())
        else:
            def count_tokens(text: str) -> int: return len(self.nlp_processor.nlp.tokenizer(text))
        
        # Iterate backwards through history (from newest to oldest) to fill context window
        for node in history_nodes:
            metadata = node.get('metadata', {})
            if not isinstance(metadata, dict): continue
            user_msg = metadata.get('input_text')
            ai_msg = metadata.get('output_text')
            if not (user_msg and ai_msg): continue

            # Count tokens for this turn
            turn_tokens = count_tokens(user_msg) + count_tokens(ai_msg)
            if current_token_count + turn_tokens > max_tokens: break

            # Add to the *front* of the list to maintain chronological order
            chat_history.insert(0, {"role": "assistant", "content": ai_msg})
            chat_history.insert(0, {"role": "user", "content": user_msg})
            current_token_count += turn_tokens
            
        chat_history.insert(0, {"role": "system", "content": "You are Prometheus, a helpful AI assistant."})
        return chat_history

    async def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        self._check_initialized()
        if node_id in self.active_memories: return self._format_node_as_dict(self.active_memories[node_id])
        record = await self._fetchrow("SELECT * FROM memory_nodes WHERE node_id = $1", node_id)
        return self._format_record_as_dict(record) if record else None

    async def delete(self, node_id: str) -> bool:
        self._check_initialized()
        if node_id in self.active_memories: del self.active_memories[node_id]
        status = await self._execute("DELETE FROM memory_nodes WHERE node_id = $1", node_id)
        return "DELETE 1" in status

    async def clear(self, session_id: Optional[str] = None):
        self._check_initialized()
        if session_id:
            self.active_memories = {nid: n for nid, n in self.active_memories.items() if n.metadata.get('session_id') != session_id}
            await self._execute("DELETE FROM memory_nodes WHERE metadata->>'session_id' = $1", session_id)
            await self._execute("DELETE FROM memory_chunks WHERE metadata->>'session_id' = $1", session_id) # Need to filter chunks too
        else:
            self.active_memories.clear()
            await self._execute("TRUNCATE TABLE memory_nodes, memory_chunks RESTART IDENTITY CASCADE;")

    async def _store_node_in_db(self, node: MemoryNode, tier: MemoryTier, connection: asyncpg.Connection):
        embedding_str = '[' + ','.join(map(str, node.embedding)) + ']' if node.embedding else None
        await connection.execute(
            """INSERT INTO memory_nodes (node_id, tier, content, embedding, timestamp, token_count, importance_score, metadata) 
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8) 
               ON CONFLICT (node_id) DO UPDATE SET 
                    tier = EXCLUDED.tier, 
                    access_frequency = memory_nodes.access_frequency + 1, 
                    metadata = EXCLUDED.metadata;""", 
            node.node_id, tier.value, node.content, embedding_str, node.timestamp, 
            node.token_count, node.importance_score, json.dumps(node.metadata)
        )
    
    async def _search_active_memories(self, query_embedding_list: Optional[List[float]], limit: int, session_id: Optional[str], is_history: bool) -> List[Dict[str, Any]]:
        candidate_nodes = [n for n in self.active_memories.values() if not session_id or n.metadata.get('session_id') == session_id]
        if not candidate_nodes: return []
        if is_history: return [self._format_node_as_dict(n) for n in sorted(candidate_nodes, key=lambda n: n.timestamp, reverse=True)[:limit]]
        
        if not query_embedding_list: return []
        q_emb = np.array(query_embedding_list, dtype=np.float32)
        
        for node in candidate_nodes:
            node.similarity = 0.0 # Reset similarity
            if node.embedding:
                n_emb = np.array(node.embedding, dtype=np.float32)
                norm_q = np.linalg.norm(q_emb)
                norm_n = np.linalg.norm(n_emb)
                if norm_q > 0 and norm_n > 0:
                    node.similarity = float(np.dot(q_emb, n_emb) / (norm_q * norm_n))

        candidate_nodes.sort(key=lambda n: n.similarity, reverse=True)
        return [self._format_node_as_dict(node, node.similarity) for node in candidate_nodes[:limit]]

    async def _search_recent_tier_db(self, query_embedding_list: Optional[List[float]], limit: int, session_id: Optional[str], min_importance: float, is_history: bool) -> List[Dict[str, Any]]:
        conditions, params = ["tier = $1"], [MemoryTier.RECENT.value]
        param_idx = 2
        if session_id: params.append(session_id); conditions.append(f"metadata->>'session_id' = ${param_idx}"); param_idx += 1
        if min_importance > 0: params.append(min_importance); conditions.append(f"importance_score >= ${param_idx}"); param_idx += 1
        
        order_by, sim_proj = "ORDER BY timestamp DESC", "1.0 AS similarity"
        if not is_history and query_embedding_list:
            # --- THIS IS THE FIX ---
            # Convert the list to a string representation for the query
            query_embedding_str = '[' + ','.join(map(str, query_embedding_list)) + ']'
            params.append(query_embedding_str)
            # --- END OF FIX ---
            sim_proj = f"1 - (embedding <=> ${param_idx}) AS similarity"
            order_by = f"ORDER BY similarity DESC"
            conditions.append(f"{sim_proj.replace(' AS similarity', '')} > 0.5") # Using WHERE on calculated similarity
            param_idx += 1

        params.append(limit)
        query = f"SELECT *, {sim_proj} FROM memory_nodes WHERE {' AND '.join(conditions)} {order_by} LIMIT ${param_idx}"
        return [self._format_record_as_dict(rec) for rec in await self._fetch(query, *params)]

    def _format_node_as_dict(self, node: MemoryNode, similarity: float = 1.0) -> Dict: 
        return {'node_id': node.node_id, 'content': node.content, 'timestamp': node.timestamp, 
                'importance': node.importance_score, 'similarity': similarity, 
                'tier': node.tier.value, 'metadata': node.metadata}

    def _format_record_as_dict(self, record: asyncpg.Record) -> Dict: 
        metadata_val = record.get('metadata')
        return {'node_id': record['node_id'], 'content': record['content'], 
                'timestamp': record['timestamp'], 'importance': record['importance_score'], 
                'similarity': record.get('similarity', 0.0), 'tier': record['tier'], 
                'metadata': json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val or {}}

    def _record_to_node(self, record: asyncpg.Record) -> MemoryNode:
        metadata_val = record.get('metadata')
        return MemoryNode(node_id=record['node_id'], content=record['content'], 
                        embedding=record['embedding'], timestamp=record['timestamp'], 
                        token_count=record['token_count'], importance_score=record['importance_score'], 
                        metadata=json.loads(metadata_val) if isinstance(metadata_val, str) else metadata_val or {}, 
                        tier=MemoryTier(record['tier']))

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text: return None
        try: 
            result_dict = await self.embedding_model.process({'text': text})
            embedding: Optional[List[float]] = result_dict.get('state')
            if embedding and len(embedding) == self.vector_dimension: return embedding
            self.logger.warning(f"Embedding generation returned None or mismatched dimension for text: '{text[:50]}...'")
            return None
        except Exception as e: 
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return None