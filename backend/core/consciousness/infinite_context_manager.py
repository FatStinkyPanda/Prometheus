# backend/core/consciousness/infinite_context_manager.py

import asyncio
import hashlib
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import json

from backend.memory.base_memory import BaseMemory
from backend.memory.hierarchical_memory import HierarchicalMemory, MemoryTier
from backend.utils.logger import Logger

class ContextWindow:
    """Represents a sliding context window with dynamic compression."""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.tokens = deque(maxlen=max_tokens)
        self.token_metadata = deque(maxlen=max_tokens)
        self.compression_ratio = 1.0
        self.importance_scores = deque(maxlen=max_tokens)
        
    def add_tokens(self, new_tokens: List[str], metadata: List[Dict], importance: List[float]):
        """Add new tokens with metadata and importance scores."""
        for token, meta, imp in zip(new_tokens, metadata, importance):
            self.tokens.append(token)
            self.token_metadata.append(meta)
            self.importance_scores.append(imp)
            
    def get_compressed_context(self, target_tokens: int) -> Tuple[List[str], float]:
        """Get a compressed version of the context that fits within target tokens."""
        if len(self.tokens) <= target_tokens:
            return list(self.tokens), 1.0
            
        # Implement importance-based compression
        token_importance_pairs = list(zip(self.tokens, self.importance_scores))
        token_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_tokens = [pair[0] for pair in token_importance_pairs[:target_tokens]]
        compression_ratio = target_tokens / len(self.tokens)
        
        return selected_tokens, compression_ratio

@dataclass
class StreamingContext:
    """Manages streaming context across unlimited tokens."""
    session_id: str
    current_position: int = 0
    total_tokens_processed: int = 0
    active_windows: List[ContextWindow] = field(default_factory=list)
    context_checkpoints: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    streaming_state: Dict[str, Any] = field(default_factory=dict)

class InfiniteContextManager:
    """
    Manages virtually unlimited context through intelligent streaming,
    compression, and hierarchical attention mechanisms.
    """
    
    def __init__(self, hierarchical_memory: HierarchicalMemory, nlp_processor, embedding_model):
        self.logger = Logger(__name__)
        self.hierarchical_memory = hierarchical_memory
        self.nlp_processor = nlp_processor
        self.embedding_model = embedding_model
        
        # Configuration for context management
        self.window_size = 4096  # Base context window
        self.num_parallel_windows = 4  # Number of overlapping windows
        self.checkpoint_interval = 10000  # Create checkpoint every N tokens
        self.compression_levels = [1.0, 0.5, 0.25, 0.1]  # Progressive compression
        
        # Active streaming contexts
        self.streaming_contexts: Dict[str, StreamingContext] = {}
        
        # Learning parameters for self-improvement
        self.attention_patterns: Dict[str, np.ndarray] = {}
        self.context_importance_model = None
        
        self.logger.info("InfiniteContextManager initialized with streaming capabilities")
        
    async def create_streaming_session(self, session_id: str) -> StreamingContext:
        """Create a new streaming context session."""
        context = StreamingContext(
            session_id=session_id,
            active_windows=[ContextWindow(self.window_size) for _ in range(self.num_parallel_windows)]
        )
        self.streaming_contexts[session_id] = context
        
        # Initialize context with system prompt
        await self._initialize_context_windows(context)
        
        self.logger.info(f"Created streaming session {session_id} with {self.num_parallel_windows} parallel windows")
        return context
        
    async def stream_input(self, session_id: str, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an unlimited stream of input text while maintaining context awareness.
        Yields incremental processing results.
        """
        context = self.streaming_contexts.get(session_id)
        if not context:
            context = await self.create_streaming_session(session_id)
            
        buffer = ""
        tokens_in_buffer = 0
        
        async for chunk in text_stream:
            buffer += chunk
            
            # Tokenize the buffer
            doc = self.nlp_processor.nlp(buffer)
            new_tokens = [token.text for token in doc]
            
            # Process when we have enough tokens or hit a sentence boundary
            if len(new_tokens) - tokens_in_buffer > 100 or chunk.endswith(('.', '!', '?')):
                # Extract the complete sentences
                sentences = [sent.text for sent in doc.sents]
                if len(sentences) > 1:
                    complete_text = ' '.join(sentences[:-1])
                    buffer = sentences[-1]  # Keep incomplete sentence
                    
                    # Process the complete text
                    result = await self._process_streaming_chunk(context, complete_text)
                    
                    # Update token count
                    context.total_tokens_processed += len(new_tokens) - len(buffer.split())
                    tokens_in_buffer = len(buffer.split())
                    
                    yield result
                    
                    # Create checkpoint if needed
                    if context.total_tokens_processed % self.checkpoint_interval == 0:
                        await self._create_context_checkpoint(context)
                        
        # Process any remaining buffer
        if buffer.strip():
            result = await self._process_streaming_chunk(context, buffer)
            yield result
            
    async def _process_streaming_chunk(self, context: StreamingContext, text: str) -> Dict[str, Any]:
        """Process a chunk of text within the streaming context."""
        # Generate embedding for the chunk
        embedding = await self._generate_embedding(text)
        
        # Calculate importance score for the chunk
        importance_score = await self._calculate_chunk_importance(text, context)
        
        # Update all active context windows with different strategies
        window_results = []
        for i, window in enumerate(context.active_windows):
            # Each window uses a different compression level
            compression_level = self.compression_levels[min(i, len(self.compression_levels)-1)]
            
            # Add tokens to window
            tokens = self.nlp_processor.nlp.tokenizer(text)
            importance_scores = [importance_score] * len(tokens)
            metadata = [{"chunk_id": context.current_position, "compression": compression_level}] * len(tokens)
            
            window.add_tokens(tokens, metadata, importance_scores)
            
            # Get compressed context for this window
            compressed_context, ratio = window.get_compressed_context(int(self.window_size * compression_level))
            window_results.append({
                "window_id": i,
                "compression_ratio": ratio,
                "context_size": len(compressed_context)
            })
            
        # Store in hierarchical memory with enhanced metadata
        await self.hierarchical_memory.add(
            content=text,
            importance_score=importance_score,
            metadata={
                "session_id": context.session_id,
                "position": context.current_position,
                "total_tokens": context.total_tokens_processed,
                "embedding": embedding,
                "window_states": window_results
            }
        )
        
        context.current_position += 1
        
        # Build multi-scale context representation
        multi_scale_context = await self._build_multi_scale_context(context)
        
        return {
            "chunk_position": context.current_position,
            "total_tokens_processed": context.total_tokens_processed,
            "importance_score": importance_score,
            "compression_ratios": [w["compression_ratio"] for w in window_results],
            "multi_scale_context": multi_scale_context,
            "memory_utilization": self._calculate_memory_utilization()
        }
        
    async def get_relevant_context(self, session_id: str, query: str, max_context_tokens: int = 8192) -> Dict[str, Any]:
        """
        Retrieve the most relevant context for a query from potentially billions of tokens.
        Uses multi-scale attention and intelligent retrieval.
        """
        context = self.streaming_contexts.get(session_id)
        if not context:
            return {"error": "Session not found"}
            
        # Perform multi-scale search across all memory tiers
        search_tasks = [
            self._search_active_context(context, query),
            self._search_compressed_context(context, query),
            self._search_hierarchical_memory(session_id, query),
            self._search_checkpoints(context, query)
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine and rank results
        all_results = []
        for tier_results in search_results:
            all_results.extend(tier_results)
            
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Build optimal context within token limit
        selected_context = []
        current_tokens = 0
        
        for result in all_results:
            result_tokens = len(self.nlp_processor.nlp.tokenizer(result['content']))
            if current_tokens + result_tokens <= max_context_tokens:
                selected_context.append(result)
                current_tokens += result_tokens
            else:
                # Try to compress and fit
                compressed = await self._compress_result(result, max_context_tokens - current_tokens)
                if compressed:
                    selected_context.append(compressed)
                break
                
        # Generate context summary
        context_summary = await self._generate_context_summary(selected_context)
        
        return {
            "relevant_context": selected_context,
            "total_results": len(all_results),
            "context_tokens": current_tokens,
            "context_summary": context_summary,
            "search_coverage": {
                "total_tokens_in_session": context.total_tokens_processed,
                "chunks_searched": len(all_results),
                "compression_ratio": current_tokens / context.total_tokens_processed if context.total_tokens_processed > 0 else 0
            }
        }
        
    async def _build_multi_scale_context(self, context: StreamingContext) -> Dict[str, Any]:
        """Build a multi-scale representation of the current context."""
        scales = {}
        
        # Immediate context (last 100 tokens)
        immediate_tokens = list(context.active_windows[0].tokens)[-100:]
        scales['immediate'] = ' '.join(immediate_tokens) if immediate_tokens else ""
        
        # Recent context (last 1000 tokens with compression)
        if len(context.active_windows) > 1:
            recent_compressed, _ = context.active_windows[1].get_compressed_context(500)
            scales['recent'] = ' '.join(recent_compressed)
            
        # Extended context from checkpoints
        if context.context_checkpoints:
            latest_checkpoint = max(context.context_checkpoints.keys())
            scales['extended'] = context.context_checkpoints[latest_checkpoint].get('summary', '')
            
        # Global context from hierarchical memory
        global_summary = await self._get_global_context_summary(context.session_id)
        scales['global'] = global_summary
        
        return scales
        
    async def _create_context_checkpoint(self, context: StreamingContext):
        """Create a checkpoint of the current context state."""
        checkpoint_data = {
            'position': context.current_position,
            'total_tokens': context.total_tokens_processed,
            'timestamp': datetime.utcnow().isoformat(),
            'window_states': []
        }
        
        # Compress and store each window state
        for i, window in enumerate(context.active_windows):
            compressed_tokens, ratio = window.get_compressed_context(1000)
            checkpoint_data['window_states'].append({
                'window_id': i,
                'compressed_tokens': compressed_tokens,
                'compression_ratio': ratio,
                'importance_distribution': list(window.importance_scores)
            })
            
        # Generate checkpoint summary
        checkpoint_text = ' '.join([' '.join(ws['compressed_tokens'][:100]) for ws in checkpoint_data['window_states']])
        checkpoint_data['summary'] = self.nlp_processor.summarize(checkpoint_text, ratio=0.1)
        
        # Store checkpoint
        context.context_checkpoints[context.current_position] = checkpoint_data
        
        # Store in hierarchical memory for long-term retrieval
        await self.hierarchical_memory.add(
            content=checkpoint_data['summary'],
            importance_score=0.9,  # Checkpoints are important
            metadata={
                'type': 'checkpoint',
                'session_id': context.session_id,
                'checkpoint_position': context.current_position,
                'checkpoint_data': checkpoint_data
            }
        )
        
        self.logger.info(f"Created checkpoint at position {context.current_position} for session {context.session_id}")
        
    async def _calculate_chunk_importance(self, text: str, context: StreamingContext) -> float:
        """Calculate importance score for a text chunk using multiple factors."""
        importance_factors = []
        
        # 1. Entity density
        entities = self.nlp_processor.extract_entities(text)
        entity_density = len(entities) / max(len(text.split()), 1)
        importance_factors.append(entity_density * 2.0)
        
        # 2. Semantic novelty compared to recent context
        if context.active_windows[0].tokens:
            recent_text = ' '.join(list(context.active_windows[0].tokens)[-500:])
            recent_embedding = await self._generate_embedding(recent_text)
            chunk_embedding = await self._generate_embedding(text)
            
            if recent_embedding and chunk_embedding:
                similarity = np.dot(recent_embedding, chunk_embedding) / (np.linalg.norm(recent_embedding) * np.linalg.norm(chunk_embedding))
                novelty = 1.0 - similarity
                importance_factors.append(novelty)
                
        # 3. Question indicators
        if '?' in text:
            importance_factors.append(1.5)
            
        # 4. Emotional content
        # This would use the emotional mind but simplified here
        emotional_indicators = ['important', 'critical', 'urgent', 'remember', 'key', 'crucial']
        emotional_score = sum(1 for word in emotional_indicators if word in text.lower()) / 10.0
        importance_factors.append(emotional_score)
        
        # 5. Structural importance (headings, lists, etc.)
        if text.strip().startswith(('#', '*', '-', '1.', 'Chapter', 'Section')):
            importance_factors.append(1.2)
            
        # Combine factors
        base_importance = np.mean(importance_factors) if importance_factors else 0.5
        
        # Apply learned attention patterns if available
        if context.session_id in self.attention_patterns:
            pattern_weight = self._apply_attention_pattern(text, context)
            base_importance = base_importance * 0.7 + pattern_weight * 0.3
            
        return min(max(base_importance, 0.1), 1.0)
        
    async def learn_attention_patterns(self, session_id: str, user_feedback: Dict[str, Any]):
        """Learn from user feedback to improve context importance scoring."""
        if session_id not in self.attention_patterns:
            self.attention_patterns[session_id] = np.zeros(100)  # Simplified pattern vector
            
        # Update attention patterns based on feedback
        # This is a simplified version - in production would use more sophisticated ML
        if user_feedback.get('relevant_chunks'):
            for chunk_id in user_feedback['relevant_chunks']:
                # Increase weight for similar patterns
                self.attention_patterns[session_id][chunk_id % 100] += 0.1
                
        if user_feedback.get('irrelevant_chunks'):
            for chunk_id in user_feedback['irrelevant_chunks']:
                # Decrease weight for similar patterns
                self.attention_patterns[session_id][chunk_id % 100] -= 0.05
                
        # Normalize
        self.attention_patterns[session_id] = np.clip(self.attention_patterns[session_id], -1, 1)
        
        self.logger.info(f"Updated attention patterns for session {session_id}")
        
    async def export_session_context(self, session_id: str) -> Dict[str, Any]:
        """Export the full context of a session for analysis or backup."""
        context = self.streaming_contexts.get(session_id)
        if not context:
            return {"error": "Session not found"}
            
        export_data = {
            "session_id": session_id,
            "total_tokens_processed": context.total_tokens_processed,
            "num_chunks": context.current_position,
            "checkpoints": []
        }
        
        # Export checkpoints
        for position, checkpoint in context.context_checkpoints.items():
            export_data["checkpoints"].append({
                "position": position,
                "summary": checkpoint.get("summary", ""),
                "timestamp": checkpoint.get("timestamp", "")
            })
            
        # Export memory statistics
        memory_stats = await self._get_session_memory_stats(session_id)
        export_data["memory_stats"] = memory_stats
        
        return export_data
        
    def _calculate_memory_utilization(self) -> Dict[str, float]:
        """Calculate current memory utilization metrics."""
        active_sessions = len(self.streaming_contexts)
        total_windows = sum(len(ctx.active_windows) for ctx in self.streaming_contexts.values())
        total_checkpoints = sum(len(ctx.context_checkpoints) for ctx in self.streaming_contexts.values())
        
        return {
            "active_sessions": active_sessions,
            "total_context_windows": total_windows,
            "total_checkpoints": total_checkpoints,
            "estimated_memory_mb": (total_windows * self.window_size * 50) / (1024 * 1024)  # Rough estimate
        }
        
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using the embedding model."""
        try:
            result = await self.embedding_model.process({'text': text})
            return result.get('state')
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
            
    async def _search_active_context(self, context: StreamingContext, query: str) -> List[Dict[str, Any]]:
        """Search within active context windows."""
        results = []
        query_embedding = await self._generate_embedding(query)
        
        for i, window in enumerate(context.active_windows):
            if window.tokens:
                window_text = ' '.join(window.tokens)
                window_embedding = await self._generate_embedding(window_text)
                
                if query_embedding and window_embedding:
                    similarity = np.dot(query_embedding, window_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(window_embedding))
                    
                    if similarity > 0.7:
                        results.append({
                            'content': window_text[:500],  # Truncate for preview
                            'relevance_score': float(similarity),
                            'source': f'active_window_{i}',
                            'position': context.current_position
                        })
                        
        return results
        
    async def _search_compressed_context(self, context: StreamingContext, query: str) -> List[Dict[str, Any]]:
        """Search within compressed context representations."""
        results = []
        
        # Search through different compression levels
        for checkpoint_pos, checkpoint_data in context.context_checkpoints.items():
            if 'summary' in checkpoint_data:
                summary_embedding = await self._generate_embedding(checkpoint_data['summary'])
                query_embedding = await self._generate_embedding(query)
                
                if summary_embedding and query_embedding:
                    similarity = np.dot(query_embedding, summary_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(summary_embedding))
                    
                    if similarity > 0.6:
                        results.append({
                            'content': checkpoint_data['summary'],
                            'relevance_score': float(similarity),
                            'source': f'checkpoint_{checkpoint_pos}',
                            'position': checkpoint_pos
                        })
                        
        return results
        
    async def _search_hierarchical_memory(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """Search within the hierarchical memory system."""
        results = await self.hierarchical_memory.search(
            query=query,
            session_id=session_id,
            limit=20
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.get('content', ''),
                'relevance_score': result.get('similarity', 0.5),
                'source': f"hierarchical_{result.get('tier', 'unknown')}",
                'position': result.get('metadata', {}).get('position', -1)
            })
            
        return formatted_results
        
    async def _search_checkpoints(self, context: StreamingContext, query: str) -> List[Dict[str, Any]]:
        """Search through context checkpoints."""
        # This is similar to compressed context search but with more detail
        return await self._search_compressed_context(context, query)
        
    async def _compress_result(self, result: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """Compress a result to fit within token limit."""
        content = result.get('content', '')
        if not content:
            return None
            
        # Use NLP processor to create a summary
        summary = self.nlp_processor.summarize(content, ratio=max_tokens/len(content.split()))
        
        if summary:
            result['content'] = summary
            result['compressed'] = True
            return result
            
        return None
        
    async def _generate_context_summary(self, contexts: List[Dict[str, Any]]) -> str:
        """Generate a summary of the selected contexts."""
        if not contexts:
            return "No relevant context found."
            
        # Combine all context content
        all_content = ' '.join([ctx.get('content', '') for ctx in contexts])
        
        # Generate summary
        summary = self.nlp_processor.summarize(all_content, ratio=0.2)
        
        return summary or "Context available but summary generation failed."
        
    async def _get_global_context_summary(self, session_id: str) -> str:
        """Get a high-level summary of the entire session context."""
        # This would retrieve and summarize key points from the entire session
        # For now, return a placeholder
        return f"Session {session_id} with extensive context across multiple scales."
        
    async def _get_session_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """Get memory statistics for a session."""
        # This would gather detailed stats from the hierarchical memory
        return {
            "total_memories": 0,  # Would query the database
            "memory_tiers": {
                "active": 0,
                "recent": 0,
                "long_term": 0,
                "archive": 0
            },
            "total_embeddings": 0,
            "compression_ratio": 0.0
        }
        
    def _apply_attention_pattern(self, text: str, context: StreamingContext) -> float:
        """Apply learned attention patterns to adjust importance scoring."""
        if context.session_id not in self.attention_patterns:
            return 0.5
            
        # Simple pattern matching - in production would use more sophisticated methods
        pattern = self.attention_patterns[context.session_id]
        text_features = self._extract_text_features(text)
        
        # Compute weighted score based on pattern
        score = np.dot(pattern[:len(text_features)], text_features) / (np.linalg.norm(pattern[:len(text_features)]) * np.linalg.norm(text_features) + 1e-8)
        
        return float(np.clip(score, 0, 1))
        
    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract simple features from text for pattern matching."""
        features = np.zeros(100)
        
        # Simple feature extraction - in production would use more sophisticated methods
        text_lower = text.lower()
        
        # Feature 0-9: Common important words
        important_words = ['important', 'key', 'critical', 'remember', 'note', 'summary', 'conclusion', 'question', 'answer', 'define']
        for i, word in enumerate(important_words[:10]):
            features[i] = 1.0 if word in text_lower else 0.0
            
        # Feature 10-19: Punctuation and structure
        features[10] = text.count('?') / max(len(text), 1)
        features[11] = text.count('!') / max(len(text), 1)
        features[12] = text.count('\n') / max(len(text), 1)
        
        # Feature 20-29: Length indicators
        features[20] = min(len(text) / 1000, 1.0)  # Normalized length
        features[21] = min(len(text.split()) / 100, 1.0)  # Word count
        
        return features