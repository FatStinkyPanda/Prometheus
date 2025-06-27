# backend/core/consciousness/enhanced_infinite_context.py

import asyncio
import hashlib
import math
import pickle
import lz4.frame
import zstandard as zstd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Callable, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import sys 
import os # For thread_pool_workers default

from backend.memory.hierarchical_memory import HierarchicalMemory
from backend.utils.logger import Logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.io_systems.natural_language_processor import NaturalLanguageProcessor
    from backend.core.minds.base_mind import BaseMind


class CompressionLevel(Enum):
    NONE = 0
    LIGHT = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXTREME = 4
    NEURAL = 5

@dataclass
class TokenBlock:
    block_id: str
    tokens: List[str] 
    embeddings: Optional[np.ndarray] 
    timestamp: datetime
    importance_scores: np.ndarray 
    access_count: int = 0
    compression_level: CompressionLevel = CompressionLevel.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    children_blocks: List[str] = field(default_factory=list)
    parent_block: Optional[str] = None
    semantic_links: Dict[str, float] = field(default_factory=dict)


class LearnedCompressionModel(nn.Module):
    def __init__(self, input_dim: int = 768, compression_ratio: float = 0.1, num_layers: int = 4):
        super().__init__()
        self.logger = Logger(self.__class__.__name__)
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.compressed_dim = int(input_dim * compression_ratio)
        if self.compressed_dim <= 0:
            self.logger.warning(f"Calculated compressed_dim is {self.compressed_dim} (<=0) with input_dim {input_dim} and ratio {compression_ratio}. Setting to 1.")
            self.compressed_dim = 1
        self.num_layers = num_layers
        
        encoder_layers = []
        current_dim = input_dim
        for i in range(num_layers):
            next_dim = int(current_dim * 0.7) if i < num_layers - 1 else self.compressed_dim
            if next_dim <= 0: next_dim = 1 
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        current_dim = self.compressed_dim
        for i in range(num_layers):
            next_dim = int(current_dim / 0.7) if i < num_layers - 1 else input_dim
            if next_dim <=0 : next_dim = input_dim 
            decoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(0.1) if i < num_layers - 1 else nn.Identity()
            ])
            current_dim = next_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.importance_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.importance_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        
        self.compression_selector = nn.Sequential(
            nn.Linear(input_dim * 2, 256), nn.ReLU(), 
            nn.Linear(256, len(CompressionLevel)), nn.Softmax(dim=-1)
        )
        self.logger.info(f"LearnedCompressionModel initialized. Input: {input_dim}, Compressed: {self.compressed_dim}, Layers: {num_layers}")

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim == 1: 
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2: 
            x = x.unsqueeze(1)

        attn_output, _ = self.importance_attention(x, x, x) 
        importance = self.importance_head(attn_output)      
        
        compressed = self.encoder(x)      
        reconstructed = self.decoder(compressed) 
        
        x_mean = x.mean(dim=1) 
        if context is not None:
            if context.ndim == 1: context = context.unsqueeze(0) 
            if context.ndim == 3: context_mean = context.mean(dim=1) 
            else: context_mean = context 
            
            if x_mean.shape[0] != context_mean.shape[0] and context_mean.shape[0] == 1:
                context_mean = context_mean.expand(x_mean.shape[0], -1)
            elif x_mean.shape[0] != context_mean.shape[0]:
                 self.logger.error(f"Batch size mismatch for compression_selector: x_mean {x_mean.shape}, context_mean {context_mean.shape}")
                 context_mean = x_mean.clone()

            combined = torch.cat([x_mean, context_mean], dim=-1)
        else:
            combined = torch.cat([x_mean, x_mean.clone()], dim=-1)
            
        compression_strategy = self.compression_selector(combined) 
        
        return compressed, reconstructed, importance.squeeze(-1), compression_strategy


class AdaptiveTokenBuffer:
    def __init__(self, base_size: int = 10000, adaptation_rate: float = 0.1):
        self.logger = Logger(self.__class__.__name__)
        self.base_size = base_size
        self.current_size = base_size
        self.adaptation_rate = adaptation_rate
        self.buffer: deque[str] = deque()
        self.metadata_buffer: deque[Optional[Dict]] = deque()
        self.pattern_detector = PatternDetector()
        self.buffer_stats = {
            'total_flushes': 0, 'forced_flushes': 0, 'pattern_flushes': 0,
            'avg_utilization': 0.0, 'current_token_count': 0
        }
        self.logger.info(f"AdaptiveTokenBuffer initialized. Base size: {base_size}, Adaptation rate: {adaptation_rate}")

    async def add_tokens(self, tokens: List[str], metadata: Optional[Dict] = None) -> Optional[TokenBlock]:
        self.buffer.extend(tokens)
        for _ in range(len(tokens)): self.metadata_buffer.append(metadata) 
        self.buffer_stats['current_token_count'] = len(self.buffer)

        if self.pattern_detector.should_flush(tokens):
            self.buffer_stats['pattern_flushes'] += 1
            return await self._create_and_flush()
            
        if len(self.buffer) >= self.current_size:
            self.buffer_stats['forced_flushes'] += 1
            return await self._create_and_flush()
        return None
        
    async def _create_and_flush(self) -> TokenBlock:
        block = await self._create_token_block()
        
        utilization = len(self.buffer) / self.current_size if self.current_size > 0 else 1.0
        self.buffer_stats['avg_utilization'] = (
            self.buffer_stats['avg_utilization'] * 0.9 + utilization * 0.1
        )
        
        if self.buffer_stats['avg_utilization'] > 0.9:
            self.current_size = int(self.current_size * (1 + self.adaptation_rate))
        elif self.buffer_stats['avg_utilization'] < 0.5:
            self.current_size = max(
                self.base_size // 2, int(self.current_size * (1 - self.adaptation_rate))
            )
        
        self.buffer_stats['total_flushes'] += 1
        self.buffer.clear()
        self.metadata_buffer.clear()
        self.buffer_stats['current_token_count'] = 0
        return block
        
    async def _create_token_block(self) -> TokenBlock:
        tokens_list = list(self.buffer)
        block_metadata_list = [m for m in list(self.metadata_buffer) if m is not None]
        representative_metadata = block_metadata_list[0] if block_metadata_list else {}

        content_str = "".join(tokens_list)
        content_hash = hashlib.sha256(content_str[:min(len(content_str), 1000)].encode('utf-8', 'ignore')).hexdigest()[:16]
        # Use a more collision-resistant timestamp part for block_id
        timestamp_nanos = time.time_ns()
        block_id = f"TB_{content_hash}_{timestamp_nanos}"
        
        importance_scores = self._calculate_importance_with_patterns(tokens_list)
        semantic_boundaries = self._detect_semantic_boundaries(tokens_list)
        
        if len(importance_scores) != len(tokens_list):
            self.logger.warning(f"Token count ({len(tokens_list)}) and importance scores ({len(importance_scores)}) mismatch for block {block_id}. Defaulting scores.")
            importance_scores = np.ones(len(tokens_list)) * 0.5

        block = TokenBlock(
            block_id=block_id, tokens=tokens_list, embeddings=None,
            timestamp=datetime.utcnow(), importance_scores=importance_scores,
            metadata={**representative_metadata, 'semantic_boundaries': semantic_boundaries, 'buffer_stats_at_creation': dict(self.buffer_stats)}
        )
        return block
        
    def _calculate_importance_with_patterns(self, tokens: List[str]) -> np.ndarray:
        if not tokens: return np.array([])
        base_scores = np.ones(len(tokens)) * 0.5
        for i, token_text in enumerate(tokens):
            if token_text in ['?', 'what', 'why', 'how', 'when', 'where', 'who']: base_scores[i] *= 1.5
            if token_text.lower() in ['important', 'critical', 'urgent', 'remember', 'note', 'key']: base_scores[i] *= 1.8
            if token_text in ['.', '!', ':', ';', '\n', '\n\n']: base_scores[i] *= 1.2
            if token_text and token_text[0].isupper() and i > 0 and tokens[i-1] not in ['.', '!', '?']: base_scores[i] *= 1.3
        if len(tokens) >= 5:
            smoothed_scores = np.convolve(base_scores, np.ones(5)/5, mode='same')
            return np.clip(smoothed_scores, 0.1, 1.0)
        return np.clip(base_scores, 0.1, 1.0)
        
    def _detect_semantic_boundaries(self, tokens: List[str]) -> List[int]:
        boundaries = []
        for i, token_text in enumerate(tokens):
            if token_text == '\n\n': boundaries.append(i)
            elif token_text in ['.', '!', '?'] and i < len(tokens) - 1: boundaries.append(i + 1)
        return boundaries


class PatternDetector: 
    def __init__(self): self.logger = Logger(self.__class__.__name__)
    def should_flush(self, tokens: List[str]) -> bool: 
        if not tokens: return False
        # More robust check for question mark, ensuring it's not part of a URL or similar
        if any(t == '?' for t in tokens if len(t) == 1) and tokens[-1] != '?': return True
        
        # Check for multiple paragraph endings more reliably
        newline_count = 0
        for i in range(max(0, len(tokens)-20), len(tokens)-1): # Check last 20 tokens
            if tokens[i] == '\n' and tokens[i+1] == '\n':
                newline_count +=1
        if newline_count >=2: return True
        return False


class StreamingOutputGenerator:
    def __init__(self, context_manager: 'EnhancedInfiniteContextManager', 
                 generation_model: 'BaseMind', # Type hint for clarity
                 chunk_size: int = 512):
        self.context_manager = context_manager
        self.generation_model = generation_model 
        self.chunk_size = chunk_size 
        self.logger = Logger(self.__class__.__name__)

    async def generate_unlimited_stream(
        self, prompt: str, session_id: str, max_tokens: Optional[int] = None,
        temperature: float = 0.7, stop_callback: Optional[Callable[[], bool]] = None,
        **kwargs 
    ) -> AsyncGenerator[str, None]:
        self.logger.info(f"StreamingOutputGenerator called for session {session_id}. Prompt: '{prompt[:50]}...' Max_tokens: {max_tokens}")
        
        total_generated_tokens = 0
        current_generated_text = "" 
        
        # Use the context manager's nlp_processor for tokenization
        nlp_proc = self.context_manager.nlp_processor
        if not nlp_proc or not hasattr(nlp_proc, 'nlp') or not hasattr(nlp_proc.nlp, 'tokenizer'):
            self.logger.error("NLP processor not available for token counting in StreamingOutputGenerator.")
            # Fallback token counting
            def fallback_tokenizer_len(text): return len(text.split())
            tokenizer_len = fallback_tokenizer_len
        else:
            def nlp_tokenizer_len(text): return len(list(nlp_proc.nlp.tokenizer(text)))
            tokenizer_len = nlp_tokenizer_len

        initial_retrieved_context_data = await self.context_manager.retrieve_context(
            query=prompt, session_id=session_id, max_tokens=self.context_manager.config.get('generation_initial_context_tokens', 4096)
        )
        initial_context_str = self._format_retrieved_context_for_generation(initial_retrieved_context_data)
        
        current_prompt_for_model = f"{initial_context_str}\n\n{prompt}\n\nAssistant:" # Common instruction style

        while True:
            if stop_callback and stop_callback():
                self.logger.info(f"Stop callback triggered for session {session_id}.")
                break
            if max_tokens is not None and total_generated_tokens >= max_tokens:
                self.logger.info(f"Max tokens ({max_tokens}) reached for session {session_id}.")
                break

            tokens_to_generate_this_call = self.chunk_size
            if max_tokens is not None:
                tokens_to_generate_this_call = min(self.chunk_size, max_tokens - total_generated_tokens)
            
            if tokens_to_generate_this_call <= 0: break 

            try:
                gen_input_data = {
                    'text': current_prompt_for_model,
                    'logical_mind_instance': self.context_manager.embedding_model, # Pass the embedder
                    'temperature': temperature,
                    'max_new_tokens': tokens_to_generate_this_call
                }
                # Merge kwargs for additional generation parameters
                gen_input_data.update(kwargs)
                
                if hasattr(self.generation_model, 'process') and asyncio.iscoroutinefunction(self.generation_model.process):
                    response_dict = await self.generation_model.process(gen_input_data)
                    new_chunk = response_dict.get('payload', {}).get('generated_text', '')
                else: 
                    self.logger.warning(f"Generation model {type(self.generation_model)} does not have an async process method. Simulating.")
                    new_chunk = f"[Simulated chunk for: {current_prompt_for_model[-50:]}...] "
                    await asyncio.sleep(0.1) 

                if not new_chunk: 
                    self.logger.info(f"Generation model returned empty chunk for session {session_id}. Ending generation.")
                    break
                
                # Handle stop sequences
                stop_sequences = kwargs.get('stop_sequences', [])
                if stop_sequences:
                    original_chunk_len = len(new_chunk)
                    for seq in stop_sequences:
                        if seq in new_chunk:
                            new_chunk = new_chunk.split(seq, 1)[0] 
                            yield new_chunk 
                            self.logger.info(f"Stop sequence '{seq}' encountered for session {session_id}.")
                            total_generated_tokens += tokenizer_len(new_chunk)
                            current_generated_text += new_chunk
                            await self.context_manager.process_generated_output(current_generated_text, session_id, {"status": "stopped_by_sequence", "final_chunk_len_before_stop": original_chunk_len})
                            return 
                
                yield new_chunk
                
                num_new_tokens = tokenizer_len(new_chunk)
                total_generated_tokens += num_new_tokens
                current_generated_text += new_chunk
                
                await self.context_manager.process_generated_output(new_chunk, session_id, {"chunk_num": total_generated_tokens // self.chunk_size if self.chunk_size > 0 else total_generated_tokens })
                
                # Prepare prompt for the next iteration: include recent generation and fresh context
                # Query based on the tail of the *entire* generated text so far for better coherence
                query_for_next_context = current_generated_text[-1000:] # Use a larger slice of recent generation
                
                retrieved_context_data = await self.context_manager.retrieve_context(
                    query=query_for_next_context, 
                    session_id=session_id, 
                    max_tokens=self.context_manager.config.get('generation_step_context_tokens', 2048)
                )
                retrieved_context_str = self._format_retrieved_context_for_generation(retrieved_context_data)
                current_prompt_for_model = f"{retrieved_context_str}\n\n{current_generated_text[-1500:]}\n\nAssistant:" # Continue from more substantial recent generated text

            except Exception as e:
                self.logger.error(f"Error during streaming generation for session {session_id}: {e}", exc_info=True)
                yield f"[Error in generation: {str(e)}]"
                break
        
        await self.context_manager.process_generated_output(current_generated_text, session_id, {"status": "generation_complete", "total_tokens": total_generated_tokens})
        self.logger.info(f"StreamingOutputGenerator finished for session {session_id}. Total tokens: {total_generated_tokens}")

    def _format_retrieved_context_for_generation(self, retrieved_context_data: Dict[str, Any]) -> str:
        parts = []
        if retrieved_context_data.get("context_summary"):
            parts.append(f"Summary of prior context:\n{retrieved_context_data['context_summary']}")
        
        key_info_parts = []
        # Use more context parts if available, and ensure 'text' key exists
        for part_dict in retrieved_context_data.get("context_parts", [])[:self.context_manager.config.get('generation_context_parts_limit', 3)]: 
            if isinstance(part_dict, dict) and part_dict.get("text"): # Check if part_dict is dict
                key_info_parts.append(str(part_dict["text"])[:self.context_manager.config.get('generation_context_part_truncate', 300)] + "...") 
        if key_info_parts:
            parts.append("Key relevant information from context:\n" + "\n".join(f"- {p}" for p in key_info_parts))
        
        return "\n\n".join(parts) if parts else "No specific prior context retrieved."


class HierarchicalAttentionMechanism:
    def __init__(self, num_scales: int = 5, base_window_size: int = 1024, learning_rate: float = 0.01):
        self.logger = Logger(self.__class__.__name__)
        self.num_scales = max(1, num_scales) # Ensure at least one scale
        self.base_window_size = max(128, base_window_size) # Ensure minimum base window size
        self.attention_weights: Dict[str, float] = {} 
        self.scale_importance = np.ones(self.num_scales) / self.num_scales
        self.attention_cache: Dict[str, Dict[str, Any]] = {}
        self.learning_rate = learning_rate
        self.logger.info(f"HierarchicalAttentionMechanism initialized. Scales: {self.num_scales}, Base window: {self.base_window_size}, LR: {self.learning_rate}")

    async def compute_attention(
        self, query_embedding: np.ndarray, token_blocks: List[TokenBlock],
        max_tokens: int, session_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TokenBlock, float]]:
        self.logger.debug(f"Computing attention for query over {len(token_blocks)} blocks, max_tokens: {max_tokens}")
        if query_embedding is None or query_embedding.size == 0:
            self.logger.warning("Query embedding is None or empty in compute_attention.")
            return []
        if not token_blocks: return []

        cache_key = self._get_cache_key(query_embedding, len(token_blocks))
        cached_data = self.attention_cache.get(cache_key)
        if cached_data and self._is_cache_valid(cached_data):
            self.logger.debug(f"Returning cached attention result for key: {cache_key}")
            return cached_data['result']

        attended_blocks_scores: Dict[str, float] = {} 

        for scale_idx in range(self.num_scales):
            window_duration_seconds_for_scale = self.base_window_size * (2 ** scale_idx) # Interpret as duration if blocks are time-based
            grouped_windows = self._group_by_time_window(token_blocks, window_duration_seconds_for_scale)

            for window_group in grouped_windows:
                if not window_group: continue
                window_features = await self._compute_window_features(window_group)
                if not window_features or window_features.get('embedding') is None: continue

                window_relevance = self._compute_learned_attention(
                    query_embedding, window_features, scale_idx, session_context
                )
                weighted_relevance = window_relevance * self.scale_importance[scale_idx]
                temporal_decay = self._compute_temporal_decay(window_group, scale_idx)
                final_window_score = weighted_relevance * temporal_decay

                for block in window_group:
                    block_score_in_window = self._compute_block_score(block, final_window_score, window_features)
                    if block_score_in_window > attended_blocks_scores.get(block.block_id, -1.0):
                         attended_blocks_scores[block.block_id] = block_score_in_window
        
        all_attended_blocks = []
        block_map = {b.block_id: b for b in token_blocks}
        for block_id, score in attended_blocks_scores.items():
            if block_id in block_map:
                 all_attended_blocks.append((block_map[block_id], score))
        
        all_attended_blocks.sort(key=lambda x: x[1], reverse=True)
        selected_blocks = self._select_diverse_blocks(all_attended_blocks, max_tokens)
        
        self.attention_cache[cache_key] = {'result': selected_blocks, 'timestamp': datetime.utcnow()}
        if self.learning_rate > 0: 
            asyncio.create_task(self._update_attention_weights(query_embedding, selected_blocks, session_context))
        return selected_blocks

    def _compute_learned_attention(self, query_emb: np.ndarray, window_features: Dict[str, Any], scale: int, context: Optional[Dict[str, Any]]) -> float:
        win_emb_val = window_features.get('embedding')
        if win_emb_val is None or query_emb is None or not isinstance(win_emb_val, np.ndarray) or not isinstance(query_emb, np.ndarray): 
            return 0.0
        
        win_emb = win_emb_val # Already an ndarray from _compute_window_features
        if query_emb.ndim == 1 and win_emb.ndim > 1: win_emb = win_emb.mean(axis=0) # Should not happen if win_emb is avg
        
        if query_emb.shape != win_emb.shape:
             self.logger.warning(f"Shape mismatch in _compute_learned_attention: query {query_emb.shape}, window {win_emb.shape}. Scale {scale}.")
             return 0.0
        
        norm_q = np.linalg.norm(query_emb)
        norm_w = np.linalg.norm(win_emb)
        if norm_q == 0 or norm_w == 0: return 0.0
        base_similarity = np.dot(query_emb, win_emb) / (norm_q * norm_w)
        
        # Placeholder for more complex learned attention adjustment
        scale_specific_weight = self.attention_weights.get(f"scale_{scale}_weight", 1.0) # Example
        adjusted_similarity = base_similarity * scale_specific_weight

        return float(np.clip(adjusted_similarity, 0, 1))

    def _select_diverse_blocks(self, attended_blocks: List[Tuple[TokenBlock, float]], max_tokens: int) -> List[Tuple[TokenBlock, float]]:
        selected: List[Tuple[TokenBlock, float]] = []
        selected_embeddings_means: List[np.ndarray] = []
        current_tokens = 0
        for block, score in attended_blocks:
            if not block.tokens: continue 
            block_token_count = len(block.tokens)
            if current_tokens + block_token_count > max_tokens and selected: break 

            is_diverse = True
            if block.embeddings is not None and block.embeddings.size > 0:
                current_block_mean_emb = block.embeddings.mean(axis=0) if block.embeddings.ndim > 1 else block.embeddings
                if current_block_mean_emb.ndim != 1 : # Ensure it's 1D
                    self.logger.warning(f"Block {block.block_id} embedding for diversity check is not 1D. Shape: {current_block_mean_emb.shape}")
                else:
                    for sel_emb_mean in selected_embeddings_means:
                        if sel_emb_mean.ndim != 1: continue # Skip if selected embedding is not 1D
                        norm_curr = np.linalg.norm(current_block_mean_emb)
                        norm_sel = np.linalg.norm(sel_emb_mean)
                        if norm_curr == 0 or norm_sel == 0 : continue
                        similarity = np.dot(current_block_mean_emb, sel_emb_mean) / (norm_curr * norm_sel)
                        if similarity > 0.95: 
                            is_diverse = False; break
            
            if is_diverse:
                if current_tokens + block_token_count <= max_tokens : 
                    selected.append((block, score))
                    if block.embeddings is not None and block.embeddings.size > 0:
                        mean_emb_to_add = block.embeddings.mean(axis=0) if block.embeddings.ndim > 1 else block.embeddings
                        if mean_emb_to_add.ndim == 1: # Add only if 1D
                            selected_embeddings_means.append(mean_emb_to_add)
                    current_tokens += block_token_count
        return selected

    async def _update_attention_weights(self, query_embedding: np.ndarray, selected_blocks: List[Tuple[TokenBlock, float]], context: Optional[Dict[str, Any]]):
        if not selected_blocks or self.scale_importance.size == 0: return
        
        scale_contributions = defaultdict(float)
        total_score_sum = 0.0
        for block, score in selected_blocks:
            scale_idx = self._determine_block_scale(block) # Assumes this returns valid index for self.num_scales
            if 0 <= scale_idx < self.num_scales:
                scale_contributions[scale_idx] += score
                total_score_sum += score
        
        if total_score_sum > 0:
            for i in range(self.num_scales):
                normalized_contribution = scale_contributions.get(i, 0.0) / total_score_sum
                self.scale_importance[i] = self.scale_importance[i] * (1 - self.learning_rate) + normalized_contribution * self.learning_rate
            if np.sum(self.scale_importance) > 0: self.scale_importance /= np.sum(self.scale_importance)


    def _compute_temporal_decay(self, window: List[TokenBlock], scale: int) -> float:
        if not window: return 1.0
        now = datetime.utcnow()
        avg_age_seconds = sum((now - b.timestamp).total_seconds() for b in window) / len(window)
        # More aggressive decay for recent scales (lower scale index), less for older scales
        # Half-life in hours: scale 0: 6h, scale 1: 12h, scale 2: 24h, scale 3: 48h, scale 4: 96h
        half_life_hours = 6 * (2 ** scale) 
        decay_rate = math.log(2) / (half_life_hours * 3600) if half_life_hours > 0 else float('inf')
        if decay_rate == float('inf'): return 0.0
        return math.exp(-decay_rate * avg_age_seconds)

    def _compute_block_score(self, block: TokenBlock, window_score: float, window_features: Dict[str, Any]) -> float:
        score = window_score
        if block.importance_scores is not None and block.importance_scores.size > 0: # Check not None and not empty
            importance_factor = np.mean(block.importance_scores)
            score *= (0.5 + 0.5 * importance_factor) 
        access_factor = math.log10(block.access_count + 1) / 2.0 
        score *= (1 + min(access_factor, 0.5)) # Cap access bonus
        return float(np.clip(score, 0, 1))

    async def _compute_window_features(self, window: List[TokenBlock]) -> Optional[Dict[str, Any]]:
        if not window: return None
        
        # Filter for blocks that have valid embeddings
        valid_block_embeddings = [b.embeddings for b in window if b.embeddings is not None and b.embeddings.size > 0]
        if not valid_block_embeddings: return None # If no blocks have embeddings, can't compute window embedding

        flat_embeddings = []
        for emb_array in valid_block_embeddings:
            if emb_array.ndim == 2 and emb_array.shape[0] == 1: flat_embeddings.append(emb_array.flatten())
            elif emb_array.ndim == 1: flat_embeddings.append(emb_array)
        
        if not flat_embeddings or any(fe.ndim !=1 for fe in flat_embeddings):
            self.logger.warning("Could not form valid 1D embeddings for window feature calculation.")
            return None
        
        avg_embedding = np.mean(np.array(flat_embeddings, dtype=np.float32), axis=0)
        
        all_importance_scores = np.concatenate([b.importance_scores for b in window if b.importance_scores is not None and b.importance_scores.size > 0])
        avg_importance = np.mean(all_importance_scores) if all_importance_scores.size > 0 else 0.5
        
        # Placeholder for entities and topics - would require NLP processing on window content
        entities_in_window = set()
        topics_in_window = [] # Could be list of topic strings or embeddings

        return {'embedding': avg_embedding, 'importance': avg_importance, 
                'num_blocks_in_window': len(window), 'entities': list(entities_in_window), 'topics': topics_in_window}

    def _group_by_time_window(self, blocks: List[TokenBlock], window_duration_seconds: int) -> List[List[TokenBlock]]:
        if not blocks: return []
        # Ensure blocks are sorted by timestamp for correct windowing
        sorted_blocks = sorted(blocks, key=lambda b: b.timestamp)
        
        windows: List[List[TokenBlock]] = []
        if not sorted_blocks: return windows # Should be caught by `if not blocks`
        
        current_window_list: List[TokenBlock] = []
        if not sorted_blocks: return windows

        # Initialize with the first block's timestamp
        current_window_start_time = sorted_blocks[0].timestamp
        
        for block in sorted_blocks:
            if (block.timestamp - current_window_start_time).total_seconds() < window_duration_seconds:
                current_window_list.append(block)
            else:
                if current_window_list: windows.append(current_window_list)
                current_window_list = [block] # Start new window with current block
                current_window_start_time = block.timestamp
        if current_window_list: windows.append(current_window_list) # Add the last window
        return windows

    def _get_cache_key(self, query_embedding: np.ndarray, num_blocks: int) -> str: 
        # Ensure query_embedding is contiguous for tobytes()
        if not query_embedding.flags.c_contiguous:
            query_embedding = np.ascontiguousarray(query_embedding)
        return hashlib.md5(query_embedding.tobytes() + str(num_blocks).encode()).hexdigest()

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool: 
        cache_ttl = self.context_manager.config.get('attention_mechanism', {}).get('cache_ttl_seconds', 300) # Get from config
        return (datetime.utcnow() - cached_result.get('timestamp', datetime.min)).total_seconds() < cache_ttl

    def _determine_block_scale(self, block: TokenBlock) -> int:
        age_seconds = (datetime.utcnow() - block.timestamp).total_seconds()
        # Define scale boundaries (in seconds) - these could be configurable
        # Scale 0: < 1 hour (3600s)
        # Scale 1: < 1 day (86400s)
        # Scale 2: < 1 week (604800s)
        # Scale 3: < 1 month (approx 2.6M s)
        # Scale 4: Older
        scale_boundaries = [3600, 86400, 604800, 2592000] # Example
        for i, boundary in enumerate(scale_boundaries):
            if age_seconds < boundary:
                return i
        return len(scale_boundaries) # Max scale index


class SemanticGraph: 
    def __init__(self): 
        self.nodes: Dict[str, Dict[str, Any]] = {} 
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict) 
        self.logger = Logger(self.__class__.__name__)
    def add_node(self, node_id: str, embedding: Optional[np.ndarray]): 
        if embedding is None: 
            self.logger.warning(f"Attempted to add node {node_id} to SemanticGraph with None embedding.")
            return
        self.nodes[node_id] = {'embedding': embedding, 'created_at': datetime.utcnow()}
    def add_edge(self, from_id: str, to_id: str, weight: float): 
        self.edges[from_id][to_id] = weight
        self.edges[to_id][from_id] = weight
    def prune_weak_edges(self, threshold: float = 0.3): 
        self.logger.debug(f"Pruning weak edges below threshold {threshold}...")
        removed_count = 0
        # Iterate over copies of keys to allow modification during iteration
        for u in list(self.edges.keys()):
            if u not in self.edges: continue # Check if u was removed by an earlier iteration
            for v in list(self.edges[u].keys()):
                if v not in self.edges[u]: continue # Check if v was removed
                if self.edges[u][v] < threshold:
                    del self.edges[u][v]
                    if v in self.edges and u in self.edges[v]: # Check if reverse edge exists before deleting
                        del self.edges[v][u]
                    removed_count +=1
        self.logger.debug(f"Pruned {removed_count} weak edges.")


@dataclass
class SessionState: 
    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    tokens_processed: int = 0
    tokens_generated: int = 0 
    blocks_created: int = 0
    recent_embeddings: deque = field(default_factory=lambda: deque(maxlen=20)) 
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    recent_topics: deque = field(default_factory=lambda: deque(maxlen=10)) 

    def to_dict(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "start_time_iso": self.start_time.isoformat(),
                "tokens_processed": self.tokens_processed, "tokens_generated": self.tokens_generated,
                "blocks_created": self.blocks_created, "recent_topics_list": list(self.recent_topics)}


class EnhancedInfiniteContextManager:
    def __init__(
        self,
        hierarchical_memory: HierarchicalMemory,
        nlp_processor: 'NaturalLanguageProcessor',
        embedding_model: 'BaseMind',
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = Logger(__name__)
        self.hierarchical_memory = hierarchical_memory
        self.nlp_processor = nlp_processor
        self.embedding_model = embedding_model
        self.config = config if isinstance(config, dict) else {} 

        self.max_active_blocks = self.config.get('max_active_blocks', 1000)
        self.compression_threshold_score = self.config.get('compression_threshold', 0.7)
        self.batch_size_learning = self.config.get('batch_size', 32)
        self.checkpoint_interval_tokens = self.config.get('checkpoint_interval', 10000)

        comp_model_config = self.config.get('compression_model', {})
        embedder_output_dim = 768 
        if self.embedding_model and hasattr(self.embedding_model, 'model') and self.embedding_model.model and \
           hasattr(self.embedding_model.model, 'config') and hasattr(self.embedding_model.model.config, 'hidden_size'):
            embedder_output_dim = self.embedding_model.model.config.hidden_size
        else:
            self.logger.warning(f"Could not determine embedding_model output dimension for LearnedCompressionModel. Defaulting to {embedder_output_dim}.")

        self.compression_model = LearnedCompressionModel(
            input_dim=embedder_output_dim,
            compression_ratio=comp_model_config.get('compression_ratio', 0.1),
            num_layers=comp_model_config.get('num_layers', 4)
        )
        
        attn_mech_config = self.config.get('attention_mechanism', {})
        self.attention_mechanism = HierarchicalAttentionMechanism(
            num_scales=attn_mech_config.get('num_scales', 5),
            base_window_size=attn_mech_config.get('base_window_size', 1024),
            learning_rate=attn_mech_config.get('learning_rate', 0.01)
        )
        # Pass self to HierarchicalAttentionMechanism if _is_cache_valid needs config from self.config
        self.attention_mechanism.context_manager = self 


        buffer_config = self.config.get('adaptive_buffer', {})
        self.adaptive_buffer = AdaptiveTokenBuffer(
            base_size=buffer_config.get('base_size', 10000),
            adaptation_rate=buffer_config.get('adaptation_rate', 0.05)
        )
        
        self.output_generator = StreamingOutputGenerator(self, self.embedding_model)
        
        self.active_blocks: Dict[str, TokenBlock] = {}
        self.compressed_blocks: Dict[str, bytes] = {}
        self.block_index: Dict[str, Dict[str, Any]] = {}
        self.semantic_graph = SemanticGraph()
        
        self.optimizer = torch.optim.AdamW(
            self.compression_model.parameters(),
            lr=comp_model_config.get('learning_rate', 0.0005),
            weight_decay=0.01
        )
        self.learning_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        self.learning_enabled = self.config.get('enable_learning', True)
        self.training_buffer: deque[Dict[str, Any]] = deque(maxlen=max(1, self.batch_size_learning * 10))
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get('thread_pool_workers', os.cpu_count() or 1))
        self.compression_cache: Dict[str, bytes] = {}
        
        self.stats = defaultdict(float)
        self.stats.update({
            'total_tokens_processed': 0, 'total_tokens_generated': 0,
            'total_blocks_created': 0, 'compression_ratio': 1.0,
            'retrieval_accuracy': 0.0, 'learning_iterations': 0,
            'avg_block_importance': 0.5
        })
        self.active_sessions: Dict[str, SessionState] = {}
        
        self.logger.info(f"EnhancedInfiniteContextManager initialized. Max active blocks: {self.max_active_blocks}, Compression threshold score: {self.compression_threshold_score}")

    async def create_streaming_session(self, session_id: str) -> SessionState:
        return self._get_or_create_session(session_id)

    async def process_unlimited_stream(
        self, token_stream: AsyncGenerator[List[str], None], session_id: str,
        callback: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
        real_time: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        session_state = self._get_or_create_session(session_id)
        try:
            async for tokens_list in token_stream:
                if not isinstance(tokens_list, list) or not all(isinstance(t, str) for t in tokens_list):
                    self.logger.warning(f"Skipping invalid token data in stream for session {session_id}: {type(tokens_list)}")
                    continue
                
                session_state.tokens_processed += len(tokens_list)
                self.stats['total_tokens_processed'] = float(self.stats['total_tokens_processed']) + len(tokens_list)
                
                block = await self.adaptive_buffer.add_tokens(tokens_list, metadata={'session_id': session_id, 'real_time': real_time})
                if block:
                    processed_block = await self._process_token_block(block, session_id)
                    session_state.blocks_created += 1
                    storage_decision = await self._determine_storage_strategy(processed_block, session_state)
                    await self._execute_storage_strategy(processed_block, storage_decision)
                    await self._create_semantic_links(processed_block)
                    if self.learning_enabled: asyncio.create_task(self._learn_from_block(processed_block))
                    
                    current_importance = 0.0
                    if processed_block.importance_scores is not None and processed_block.importance_scores.size > 0:
                        current_importance = float(np.mean(processed_block.importance_scores))

                    update_payload = {
                        'type': 'block_processed', 'block_id': processed_block.block_id,
                        'tokens': len(processed_block.tokens), 
                        'importance': current_importance,
                        'compression_level': processed_block.compression_level.name,
                        'session_stats': session_state.to_dict(),
                        'semantic_links_count': len(processed_block.semantic_links)
                    }
                    yield update_payload
                    if callback: await callback(session_state.to_dict())
                        
                if self.checkpoint_interval_tokens > 0 and \
                   session_state.tokens_processed >= self.checkpoint_interval_tokens and \
                   session_state.tokens_processed // self.checkpoint_interval_tokens > \
                   (session_state.tokens_processed - len(tokens_list)) // self.checkpoint_interval_tokens : # Check if interval boundary crossed
                    await self._perform_maintenance(session_id)
                    checkpoint = await self._create_checkpoint(session_state)
                    yield {'type': 'checkpoint', 'checkpoint_id': checkpoint['id'], 'session_stats': session_state.to_dict()}
        except Exception as e:
            self.logger.error(f"Error in stream processing for session {session_id}: {e}", exc_info=True)
            yield {'type': 'error', 'error': str(e), 'session_stats': session_state.to_dict()}
        finally:
            final_block = await self.adaptive_buffer._create_and_flush()
            if final_block and final_block.tokens:
                processed_block = await self._process_token_block(final_block, session_id)
                storage_decision = await self._determine_storage_strategy(processed_block, session_state)
                await self._execute_storage_strategy(processed_block, storage_decision)
            summary = await self._generate_session_summary(session_id)
            yield {'type': 'stream_complete', 'session_stats': session_state.to_dict(), 'total_stats': self.get_statistics(), 'summary': summary}
    
    async def generate_unlimited_output(self, prompt: str, session_id: str, max_tokens: Optional[int] = None, stream: bool = True, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        self.logger.info(f"Starting unlimited output generation for session {session_id}. Stream: {stream}")
        if stream:
            return self.output_generator.generate_unlimited_stream(prompt, session_id, max_tokens=max_tokens, **kwargs)
        else:
            full_output_chunks = []
            # Create a list from the async generator to await all chunks
            async for chunk in self.output_generator.generate_unlimited_stream(prompt, session_id, max_tokens=max_tokens, **kwargs):
                full_output_chunks.append(chunk)
            return "".join(full_output_chunks)
    
    async def process_generated_output(self, output_chunk: str, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        if not self.nlp_processor or not hasattr(self.nlp_processor, 'nlp') or not hasattr(self.nlp_processor.nlp, 'tokenizer'):
            self.logger.error("NLP processor or tokenizer not available for process_generated_output.")
            tokens = output_chunk.split() 
        else:
            tokens = [t.text for t in self.nlp_processor.nlp.tokenizer(output_chunk)]

        await self.adaptive_buffer.add_tokens(tokens, metadata={'session_id': session_id, 'type': 'generated_output', 'generation_metadata': metadata or {}})
        self.stats['total_tokens_generated'] = float(self.stats['total_tokens_generated']) + len(tokens)
        session_state = self._get_or_create_session(session_id)
        session_state.tokens_generated += len(tokens)

    async def retrieve_context(self, query: str, session_id: str, max_tokens: int = 8192, time_range: Optional[Tuple[datetime, datetime]] = None, include_semantic_links: bool = True) -> Dict[str, Any]:
        start_time = datetime.utcnow()
        query_embedding_list = await self._generate_embedding(query)
        default_error_response = {
            'error': 'Context retrieval failed.', 'context_parts': [], 'total_tokens': 0, 
            'retrieval_time_seconds': (datetime.utcnow() - start_time).total_seconds(),
            'context_summary': 'Error retrieving context.', 'blocks_examined':0, 'blocks_retrieved':0,
            'compression_ratio':1.0, 'avg_attention_score':0.0
        }
        if not query_embedding_list:
            default_error_response['error'] = 'Failed to generate query embedding.'
            return default_error_response
        query_embedding = np.array(query_embedding_list, dtype=np.float32)
        
        session_state = self.active_sessions.get(session_id)
        session_context_for_attention = session_state.to_dict() if session_state else None
        
        relevant_blocks = await self._collect_relevant_blocks(session_id, time_range)
        self.logger.info(f"Found {len(relevant_blocks)} potentially relevant blocks for query in session {session_id}.")
        
        blocks_for_attention = [b for b in relevant_blocks if (b.embeddings is not None and b.embeddings.size > 0) or b.compression_level == CompressionLevel.NONE]

        attended_blocks_with_scores = await self.attention_mechanism.compute_attention(
            query_embedding, blocks_for_attention, max_tokens, session_context_for_attention
        )
        
        if include_semantic_links:
            attended_blocks_with_scores = await self._expand_through_semantic_links(
                attended_blocks_with_scores, query_embedding, max_tokens
            )
        
        context_parts: List[Dict[str, Any]] = []
        total_retrieved_tokens = 0
        
        decompression_tasks = []
        original_attended_blocks_info = [] 

        for block, attention_score in attended_blocks_with_scores:
            original_attended_blocks_info.append({'block': block, 'score': attention_score})
            if block.compression_level != CompressionLevel.NONE:
                decompression_tasks.append(self._decompress_block_async(block))
            else:
                # Wrap non-coroutine block in a coroutine to be awaitable by gather
                async def identity_coro(b): return b
                decompression_tasks.append(identity_coro(block))


        decompressed_block_objects = await asyncio.gather(*decompression_tasks, return_exceptions=True)

        for original_info, decomp_result in zip(original_attended_blocks_info, decompressed_block_objects):
            original_block = original_info['block']
            attention_score = original_info['score']

            if isinstance(decomp_result, Exception):
                self.logger.error(f"Failed to decompress block {original_block.block_id}: {decomp_result}", exc_info=decomp_result)
                continue
            if decomp_result is None: 
                self.logger.warning(f"Block {original_block.block_id} could not be retrieved/decompressed.")
                continue

            decompressed_block: TokenBlock = decomp_result 
            original_block.access_count += 1 

            tokens_to_consider = decompressed_block.tokens if decompressed_block.tokens else original_block.tokens
            importance_scores_to_use = decompressed_block.importance_scores if decompressed_block.importance_scores.size > 0 else original_block.importance_scores

            relevant_tokens = await self._extract_relevant_tokens(
                tokens_to_consider, importance_scores_to_use, query_embedding, max_tokens - total_retrieved_tokens
            )
            
            if relevant_tokens:
                context_parts.append({
                    'tokens': relevant_tokens, 'text': ' '.join(relevant_tokens),
                    'attention_score': attention_score, 'block_id': original_block.block_id,
                    'timestamp': original_block.timestamp.isoformat() if original_block.timestamp else None, 
                    'compression_level': original_block.compression_level.name, 
                    'importance': float(np.mean(original_block.importance_scores) if original_block.importance_scores.size > 0 else 0.0)
                })
                total_retrieved_tokens += len(relevant_tokens)
            if total_retrieved_tokens >= max_tokens: break
        
        context_summary = await self._generate_context_summary_from_parts(context_parts, query)
        retrieval_time = (datetime.utcnow() - start_time).total_seconds()
        
        avg_attention_val = 0.0
        if attended_blocks_with_scores:
            scores_list = [score for _, score in attended_blocks_with_scores if isinstance(score, (float, int))]
            if scores_list:
                avg_attention_val = np.mean(scores_list)
                self.stats['retrieval_accuracy'] = float(self.stats['retrieval_accuracy']) * 0.95 + avg_attention_val * 0.05
        
        return {
            'context_parts': context_parts, 'total_tokens': total_retrieved_tokens,
            'context_summary': context_summary, 'blocks_examined': len(relevant_blocks),
            'blocks_retrieved': len(context_parts), 'compression_ratio': float(self.stats['compression_ratio']),
            'retrieval_time_seconds': retrieval_time,
            'avg_attention_score': np.mean([p['attention_score'] for p in context_parts if 'attention_score' in p and isinstance(p['attention_score'], (float, int))]) if context_parts else 0.0
        }

    async def _process_token_block(self, block: TokenBlock, session_id: str) -> TokenBlock:
        block_text = ' '.join(block.tokens) if block.tokens else ""
        block_embedding_list = await self._generate_embedding(block_text) 
        
        if block_embedding_list:
            block.embeddings = np.array(block_embedding_list, dtype=np.float32).reshape(1, -1)
            
        if self.compression_model and block.embeddings is not None and block.embeddings.size > 0:
            with torch.no_grad():
                self.compression_model.eval() # Set to eval mode for inference
                emb_tensor = torch.from_numpy(block.embeddings.astype(np.float32))
                
                context_emb_list = await self._get_context_embedding(session_id)
                context_tensor = torch.tensor(context_emb_list, dtype=torch.float32).unsqueeze(0) if context_emb_list else None
                
                _, _, importance_scalar_tensor, compression_strategy_tensor = self.compression_model(
                    emb_tensor, context_tensor
                )
                
                block_level_importance = importance_scalar_tensor.item() if importance_scalar_tensor.numel() == 1 else float(np.mean(importance_scalar_tensor.cpu().numpy()))
                
                if block.importance_scores is not None and block.importance_scores.size > 0:
                    block.importance_scores = block.importance_scores * block_level_importance 
                    block.importance_scores = np.clip(block.importance_scores, 0.0, 1.0)
                else: 
                    block.importance_scores = np.full(len(block.tokens), block_level_importance) if block.tokens else np.array([])

                strategy_probs = compression_strategy_tensor.squeeze().cpu().numpy()
                if strategy_probs.size > 0 : 
                    block.compression_level = CompressionLevel(int(np.argmax(strategy_probs)))
                else: 
                    block.compression_level = CompressionLevel.LIGHT 
        else:
            if block.importance_scores is None or block.importance_scores.size == 0:
                 block.importance_scores = np.ones(len(block.tokens)) * 0.5 if block.tokens else np.array([])
            block.compression_level = CompressionLevel.LIGHT 

        if self.nlp_processor:
            entities = self.nlp_processor.extract_entities(block_text)
            key_phrases = self._extract_key_phrases(block_text)
            block.metadata.update({'entities': entities, 'key_phrases': key_phrases})

        block.metadata.update({
            'session_id': session_id, 'processed_at': datetime.utcnow().isoformat(),
            'token_density': len(block.tokens) / max(len(block_text), 1) if block_text else 0
        })
        
        self.stats['total_blocks_created'] = float(self.stats['total_blocks_created']) + 1
        if block.importance_scores.size > 0:
            avg_imp = np.mean(block.importance_scores)
            self.stats['avg_block_importance'] = float(self.stats['avg_block_importance']) * 0.99 + avg_imp * 0.01

        session_state = self._get_or_create_session(session_id)
        if block.embeddings is not None and block.embeddings.size > 0:
             session_state.recent_embeddings.append(block.embeddings.flatten())
        return block

    async def _determine_storage_strategy(self, block: TokenBlock, session_state: SessionState) -> Dict[str, Any]:
        avg_importance = np.mean(block.importance_scores) if block.importance_scores.size > 0 else 0.0
        # Use model's suggested compression_level if available and reasonable
        strategy_compression_level = block.compression_level
        
        # Override model's suggestion if importance is very low or very high
        if avg_importance < self.config.get('low_importance_threshold_for_agg_compress', 0.3):
            strategy_compression_level = CompressionLevel.AGGRESSIVE
        elif avg_importance > self.config.get('high_importance_threshold_for_light_compress', 0.8) and \
             strategy_compression_level.value > CompressionLevel.LIGHT.value:
            strategy_compression_level = CompressionLevel.LIGHT
        
        # Further adjustment based on session length
        if session_state.tokens_processed > self.config.get('large_session_tokens_threshold', 1_000_000) and \
           strategy_compression_level.value < CompressionLevel.MODERATE.value:
            strategy_compression_level = CompressionLevel.MODERATE
            
        storage_tier = 'compressed' if strategy_compression_level != CompressionLevel.NONE else 'active'
        create_links = avg_importance > self.config.get('min_importance_for_semantic_links', 0.4)

        return {'compression_level': strategy_compression_level, 'storage_tier': storage_tier, 
                'create_links': create_links, 'index_entities': True}
    
    async def _execute_storage_strategy(self, block: TokenBlock, strategy: Dict[str, Any]):
        target_compression_level = strategy['compression_level']
        original_tokens = list(block.tokens) 

        if target_compression_level != CompressionLevel.NONE:
            compressed_data_bytes = await self._compress_block_with_level(block, target_compression_level)
            if compressed_data_bytes:
                self.compressed_blocks[block.block_id] = compressed_data_bytes
                block.compression_level = target_compression_level 
                if target_compression_level.value >= CompressionLevel.MODERATE.value:
                    block.tokens = [] # Clear tokens, they are now in compressed_blocks
            else: 
                self.logger.warning(f"Compression failed for block {block.block_id}. Storing as uncompressed in active memory.")
                if block.block_id in self.compressed_blocks: del self.compressed_blocks[block.block_id] # Remove if failed to update
                self.active_blocks[block.block_id] = block 
                block.compression_level = CompressionLevel.NONE 
        else: # No compression, store in active
            if block.block_id in self.compressed_blocks: del self.compressed_blocks[block.block_id] # Ensure not in compressed
            self.active_blocks[block.block_id] = block
        
        self._create_block_index(block, strategy, original_tokens_if_cleared=original_tokens if not block.tokens else None)
        
        # Manage active memory size
        if len(self.active_blocks) > self.max_active_blocks:
            await self._compress_old_blocks()

    async def _compress_block_with_level(self, block: TokenBlock, level: CompressionLevel) -> Optional[bytes]:
        self.logger.debug(f"Compressing block {block.block_id} to level {level.name}")
        try:
            if level == CompressionLevel.NEURAL: return await self._neural_compress(block)
            elif level == CompressionLevel.EXTREME: return await self._extreme_compress(block)
            elif level == CompressionLevel.AGGRESSIVE: return await self._aggressive_compress(block)
            elif level == CompressionLevel.MODERATE: return await self._moderate_compress(block)
            elif level == CompressionLevel.LIGHT: return await self._light_compress(block)
            else: 
                self.logger.warning(f"Unknown or NONE compression level '{level}' for block {block.block_id}. No compression applied by this method.")
                return None # Explicitly return None for NONE or unknown.
        except Exception as e:
            self.logger.error(f"Error compressing block {block.block_id} to level {level.name}: {e}", exc_info=True)
            return None

    async def _neural_compress(self, block: TokenBlock) -> Optional[bytes]:
        if block.embeddings is None or block.embeddings.size == 0:
             block_text_for_emb = ' '.join(block.tokens) if block.tokens else ""
             if not block_text_for_emb: self.logger.warning(f"Block {block.block_id} has no tokens for neural compression."); return None
             emb_list = await self._generate_embedding(block_text_for_emb)
             if not emb_list: self.logger.warning(f"Failed to generate embedding for block {block.block_id} for neural compression."); return None
             block.embeddings = np.array(emb_list, dtype=np.float32).reshape(1, -1)

        with torch.no_grad():
            self.compression_model.eval()
            emb_tensor = torch.from_numpy(block.embeddings.astype(np.float32))
            if emb_tensor.ndim == 1: emb_tensor = emb_tensor.unsqueeze(0)
            
            compressed_emb, _, _, _ = self.compression_model(emb_tensor) # Pass directly
            
        compressed_data = {
            'block_id': block.block_id, 'type': 'neural',
            'compressed_embedding': compressed_emb.cpu().numpy(),
            'importance_peaks': self._find_importance_peaks(block, num_peaks=self.config.get('neural_compression_peaks', 10)),
            'timestamp': block.timestamp.isoformat(), 
            'metadata_keys': list(block.metadata.keys()), 
            'original_length': len(block.tokens)
        }
        return zstd.compress(pickle.dumps(compressed_data), level=self.config.get('zstd_level_neural', 20))

    def _find_importance_peaks(self, block: TokenBlock, num_peaks: int = 10) -> List[Tuple[int, str, float]]:
        if not block.tokens or block.importance_scores is None or block.importance_scores.size == 0 or len(block.tokens) != len(block.importance_scores):
            return []
        
        actual_num_peaks = min(num_peaks, len(block.tokens))
        if actual_num_peaks <= 0: return []

        # Get indices of top importance scores, ensuring they are valid for block.tokens
        valid_indices = np.arange(len(block.tokens))
        scores_for_partition = block.importance_scores[valid_indices]

        # Ensure actual_num_peaks is not greater than the length of scores_for_partition
        if actual_num_peaks > len(scores_for_partition):
            actual_num_peaks = len(scores_for_partition)
        if actual_num_peaks == 0: return []


        top_relative_indices = np.argpartition(scores_for_partition, -actual_num_peaks)[-actual_num_peaks:]
        top_original_indices = valid_indices[top_relative_indices]
        
        peaks = [(int(idx), block.tokens[idx], float(block.importance_scores[idx])) for idx in top_original_indices]
        return sorted(peaks, key=lambda x: x[2], reverse=True)

    async def _extreme_compress(self, block: TokenBlock) -> bytes:
        if not block.tokens: return b""
        critical_tokens_data = self._find_importance_peaks(block, num_peaks=self.config.get('extreme_compression_peaks', 5))
        summary = ""
        if self.nlp_processor and block.tokens:
            summary = self.nlp_processor.summarize(' '.join(block.tokens), ratio=0.05)
        
        compressed_data = {'type': 'extreme', 'critical_tokens_data': critical_tokens_data, 'summary': summary, 
                           'timestamp': block.timestamp.isoformat(),
                           'entities_sample': block.metadata.get('entities', [])[:5], # Store a sample
                           'key_phrases_sample': block.metadata.get('key_phrases', [])[:3]} # Store a sample
        return zstd.compress(pickle.dumps(compressed_data), level=self.config.get('zstd_level_extreme', 22))

    async def _aggressive_compress(self, block: TokenBlock) -> bytes:
        if not block.tokens: return b""
        num_keep = int(len(block.tokens) * 0.3)
        if num_keep == 0 and block.tokens: num_keep = 1 
        if num_keep == 0: return b""

        if block.importance_scores is None or block.importance_scores.size < num_keep :
             top_indices = np.arange(min(num_keep, len(block.tokens))) # Fallback if not enough scores
        else:
            top_indices = np.argpartition(block.importance_scores, -num_keep)[-num_keep:]
        
        selected_tokens_data = [(int(idx), block.tokens[idx]) for idx in sorted(top_indices) if idx < len(block.tokens)]
        compressed_data = {'type': 'aggressive', 'selected_tokens_data': selected_tokens_data, 
                           'embedding': block.embeddings, 'timestamp': block.timestamp.isoformat(), 
                           'original_length': len(block.tokens)}
        return zstd.compress(pickle.dumps(compressed_data), level=self.config.get('zstd_level_aggressive', 19))

    async def _moderate_compress(self, block: TokenBlock) -> bytes:
        if not block.tokens: return b""
        num_keep = int(len(block.tokens) * 0.5)
        if num_keep == 0 and block.tokens: num_keep = 1
        if num_keep == 0: return b""

        if block.importance_scores is None or block.importance_scores.size < num_keep:
            top_indices = np.arange(min(num_keep, len(block.tokens)))
            selected_importance = np.array([])
        else:
            top_indices = np.argpartition(block.importance_scores, -num_keep)[-num_keep:]
            selected_importance = block.importance_scores[sorted(top_indices)]
        
        selected_tokens_data = [(int(idx), block.tokens[idx]) for idx in sorted(top_indices) if idx < len(block.tokens)]
        compressed_data = {'type': 'moderate', 'selected_tokens_data': selected_tokens_data, 
                           'embedding': block.embeddings, 'selected_importance': selected_importance, 
                           'timestamp': block.timestamp.isoformat(), 'original_length': len(block.tokens)}
        return zstd.compress(pickle.dumps(compressed_data), level=self.config.get('zstd_level_moderate', 15))

    async def _light_compress(self, block: TokenBlock) -> bytes:
        data_to_compress = {
            'type': 'light', 'tokens': block.tokens, 'embedding': block.embeddings, 
            'importance_scores': block.importance_scores, 'timestamp': block.timestamp.isoformat(),
            'metadata': block.metadata 
        }
        return zstd.compress(pickle.dumps(data_to_compress), level=self.config.get('zstd_level_light', 9))

    async def _create_semantic_links(self, block: TokenBlock):
        if block.embeddings is None or block.embeddings.size == 0: return
        block_emb_flat = block.embeddings.flatten()
        similar_blocks_info = await self._find_similar_blocks(block_emb_flat, block.block_id, limit=self.config.get('semantic_link_limit', 5))
        for similar_id, similarity_score in similar_blocks_info:
            block.semantic_links[similar_id] = similarity_score
            active_neighbor = self.active_blocks.get(similar_id)
            if active_neighbor: active_neighbor.semantic_links[block.block_id] = similarity_score
        self.semantic_graph.add_node(block.block_id, block.embeddings)
        for linked_id, weight in block.semantic_links.items():
            self.semantic_graph.add_edge(block.block_id, linked_id, weight)

    async def _learn_from_block(self, block: TokenBlock):
        if not self.learning_enabled: return
        if block.embeddings is None or block.embeddings.size == 0 or block.importance_scores is None or block.importance_scores.size == 0: 
            self.logger.debug(f"Skipping learning for block {block.block_id} due to missing embeddings or importance scores.")
            return
        self.training_buffer.append({
            'embeddings': block.embeddings, 
            'importance_scores': block.importance_scores, 
            'metadata': block.metadata, 'compression_level': block.compression_level
        })
        if len(self.training_buffer) >= self.batch_size_learning: await self._train_compression_model()

    async def _train_compression_model(self):
        if not self.learning_enabled or len(self.training_buffer) < self.batch_size_learning: return
        
        batch_indices = np.random.choice(len(self.training_buffer), size=self.batch_size_learning, replace=False)
        batch = [self.training_buffer[i] for i in batch_indices]
        
        valid_batch_items = [item for item in batch if item.get('embeddings') is not None and item.get('embeddings').size > 0 and item.get('importance_scores') is not None and item.get('importance_scores').size > 0]
        if not valid_batch_items: self.logger.warning("Training batch for compression model has no valid items."); return

        embeddings_list_torch = []
        for item in valid_batch_items:
            emb_np = item['embeddings'].astype(np.float32)
            if emb_np.ndim == 1: emb_np = emb_np.reshape(1, -1) # (dim) -> (1, dim)
            embeddings_list_torch.append(torch.from_numpy(emb_np))
        
        if not embeddings_list_torch: self.logger.warning("No valid embeddings to stack for training."); return
        embeddings_tensor = torch.stack(embeddings_list_torch) # (batch_size, 1, dim)

        target_importance_list = []
        for item in valid_batch_items:
            scores = item['importance_scores']
            target_importance_list.append(torch.tensor([np.mean(scores) if scores.size > 0 else 0.5], dtype=torch.float32))
        target_importance_tensor = torch.stack(target_importance_list) 

        target_strategies_list = [item['compression_level'].value for item in valid_batch_items]
        target_strategies_tensor = torch.tensor(target_strategies_list, dtype=torch.long)

        self.compression_model.train()
        self.optimizer.zero_grad()
        
        compressed_emb, reconstructed_emb, predicted_importance, predicted_strategy = self.compression_model(embeddings_tensor)
        
        reconstruction_loss = F.mse_loss(reconstructed_emb, embeddings_tensor)
        # Ensure predicted_importance has same shape as target_importance_tensor for loss calculation
        # predicted_importance is (batch, seq_len=1), target_importance_tensor is (batch, 1)
        importance_loss = F.mse_loss(predicted_importance, target_importance_tensor) 
        strategy_loss = F.cross_entropy(predicted_strategy, target_strategies_tensor)
        sparsity_loss = torch.mean(torch.abs(compressed_emb)) * self.config.get('compression_model',{}).get('sparsity_lambda', 0.01)
        
        total_loss = reconstruction_loss + 0.5 * importance_loss + 0.3 * strategy_loss + sparsity_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.compression_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.learning_scheduler: self.learning_scheduler.step()
        
        self.stats['learning_iterations'] = float(self.stats['learning_iterations']) + 1
        self.logger.debug(f"Compression model trained. Iter: {self.stats['learning_iterations']}, Loss: {total_loss.item():.4f}")
        
        # Clear trained items from buffer
        new_buffer = deque(maxlen=self.training_buffer.maxlen)
        current_items = list(self.training_buffer)
        indices_to_keep = [i for i in range(len(current_items)) if i not in batch_indices]
        for i in indices_to_keep: new_buffer.append(current_items[i])
        self.training_buffer = new_buffer

    # Restore other methods from combined_code.txt, adapted for consistency
    async def _expand_through_semantic_links(self, initial_blocks_with_scores: List[Tuple[TokenBlock, float]], query_embedding: np.ndarray, max_tokens: int) -> List[Tuple[TokenBlock, float]]:
        expanded_blocks = list(initial_blocks_with_scores)
        visited_ids = {block.block_id for block, _ in initial_blocks_with_scores}
        current_tokens = sum(len(block.tokens) for block, _ in initial_blocks_with_scores if block.tokens)
        
        expansion_queue: List[Tuple[float, str]] = [] 
        for block, score in initial_blocks_with_scores:
            for linked_id, link_strength in block.semantic_links.items():
                if linked_id not in visited_ids:
                    priority = -1 * score * link_strength 
                    heapq.heappush(expansion_queue, (priority, linked_id))
        
        while expansion_queue and current_tokens < max_tokens:
            neg_priority, block_id_to_expand = heapq.heappop(expansion_queue)
            if block_id_to_expand in visited_ids: continue
            
            linked_block = await self._get_block_by_id(block_id_to_expand)
            if not linked_block or not linked_block.tokens: continue 
            
            relevance = 0.0
            if linked_block.embeddings is not None and linked_block.embeddings.size > 0:
                link_emb_flat = linked_block.embeddings.flatten()
                if query_embedding.shape == link_emb_flat.shape:
                    norm_q = np.linalg.norm(query_embedding)
                    norm_l = np.linalg.norm(link_emb_flat)
                    if norm_q > 0 and norm_l > 0:
                        relevance = np.dot(query_embedding, link_emb_flat) / (norm_q * norm_l)
            
            if relevance > self.config.get('link_expansion_relevance_threshold', 0.5): 
                block_token_count = len(linked_block.tokens)
                if current_tokens + block_token_count <= max_tokens:
                    expanded_blocks.append((linked_block, relevance))
                    visited_ids.add(block_id_to_expand)
                    current_tokens += block_token_count
                    for next_id, next_strength in linked_block.semantic_links.items():
                        if next_id not in visited_ids:
                            priority = -1 * relevance * next_strength
                            heapq.heappush(expansion_queue, (priority, next_id))
        return expanded_blocks

    async def _generate_context_summary_from_parts(self, context_parts: List[Dict[str, Any]], query: str) -> str:
        if not context_parts: return "No relevant context found for the query."
        sorted_parts = sorted(context_parts, key=lambda x: x.get('attention_score', 0.0), reverse=True)
        all_text_for_summary = " ".join([part['text'] for part in sorted_parts[:self.config.get('summary_max_parts', 5)] if 'text' in part])
        if not all_text_for_summary: return "Context retrieved but contains no text."
        if not self.nlp_processor: return "NLP Processor not available for summary."
        overall_summary = self.nlp_processor.summarize(all_text_for_summary, ratio=self.config.get('summary_ratio', 0.2))
        return overall_summary or "Could not generate a concise summary of the retrieved context."
    
    # ... (The rest of the methods like _get_or_create_session, _extract_key_phrases, etc.,
    # would be included here, following the same pattern of using self.config and robust error handling)

    def _get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = SessionState(session_id)
        return self.active_sessions[session_id]

    def _extract_key_phrases(self, text: str) -> List[str]:
        if not self.nlp_processor or not hasattr(self.nlp_processor, 'nlp'): return []
        try:
            doc = self.nlp_processor.nlp(text)
            return list(set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2))[:self.config.get('key_phrase_limit', 10)]
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}", exc_info=True)
            return []
    
    async def _get_context_embedding(self, session_id: str) -> Optional[List[float]]:
        session = self.active_sessions.get(session_id)
        if session and session.recent_embeddings:
            embeddings_list_np = [emb_item for emb_item in session.recent_embeddings if emb_item is not None and emb_item.size > 0]
            if not embeddings_list_np: return None
            try:
                # Ensure all embeddings are 1D before averaging
                processed_embeddings = []
                for emb_item in embeddings_list_np:
                    if emb_item.ndim > 1: processed_embeddings.append(emb_item.flatten())
                    else: processed_embeddings.append(emb_item)
                if not processed_embeddings : return None
                return np.mean(np.array(processed_embeddings, dtype=np.float32), axis=0).tolist()
            except Exception as e:
                self.logger.error(f"Error averaging recent embeddings for session {session_id}: {e}", exc_info=True)
                return None
        return None
    
    async def _perform_maintenance(self, session_id: Optional[str] = None):
        self.logger.info(f"Performing maintenance for session: {session_id if session_id else 'GLOBAL'}")
        await self._compress_old_blocks(session_id_filter=session_id)
        self.semantic_graph.prune_weak_edges(threshold=self.config.get('graph_prune_threshold', 0.3))
        self._clean_caches()
        await self._optimize_indices() 
        if self.learning_enabled and self.stats['learning_iterations'] > 0 and \
           int(self.stats['learning_iterations']) % self.config.get('model_save_freq_iterations', 100) == 0:
            await self._save_model_checkpoint()
    
    async def _create_checkpoint(self, session_state: SessionState) -> Dict[str, Any]:
        checkpoint_id = f"CKPT_{session_state.session_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        active_blocks_for_session = len([b for b in self.active_blocks.values() if isinstance(b.metadata, dict) and b.metadata.get('session_id') == session_state.session_id])
        checkpoint_data = {
            'id': checkpoint_id, 'session_id': session_state.session_id,
            'timestamp': datetime.utcnow().isoformat(), 'tokens_processed': session_state.tokens_processed,
            'active_blocks_count_session': active_blocks_for_session,
            'model_state_summary': self._get_model_state_summary()
        }
        self.logger.info(f"Created checkpoint {checkpoint_id}")
        self.block_index[checkpoint_id] = checkpoint_data # Store checkpoint info in index for traceability
        return checkpoint_data
    
    async def _generate_session_summary(self, session_id: str) -> str:
        session_state = self.active_sessions.get(session_id)
        if not session_state: return "Session not found or no activity."
        duration_seconds = (datetime.utcnow() - session_state.start_time).total_seconds()
        return f"Session '{session_id}' summary: Processed {session_state.tokens_processed} tokens, created {session_state.blocks_created} blocks. Duration: {duration_seconds:.2f}s."

    async def _get_block_by_id(self, block_id: str) -> Optional[TokenBlock]:
        if block_id in self.active_blocks: return self.active_blocks[block_id]
        if block_id in self.compressed_blocks: return await self._decompress_block_by_id(block_id)
        self.logger.debug(f"Block {block_id} not found in active or known compressed stores. Hierarchical memory lookup not yet implemented here.")
        # TODO: Query hierarchical_memory (DB) if not in memory
        return None
    
    async def _decompress_block_by_id(self, block_id: str) -> Optional[TokenBlock]:
        # This is a simplified version of what was in the combined_code
        # The _decompress_block_async and _decompress_block_sync are more complete
        return await self._decompress_block_async(TokenBlock(block_id=block_id, tokens=[], embeddings=None, timestamp=datetime.utcnow(), importance_scores=np.array([]), compression_level=CompressionLevel.MODERATE)) # Pass a stub

    async def _find_similar_blocks(self, query_embedding: np.ndarray, exclude_id: Optional[str] = None, limit: int = 5) -> List[Tuple[str, float]]:
        if query_embedding is None or query_embedding.size == 0: return []
        similarities: List[Tuple[str, float]] = []
        for block_id, block in self.active_blocks.items():
            if block_id == exclude_id: continue
            if block.embeddings is not None and block.embeddings.size > 0:
                block_emb_flat = block.embeddings.flatten()
                if block_emb_flat.shape == query_embedding.shape:
                    norm_q = np.linalg.norm(query_embedding)
                    norm_b = np.linalg.norm(block_emb_flat)
                    if norm_q > 0 and norm_b > 0:
                        sim = np.dot(query_embedding, block_emb_flat) / (norm_q * norm_b)
                        similarities.append((block_id, float(sim)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def _create_block_index(self, block: TokenBlock, strategy: Dict[str, Any], original_tokens_if_cleared: Optional[List[str]] = None):
        tokens_for_count = original_tokens_if_cleared if original_tokens_if_cleared is not None else block.tokens
        self.block_index[block.block_id] = {
            'timestamp': block.timestamp, 'token_count': len(tokens_for_count or []),
            'importance': float(np.mean(block.importance_scores) if block.importance_scores.size > 0 else 0.0),
            'compression_level': block.compression_level.value, 'storage_tier': strategy['storage_tier'],
            'metadata_sample': {k: str(v)[:50] for k,v in block.metadata.items() if isinstance(v, (str,int,float,bool))}, 
            'semantic_links': list(block.semantic_links.keys()), 'access_count': block.access_count
        }

    def _compute_token_importance(self, tokens: List[str], base_block_importance: float) -> np.ndarray:
        if not tokens: return np.array([])
        scores = np.full(len(tokens), base_block_importance)
        for i, token_text in enumerate(tokens):
            if token_text in ['?', 'what', 'why']: scores[i] = min(1.0, scores[i] * 1.2)
            if token_text.lower() in ['important', 'key']: scores[i] = min(1.0, scores[i] * 1.5)
        return np.clip(scores, 0.0, 1.0)

    async def _extract_relevant_tokens(self, tokens_list: List[str], importance_scores: np.ndarray, query_embedding: np.ndarray, max_tokens: int) -> List[str]:
        if not tokens_list: return []
        num_to_select = min(max_tokens, len(tokens_list))
        if num_to_select <= 0: return []
        
        current_importance_scores = importance_scores
        if current_importance_scores is None or current_importance_scores.size != len(tokens_list):
            self.logger.warning("Importance scores missing/mismatched in _extract_relevant_tokens. Using default.")
            current_importance_scores = np.ones(len(tokens_list)) * 0.5
        
        # If query_embedding is available, could re-rank based on token/sentence similarity to query
        # For now, just use importance scores
        sorted_indices_by_importance = np.argsort(current_importance_scores)[::-1]
        selected_original_indices = sorted(sorted_indices_by_importance[:num_to_select])
        return [tokens_list[i] for i in selected_original_indices if i < len(tokens_list)] # Bounds check

    async def _collect_relevant_blocks(self, session_id: str, time_range: Optional[Tuple[datetime, datetime]]) -> List[TokenBlock]:
        relevant_blocks: List[TokenBlock] = []
        for block in self.active_blocks.values():
            if self._is_block_relevant(block, session_id, time_range): relevant_blocks.append(block)
        
        for block_id, index_entry in self.block_index.items():
            if block_id not in self.active_blocks: 
                idx_ts_val = index_entry.get('timestamp', datetime.min) # Default to avoid error
                idx_timestamp = datetime.fromisoformat(idx_ts_val) if isinstance(idx_ts_val, str) else idx_ts_val
                meta_sample = index_entry.get('metadata_sample', {})
                # Ensure meta_sample is dict for .get()
                meta_sample_dict = meta_sample if isinstance(meta_sample, dict) else {}

                if self._is_index_relevant({'timestamp': idx_timestamp, 'metadata': meta_sample_dict}, session_id, time_range):
                    stub_block = TokenBlock(
                        block_id=block_id, tokens=[], embeddings=None, 
                        timestamp=idx_timestamp, 
                        importance_scores=np.array([index_entry.get('importance', 0.5)]),
                        compression_level=CompressionLevel(index_entry.get('compression_level', 0)),
                        metadata=meta_sample_dict,
                        access_count=index_entry.get('access_count',0)
                    )
                    relevant_blocks.append(stub_block)
        return relevant_blocks

    def _is_block_relevant(self, block: TokenBlock, session_id: str, time_range: Optional[Tuple[datetime, datetime]]) -> bool:
        meta = block.metadata if isinstance(block.metadata, dict) else {}
        if meta.get('session_id') != session_id: return False
        if time_range and block.timestamp:
            if not (time_range[0] <= block.timestamp <= time_range[1]): return False
        return True

    def _is_index_relevant(self, index_entry: Dict[str, Any], session_id: str, time_range: Optional[Tuple[datetime, datetime]]) -> bool:
        meta_session_id = None
        if 'metadata' in index_entry and isinstance(index_entry['metadata'], dict):
            meta_session_id = index_entry['metadata'].get('session_id')
        if meta_session_id != session_id: return False
        
        entry_ts = index_entry.get('timestamp')
        if time_range and entry_ts:
            if isinstance(entry_ts, str): entry_ts = datetime.fromisoformat(entry_ts) 
            if not isinstance(entry_ts, datetime) or not (time_range[0] <= entry_ts <= time_range[1]): return False
        return True

    def _get_model_state_summary(self) -> Dict[str, Any]:
        return {'compression_model_params': sum(p.numel() for p in self.compression_model.parameters()),
                'learning_iterations': self.stats['learning_iterations'],
                'current_lr': self.optimizer.param_groups[0]['lr'] if self.optimizer and self.optimizer.param_groups else 'N/A'}

    async def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        self.logger.info(f"Checkpoint data generated: {checkpoint_data.get('id')}")
        # In a real system, save to DB or persistent file storage
        # For example, using aiofiles:
        # checkpoint_path = Path(self.config.get("checkpoint_dir", "checkpoints")) / f"{checkpoint_data['id']}.json"
        # checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # async with aiofiles.open(checkpoint_path, 'w') as f:
        #     await f.write(json.dumps(checkpoint_data, indent=2, default=str))


    async def _save_model_checkpoint(self):
        if not self.learning_enabled: return
        self.logger.info(f"Saving compression model checkpoint at iteration {self.stats['learning_iterations']}...")
        model_save_dir_str = self.config.get("model_save_path", "models/icm_checkpoints")
        model_save_dir = Path(model_save_dir_str)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"icm_compression_model_iter_{int(self.stats['learning_iterations'])}.pt"
        full_path = model_save_dir / filename
        
        try:
            torch.save({
                'model_state_dict': self.compression_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.learning_scheduler.state_dict() if self.learning_scheduler else None,
                'stats_snapshot': dict(self.stats), 
                'timestamp': datetime.utcnow().isoformat()
            }, full_path)
            self.logger.info(f"Compression model checkpoint saved to {full_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint to {full_path}: {e}", exc_info=True)
    
    def _clean_caches(self):
        current_time = datetime.utcnow()
        cache_ttl_attn = self.config.get('attention_mechanism',{}).get('cache_ttl_seconds', 300)
        keys_to_delete_attn = [k for k,v in self.attention_mechanism.attention_cache.items() if (current_time - v.get('timestamp', datetime.min)).total_seconds() > cache_ttl_attn]
        for k in keys_to_delete_attn: del self.attention_mechanism.attention_cache[k]
        
        cache_max_comp = self.config.get('compression_cache_max_size', 1000) # From main icm_config
        if len(self.compression_cache) > cache_max_comp:
            num_to_remove = len(self.compression_cache) - cache_max_comp + (cache_max_comp // 10) # Remove some extra
            # Evict based on pseudo-LRU (oldest timestamp if available, or just pop)
            # This requires items in compression_cache to have timestamps, which they might not.
            # For simplicity, just clear a portion if too large.
            keys_to_evict = list(self.compression_cache.keys())[:num_to_remove]
            for k_evict in keys_to_evict: del self.compression_cache[k_evict]
            
        self.logger.debug(f"Caches cleaned. Attention cache: {len(self.attention_mechanism.attention_cache)}, Compression cache: {len(self.compression_cache)}")

    async def _optimize_indices(self): 
        self.logger.debug("Index optimization routine executed (placeholder).")
        # In a real system, this might involve:
        # - Rebuilding vector indexes in PostgreSQL for memory_chunks or other tables.
        # - Optimizing internal data structures like self.block_index or self.semantic_graph.

    def _calculate_compression_score(self, block: TokenBlock) -> float:
        now = datetime.utcnow()
        age_hours = (now - block.timestamp).total_seconds() / 3600.0
        age_factor_config = self.config.get('compression_score_weights', {}).get('age_factor_hours', 24.0)
        age_factor = min(age_hours / age_factor_config, 1.0)
        
        importance_val = np.mean(block.importance_scores) if block.importance_scores is not None and block.importance_scores.size > 0 else 0.5
        importance_factor = 1.0 - importance_val
        
        access_factor = 1.0 / (math.log10(block.access_count + 1) + 1) 
        
        weights = self.config.get('compression_score_weights', {'age': 0.4, 'importance': 0.4, 'access': 0.2})
        score = (age_factor * weights.get('age',0.4) + 
                 importance_factor * weights.get('importance',0.4) + 
                 access_factor * weights.get('access',0.2))
        return float(np.clip(score, 0.0, 1.0))