# backend/core/consciousness/infinite_context_integration.py

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List

from backend.core.consciousness.unified_consciousness import UnifiedConsciousness, ConsciousnessState
from backend.core.consciousness.enhanced_infinite_context import EnhancedInfiniteContextManager
from backend.core.consciousness.advanced_self_improving_engine import AdvancedSelfImprovingEngine
from backend.utils.logger import Logger
# For proxied method type hints
from backend.io_systems.stream_manager import StreamManager


class InfiniteConsciousness:
    """
    Integrates the Enhanced Infinite Context Manager with UnifiedConsciousness
    to create a truly unlimited AI system that can process and generate
    billions of tokens while maintaining full awareness and learning.

    This class now proxies several key attributes and methods from the underlying
    UnifiedConsciousness instance for easier access and better integration.
    """
    
    def __init__(self, unified_consciousness_instance: UnifiedConsciousness): # Renamed for clarity
        self.logger = Logger(__name__)
        
        if not isinstance(unified_consciousness_instance, UnifiedConsciousness):
            msg = f"InfiniteConsciousness must be initialized with a UnifiedConsciousness instance, got {type(unified_consciousness_instance)}"
            self.logger.critical(msg)
            raise TypeError(msg)
            
        self.consciousness: UnifiedConsciousness = unified_consciousness_instance
        
        # Initialize the enhanced context manager
        # Ensure all required components on unified_consciousness_instance are available
        if not all([
            self.consciousness.hierarchical_memory,
            self.consciousness.nlp_processor,
            self.consciousness.logical_mind, # Used as embedding_model by EnhancedInfiniteContextManager
        ]):
            missing_comps = [
                name for name, comp in [
                    ("Hierarchical Memory", self.consciousness.hierarchical_memory),
                    ("NLP Processor", self.consciousness.nlp_processor),
                    ("Logical Mind (for embedding)", self.consciousness.logical_mind),
                ] if comp is None
            ]
            comp_error_msg = f"Cannot initialize EnhancedInfiniteContextManager: Underlying UnifiedConsciousness is missing required components: {', '.join(missing_comps)}"
            self.logger.critical(comp_error_msg)
            raise RuntimeError(comp_error_msg)

        self.infinite_context: EnhancedInfiniteContextManager = EnhancedInfiniteContextManager(
            hierarchical_memory=self.consciousness.hierarchical_memory,
            nlp_processor=self.consciousness.nlp_processor,
            embedding_model=self.consciousness.logical_mind, # logical_mind is used as the embedder here
            config=self.consciousness.config.get('infinite_context', { # Get IC config from main config
                'max_active_blocks': 2000, # Default if not in main config
                'compression_ratio': 0.1,
                'attention_scales': 6,
                'learning_rate': 0.001
            })
        )
        
        # Enhanced self-improving engine with infinite context awareness
        # Ensure hierarchical_memory exists on the base consciousness instance
        if not self.consciousness.hierarchical_memory:
             le_error_msg = "Cannot initialize AdvancedSelfImprovingEngine: Underlying UnifiedConsciousness is missing HierarchicalMemory."
             self.logger.critical(le_error_msg)
             raise RuntimeError(le_error_msg)

        self.enhanced_learning: AdvancedSelfImprovingEngine = AdvancedSelfImprovingEngine(
            consciousness=self.consciousness, # Pass the base UC
            memory=self.consciousness.hierarchical_memory,
            config=self.consciousness.config.get('self_improving_engine', { # Get SIE config from main config
                'infinite_context': True # Example specific config for SIE under IC
            })
        )
        
        self.active_streams: Dict[str, Any] = {} # Changed type hint to Any for StreamState placeholder
        
        self.logger.info("InfiniteConsciousness initialized, wrapping UnifiedConsciousness.")

    # --- Methods specific to InfiniteConsciousness ---

    async def process_unlimited_input(
        self,
        input_stream: AsyncGenerator[str, None], # Expects str chunks, tokenizer converts to tokens
        session_id: str,
        process_incrementally: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        stream_state = StreamState(session_id) # Assuming StreamState is defined or imported
        self.active_streams[session_id] = stream_state
        
        try:
            async def token_generator_wrapper(): # Wrapper to ensure NLP processor is available
                if not self.consciousness.nlp_processor:
                    raise RuntimeError("NLP Processor not available in UnifiedConsciousness for tokenizing stream.")
                async for text_chunk in input_stream:
                    # Ensure nlp object itself is used if nlp_processor.nlp is the spacy model
                    if hasattr(self.consciousness.nlp_processor, 'nlp') and \
                       self.consciousness.nlp_processor.nlp is not None and \
                       hasattr(self.consciousness.nlp_processor.nlp, 'tokenizer'):
                        tokens = [token.text for token in self.consciousness.nlp_processor.nlp.tokenizer(text_chunk)]
                        yield tokens
                    else:
                        # Fallback or raise error if tokenizer not found
                        self.logger.warning("Tokenizer not found on nlp_processor.nlp. Using simple split for stream.")
                        yield text_chunk.split()


            async for update in self.infinite_context.process_unlimited_stream(
                token_generator_wrapper(),
                session_id,
                callback=lambda stats: self._update_stream_state(session_id, stats)
            ):
                if update['type'] == 'block_processed':
                    if process_incrementally:
                        response = await self._process_block_through_consciousness(update, session_id)
                        if response:
                            yield {
                                'type': 'incremental_response',
                                'response': response,
                                'block_id': update['block_id'],
                                'progress': stream_state.to_dict()
                            }
                    await self._learn_from_block(update, session_id)
                elif update['type'] == 'checkpoint':
                    await self._save_consciousness_checkpoint(session_id)
                    yield {
                        'type': 'checkpoint',
                        'checkpoint_id': update['checkpoint_id'],
                        'message': 'Consciousness state saved at checkpoint.'
                    }
                elif update['type'] == 'stream_complete':
                    if not process_incrementally:
                        final_response = await self._process_complete_input(session_id)
                        yield {
                            'type': 'final_response',
                            'response': final_response,
                            'summary': update.get('summary', ''),
                            'total_stats': update.get('total_stats', {})
                        }
                    else:
                        yield update # Pass through stream_complete if incremental
                        
        except Exception as e:
            self.logger.error(f"Error in process_unlimited_input for session '{session_id}': {e}", exc_info=True)
            yield {'type': 'error', 'error': str(e), 'session_id': session_id}
        finally:
            if session_id in self.active_streams:
                del self.active_streams[session_id]
    
    async def generate_unlimited_output(
        self,
        prompt: str,
        session_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs # To pass other generation params like stop_sequences
    ) -> AsyncGenerator[str, None]: # Return type is AsyncGenerator[str, None] if stream=True

        initial_response = await self.consciousness.process_input(prompt, session_id)
        
        generation_context = {
            'initial_states': initial_response.get('final_states', {}),
            'truth_evaluation': initial_response.get('truth_evaluation', {}),
            'ethical_considerations': await self._get_ethical_guidelines(prompt)
        }
        
        generation_count = 0
        async for chunk in self.infinite_context.generate_unlimited_output(
            prompt=prompt,
            session_id=session_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True, # generate_unlimited_output of EnhancedICM should handle its own streaming
            **kwargs    # Pass along other generation parameters
        ):
            filtered_chunk = await self._apply_consciousness_filters(chunk, session_id)
            if filtered_chunk:
                yield filtered_chunk
                generation_count += 1
                if generation_count % 100 == 0: # Example periodic update
                    await self._update_consciousness_state(session_id, filtered_chunk)
                if generation_count % 50 == 0:
                    await self._learn_from_generation(prompt, filtered_chunk, session_id)
    
    async def query_with_infinite_context(
        self,
        query: str,
        session_id: str,
        max_context_tokens: int = 16384
        # include_semantic_links is a parameter for infinite_context.retrieve_context
    ) -> Dict[str, Any]:
        
        context_data = await self.infinite_context.retrieve_context(
            query=query,
            session_id=session_id,
            max_tokens=max_context_tokens,
            include_semantic_links=True # Defaulting to True as per previous use
        )
        
        context_summary = context_data.get('context_summary', '')
        relevant_parts = context_data.get('context_parts', [])
        relevant_parts.sort(key=lambda x: x.get('attention_score', 0.0), reverse=True)
        
        context_sections = []
        if context_summary: context_sections.append(f"Context Summary:\n{context_summary}")
        if relevant_parts:
            key_facts = [part['text'][:200] for part in relevant_parts[:5] if part.get('attention_score', 0.0) > 0.8]
            if key_facts: context_sections.append(f"Key Relevant Information:\n" + '\n'.join(f"- {fact}" for fact in key_facts))
        
        enriched_prompt = '\n\n'.join(context_sections) + f"\n\nQuery: {query}"
        
        # Use the underlying UnifiedConsciousness for processing the enriched prompt
        response = await self.consciousness.process_input(enriched_prompt, session_id)
        
        response['context_metadata'] = {
            'blocks_examined': context_data.get('blocks_examined', 0),
            'blocks_retrieved': context_data.get('blocks_retrieved', 0),
            'avg_attention_score': context_data.get('avg_attention_score', 0),
            'retrieval_time': context_data.get('retrieval_time_seconds', 0)
        }
        return response

    # --- Proxied attributes from UnifiedConsciousness ---
    @property
    def config(self) -> Dict[str, Any]:
        return self.consciousness.config

    @property
    def state(self) -> ConsciousnessState:
        return self.consciousness.state
    
    @state.setter
    def state(self, value: ConsciousnessState):
        self.consciousness.state = value

    @property
    def thinking_task(self) -> Optional[asyncio.Task]:
        return self.consciousness.thinking_task

    @property
    def dreaming_task(self) -> Optional[asyncio.Task]:
        return self.consciousness.dreaming_task

    @property
    def hierarchical_memory(self): # Type hint would be from backend.memory.hierarchical_memory
        return self.consciousness.hierarchical_memory

    @property
    def truth_memory(self): # Type hint from backend.memory.truth_memory
        return self.consciousness.truth_memory
    
    @property
    def dream_memory(self): # Type hint from backend.memory.dream_memory
        return self.consciousness.dream_memory

    @property
    def working_memory(self): # Type hint from backend.memory.working_memory
        return self.consciousness.working_memory

    @property
    def nlp_processor(self): # Type hint from backend.io_systems.natural_language_processor
        return self.consciousness.nlp_processor

    @property
    def logical_mind(self): # Type hint from backend.core.minds.logical_mind
        return self.consciousness.logical_mind

    @property
    def creative_mind(self): # Type hint from backend.core.minds.creative_mind
        return self.consciousness.creative_mind

    @property
    def emotional_mind(self): # Type hint from backend.core.minds.emotional_mind
        return self.consciousness.emotional_mind
        
    @property
    def ethical_framework(self): # Type hint from backend.core.ethics.ethical_framework
        return self.consciousness.ethical_framework

    # --- Proxied methods from UnifiedConsciousness ---
    async def process_input(self, input_text: str, session_id: str) -> Dict[str, Any]:
        """Proxies to UnifiedConsciousness.process_input."""
        # This method is crucial. If IC is to be the main interface, it should decide
        # whether to use its unlimited processing or the standard UC processing.
        # For now, let's assume standard UC processing if this specific method is called.
        # If truly unlimited processing is desired for this input,
        # the caller should use `process_unlimited_input` or `query_with_infinite_context`.
        self.logger.debug(f"InfiniteConsciousness.process_input called, proxying to UnifiedConsciousness for session '{session_id}'.")
        return await self.consciousness.process_input(input_text, session_id)

    def get_stream_manager(self) -> StreamManager:
        """Proxies to UnifiedConsciousness.get_stream_manager."""
        if not self.consciousness.stream_manager:
            self.logger.error("StreamManager not available on underlying UnifiedConsciousness.")
            # Or raise an error, or return a dummy manager.
            # For now, let's assume it's initialized.
        return self.consciousness.get_stream_manager()

    async def start_thinking(self):
        await self.consciousness.start_thinking()

    async def start_dreaming(self):
        await self.consciousness.start_dreaming()

    async def stop_autonomous_processing(self):
        await self.consciousness.stop_autonomous_processing()

    async def shutdown(self):
        self.logger.info("Shutting down InfiniteConsciousness and its components.")
        # Potentially shutdown self.infinite_context or self.enhanced_learning if they have specific cleanup.
        # For now, they don't have explicit shutdown methods.
        await self.consciousness.shutdown() # Shutdown the wrapped UnifiedConsciousness

    # --- Internal helper methods for InfiniteConsciousness ---
    # (These methods are copied from the original InfiniteConsciousness but may need adjustment
    # to use self.consciousness for base UC operations and self.infinite_context for ICM operations)

    async def _process_block_through_consciousness(self, block_update: Dict[str, Any], session_id: str) -> Optional[str]:
        # This method should use self.infinite_context.retrieve_context
        # and self.consciousness.process_input (the base UC's method)
        block_context = await self.infinite_context.retrieve_context(
            query="", session_id=session_id, max_tokens=4096
        )
        if not block_context.get('context_parts'): return None
        recent_text = ' '.join([part['text'] for part in block_context['context_parts'][-3:]])
        if '?' in recent_text or any(kw in recent_text.lower() for kw in ['explain', 'describe', 'what', 'why', 'how']):
            response = await self.consciousness.process_input(recent_text, session_id) # Base UC processing
            return response.get('output', '')
        return None

    async def _process_complete_input(self, session_id: str) -> str:
        full_context = await self.infinite_context.retrieve_context(
            query="comprehensive summary", session_id=session_id, max_tokens=8192
        )
        context_summary = full_context.get('context_summary', '')
        response = await self.consciousness.process_input(
            f"Based on the following context, provide a comprehensive response:\n\n{context_summary}",
            session_id
        )
        return response.get('output', 'Processing complete.')

    async def _apply_consciousness_filters(self, chunk: str, session_id: str) -> Optional[str]:
        if not self.consciousness.ethical_framework or not self.consciousness.truth_memory or not self.consciousness.nlp_processor:
            self.logger.warning("Skipping consciousness filters: one or more required components missing on base UC.")
            return chunk
        ethical_eval = await self.consciousness.ethical_framework.evaluate(chunk)
        if ethical_eval['decision'].value == 'BLOCK': # Assuming EthicalDecision is an Enum with .value
            self.logger.warning(f"Chunk blocked by ethical framework: {ethical_eval['reason']}")
            return None
        # Simplified truth consistency check
        # main_claim = self.consciousness.nlp_processor.summarize(chunk, ratio=0.3) or chunk # Get a key claim
        # if main_claim:
        #     eval_result, _ = await self.consciousness.truth_memory.evaluate_claim(main_claim)
        #     if eval_result.value == 'CONTRADICTORY': # Assuming TruthEvaluation is an Enum
        #         chunk = f"[Note: Adjusted for consistency] {chunk}"
        return chunk

    async def _learn_from_block(self, block_update: Dict[str, Any], session_id: str):
        learning_data = {
            'block_id': block_update['block_id'], 
            'importance': block_update.get('importance', 0.5), # Use .get for safety
            'compression_level': block_update.get('compression_level', 'NONE'),
            'semantic_links': block_update.get('semantic_links', 0),
            'session_id': session_id, 'timestamp': datetime.utcnow()
        }
        await self.enhanced_learning.learn_from_interaction(
            interaction_data=learning_data, response_data={'type': 'block_processing'}, feedback=None
        )

    async def _learn_from_generation(self, prompt: str, generated_chunk: str, session_id: str):
        quality_metrics = await self._analyze_generation_quality(prompt, generated_chunk)
        await self.enhanced_learning.learn_from_interaction(
            interaction_data={'prompt': prompt, 'generated': generated_chunk, 'session_id': session_id},
            response_data={'quality_metrics': quality_metrics}, feedback=None
        )

    async def _analyze_generation_quality(self, prompt: str, generated: str) -> Dict[str, float]:
        # Placeholder - actual analysis would involve NLP and potentially model-based evaluation
        return {'coherence': 0.8, 'relevance': 0.85, 'creativity': 0.7, 'overall': 0.78}

    async def _get_ethical_guidelines(self, prompt: str) -> Dict[str, Any]:
        if not self.consciousness.nlp_processor: return {} # Guard
        entities = self.consciousness.nlp_processor.extract_entities(prompt)
        guidelines = {'avoid_topics': [], 'emphasis': [], 'tone': 'neutral'}
        for entity, entity_type in entities:
            if entity_type == 'PERSON': guidelines['avoid_topics'].append('personal_information')
            elif entity_type in ['ORG', 'GPE']: guidelines['emphasis'].append('factual_accuracy')
        return guidelines

    async def _update_consciousness_state(self, session_id: str, recent_content: str):
        if self.consciousness.working_memory:
            await self.consciousness.working_memory.add(
                session_id=session_id, key='recent_generation_ic', value=recent_content, ttl_seconds=300
            )
        if self.consciousness.hierarchical_memory:
            await self.consciousness.hierarchical_memory.add(
                content=f"IC Generated: {recent_content[:100]}...", importance_score=0.6,
                metadata={'type': 'ic_generation', 'session_id': session_id, 'timestamp': datetime.utcnow()}
            )

    async def _save_consciousness_checkpoint(self, session_id: str):
        # This method should primarily interact with self.infinite_context for its checkpointing
        # and potentially log some high-level UC state.
        ic_checkpoint_data = await self.infinite_context._create_checkpoint( # Access internal method of ICM
             self.infinite_context._get_or_create_session(session_id) # Get ICM's session state
        )
        uc_state_name = self.consciousness.state.name if self.consciousness.state else "UNKNOWN"

        checkpoint_data = {
            'session_id': session_id, 'timestamp': datetime.utcnow(),
            'infinite_context_checkpoint_id': ic_checkpoint_data.get('id'),
            'unified_consciousness_state': uc_state_name,
            'active_streams_ic_wrapper': len(self.active_streams),
            'learning_iterations_wrapper': self.enhanced_learning.performance_tracker.get('interaction_count', 0)
        }
        if self.consciousness.hierarchical_memory:
            await self.consciousness.hierarchical_memory.add(
                content=f"Full System Checkpoint for session {session_id}", importance_score=0.95,
                metadata=checkpoint_data
            )
        self.logger.info(f"Full system checkpoint created for session {session_id}, including IC state.")

    def _update_stream_state(self, session_id: str, stats: Dict[str, Any]):
        if session_id in self.active_streams:
            self.active_streams[session_id].update(stats)

    async def get_session_summary(self, session_id: str, include_learning_insights: bool = True) -> Dict[str, Any]:
        context_stats = await self.infinite_context.export_session_data(session_id)
        consciousness_insights = {}
        if self.consciousness.hierarchical_memory:
            memories = await self.consciousness.hierarchical_memory.search(query="", session_id=session_id, limit=100)
            consciousness_insights = {
                'total_memories': len(memories),
                'key_topics': self._extract_key_topics_from_memories(memories),
                'emotional_journey': await self._analyze_emotional_journey_from_memories(memories)
            }
        learning_insights = {}
        if include_learning_insights:
            learning_insights = await self.enhanced_learning.predict_optimal_response({'session_id': session_id})
        
        return {
            'session_id': session_id, 'context_statistics': context_stats,
            'consciousness_insights': consciousness_insights, 'learning_insights': learning_insights,
            'capabilities': {
                'tokens_processed': context_stats.get('statistics', {}).get('total_tokens', 0),
                'compression_achieved': context_stats.get('statistics', {}).get('compression_ratio', 1.0),
                'can_continue': True
            }
        }

    def _extract_key_topics_from_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        if not self.consciousness.nlp_processor: return []
        topics = set()
        for memory in memories:
            metadata = memory.get('metadata', {})
            if isinstance(metadata, dict) and 'entities' in metadata and isinstance(metadata['entities'], list):
                for entity_tuple in metadata['entities'][:5]: # Process first 5 entities
                    if isinstance(entity_tuple, (list,tuple)) and len(entity_tuple) > 0 and isinstance(entity_tuple[0], str) :
                        topics.add(entity_tuple[0].lower())
        return list(topics)[:10]

    async def _analyze_emotional_journey_from_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Placeholder for more complex analysis
        return {'dominant_emotion': 'neutral', 'emotional_variance': 0.3, 'trend': 'stable'}


# Dummy StreamState class if not defined elsewhere or to avoid circular import from integration.py
# This should ideally be defined in a shared types or state management module.
class StreamState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tokens_processed = 0
        # Add other relevant attributes
    def update(self, stats: Dict[str, Any]):
        self.tokens_processed = stats.get('tokens_processed', self.tokens_processed)
    def to_dict(self) -> Dict[str, Any]:
        return {'session_id': self.session_id, 'tokens_processed': self.tokens_processed}