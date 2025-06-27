# backend/core/consciousness/unified_consciousness.py

import asyncio
import random
import math
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING, List

from backend.utils.logger import Logger
from backend.database.connection_manager import DatabaseManager

# --- Type Hinting for Circular Dependencies ---
if TYPE_CHECKING:
    from backend.core.minds.logical_mind import LogicalMind
    from backend.core.minds.creative_mind import CreativeMind
    from backend.core.minds.emotional_mind import EmotionalMind
    from backend.core.dialogue.internal_dialogue import InternalDialogue
    from backend.core.ethics.ethical_framework import EthicalFramework
    from backend.memory.working_memory import WorkingMemory
    from backend.memory.truth_memory import TruthMemory
    from backend.memory.dream_memory import DreamMemory
    from backend.memory.hierarchical_memory import HierarchicalMemory
    from backend.io_systems.natural_language_processor import NaturalLanguageProcessor
    from backend.io_systems.multimodal_input import MultimodalInput
    from backend.io_systems.output_generator import OutputGenerator
    from backend.io_systems.stream_manager import StreamManager
    from backend.core.consciousness.self_improving_engine import SelfImprovingEngine

# --- Runtime Imports ---
from backend.core.minds.logical_mind import LogicalMind
from backend.core.minds.creative_mind import CreativeMind
from backend.core.minds.emotional_mind import EmotionalMind
from backend.core.dialogue.internal_dialogue import InternalDialogue
from backend.core.ethics.ethical_framework import EthicalFramework, EthicalDecision
from backend.memory.working_memory import WorkingMemory
from backend.memory.truth_memory import TruthMemory, TruthEvaluation
from backend.memory.dream_memory import DreamMemory
from backend.memory.hierarchical_memory import HierarchicalMemory
from backend.io_systems.natural_language_processor import NaturalLanguageProcessor
from backend.io_systems.multimodal_input import MultimodalInput
from backend.io_systems.output_generator import OutputGenerator
from backend.io_systems.stream_manager import StreamManager
from backend.core.consciousness.self_improving_engine import SelfImprovingEngine


class ConsciousnessState(Enum):
    OFFLINE = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    CONVERSING = auto()
    THINKING = auto()
    DREAMING = auto()
    MAINTAINING = auto()
    SHUTTING_DOWN = auto()


class UnifiedConsciousness:
    """
    The central orchestrator of the Prometheus system.
    """

    def __init__(self, config: Dict[str, Any], thought_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.config = config
        self.logger = Logger(__name__)
        self.state = ConsciousnessState.OFFLINE
        self.thought_callback = thought_callback
        self.thinking_task: Optional[asyncio.Task] = None
        self.dreaming_task: Optional[asyncio.Task] = None
        self.maintenance_task: Optional[asyncio.Task] = None
        
        # --- FIX: Store the main event loop this instance is created in ---
        self.main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Get the singleton instance. It's guaranteed to be initialized by the launcher.
        self.db_manager: DatabaseManager = DatabaseManager()

        self.working_memory: Optional[WorkingMemory] = None
        self.truth_memory: Optional[TruthMemory] = None
        self.dream_memory: Optional[DreamMemory] = None
        self.hierarchical_memory: Optional[HierarchicalMemory] = None
        self.logical_mind: Optional[LogicalMind] = None
        self.creative_mind: Optional[CreativeMind] = None
        self.emotional_mind: Optional[EmotionalMind] = None
        self.internal_dialogue: Optional[InternalDialogue] = None
        self.ethical_framework: Optional[EthicalFramework] = None
        self.nlp_processor: Optional[NaturalLanguageProcessor] = None
        self.multimodal_handler: Optional[MultimodalInput] = None
        self.output_generator: Optional[OutputGenerator] = None
        self.stream_manager: Optional["StreamManager"] = None
        self.self_improving_engine: Optional["SelfImprovingEngine"] = None

    @classmethod
    async def create(cls, config: Dict[str, Any], thought_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> "UnifiedConsciousness":
        instance = cls(config, thought_callback)
        instance.state = ConsciousnessState.INITIALIZING
        
        # --- FIX: Capture the event loop at creation time ---
        try:
            instance.main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            instance.logger.error("Could not get running event loop during UnifiedConsciousness creation. This may cause issues if called from a non-async context initially.")
            # Attempt to get or create a loop, though this might not be the desired one.
            instance.main_event_loop = asyncio.get_event_loop_policy().get_event_loop()


        instance.logger.info("--- Prometheus Consciousness Initializing ---")
        try:
            await instance._initialize_all_components()
            instance.state = ConsciousnessState.ACTIVE
            instance.logger.info("--- Prometheus Initialization Complete. State: ACTIVE ---")
            instance.start_maintenance()
        except Exception as e:
            instance.logger.critical("Fatal error during Prometheus initialization: %s", e, exc_info=True)
            instance.state = ConsciousnessState.OFFLINE
            raise
        return instance

    async def _initialize_all_components(self):
        """Orchestrates the ordered initialization of all system components."""
        self.logger.info("Initializing I/O Systems...")
        io_config = self.config.get('io_systems', {})
        self.nlp_processor = NaturalLanguageProcessor(io_config)
        self.multimodal_handler = MultimodalInput(io_config)
        self.output_generator = OutputGenerator(io_config, self.nlp_processor)
        
        self.logger.info("Initializing Core Minds...")
        neural_config = self.config.get('neural', {})
        minds_config = self.config.get('minds', {})
        self.logical_mind = LogicalMind(minds_config.get('logical', {}), neural_config)
        self.creative_mind = CreativeMind(minds_config.get('creative', {}), neural_config)
        self.emotional_mind = EmotionalMind(minds_config.get('emotional', {}), neural_config)
        await asyncio.gather(self.logical_mind.initialize(), self.creative_mind.initialize(), self.emotional_mind.initialize())
        
        self.logger.info("Initializing Memory Systems...")
        memory_config = self.config.get('memory', {})
        self.working_memory = await WorkingMemory.create(memory_config)
        self.truth_memory = await TruthMemory.create(memory_config, self.logical_mind)
        self.dream_memory = await DreamMemory.create(memory_config, self.logical_mind) # Uses logical_mind as an embedder
        # HierarchicalMemory needs an embedder and NLP
        self.hierarchical_memory = await HierarchicalMemory.create(memory_config, self.logical_mind, self.nlp_processor)
        
        self.logger.info("Initializing Self-Improving Engine...")
        self.self_improving_engine = SelfImprovingEngine(consciousness=self, memory=self.hierarchical_memory)
        self.logger.info("Self-Improving Engine initialized.")

        self.logger.info("Initializing Dialogue and Ethics Frameworks...")
        minds = {'logical': self.logical_mind, 'creative': self.creative_mind, 'emotional': self.emotional_mind}
        self.internal_dialogue = InternalDialogue(self.config.get('dialogue', {}), minds)
        self.ethical_framework = EthicalFramework(self.config.get('ethics', {}))
        
        self.logger.info("Initializing Stream Manager...")
        self.stream_manager = StreamManager(self, self.nlp_processor, self.multimodal_handler, self.output_generator)

    async def _process_input_core(self, input_text: str, session_id: str) -> Dict[str, Any]:
        """The core cognitive loop, now including self-correction and a learning step. This method assumes it's running on the correct event loop for DB access."""
        start_time = asyncio.get_event_loop().time()
        original_state = self.state
        self.state = ConsciousnessState.CONVERSING
        self.logger.info(f"Cognitive loop started for session '{session_id}'. Input: '{input_text[:100]}...'")
        
        try:
            # Step 1: Contextualization
            if not self.hierarchical_memory:
                raise RuntimeError("HierarchicalMemory not initialized.")
            chat_history = await self.hierarchical_memory.get_context_window(session_id, input_text)
            enriched_prompt = self._format_chat_history_for_prompt(chat_history, input_text)
            
            # Step 2: Multi-Mind Processing & Internal Dialogue
            if not self.logical_mind or not self.creative_mind or not self.emotional_mind or not self.internal_dialogue:
                 raise RuntimeError("One or more core minds or internal dialogue not initialized.")
            tasks = {
                'logical': self.logical_mind.process({'text': enriched_prompt}),
                'creative': self.creative_mind.process({'text': enriched_prompt, 'logical_mind_instance': self.logical_mind}),
                'emotional': self.emotional_mind.process({'text': enriched_prompt, 'logical_mind_instance': self.logical_mind})
            }
            initial_states = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values(), return_exceptions=True)))
            for name, result in initial_states.items():
                if isinstance(result, Exception): raise RuntimeError(f"Mind '{name}' failed processing initial input.") from result

            dialogue_result = await self.internal_dialogue.run(enriched_prompt, initial_states)
            final_states = dialogue_result["final_states"]
            
            # Step 3: Extract Proposed Response
            proposed_output = final_states.get('creative', {}).get('payload', {}).get('generated_text', "I am not sure how to respond to that.")
            truth_evaluation_result = {}

            # Step 4: Self-Correction via Truth Memory
            if not self.truth_memory or not self.nlp_processor:
                raise RuntimeError("TruthMemory or NLPProcessor not initialized.")
            main_claim = self.nlp_processor.summarize(proposed_output, ratio=0.5) or proposed_output
            if main_claim:
                eval_status, related_truth = await self.truth_memory.evaluate_claim(main_claim)
                truth_evaluation_result = {'status': eval_status.value, 'evaluated_claim': main_claim, 'related_truth': related_truth}
                
                if eval_status == TruthEvaluation.CONTRADICTORY and related_truth:
                    self.logger.warning(f"Self-correction: Proposed output contradicts a stored truth. Original: '{proposed_output}'")
                    proposed_output = f"I have a conflicting belief about that. My records indicate '{related_truth['claim']}' is {related_truth['value']}. However, regarding your query: {proposed_output}"
                
                elif eval_status == TruthEvaluation.NOVEL:
                    logical_confidence = final_states.get('logical', {}).get('confidence', 0.0)
                    if logical_confidence > 0.85:
                        self.logger.info(f"Learning: Adding novel, high-confidence claim to Truth Memory: '{main_claim}'")
                        await self.truth_memory.add(claim=main_claim, value='TRUE', confidence=logical_confidence, evidence=[{'source': f'inference_session_{session_id}'}], source='self_inference')

            # Step 5: Ethical Framework Check
            if not self.ethical_framework:
                raise RuntimeError("EthicalFramework not initialized.")
            ethical_evaluation = await self.ethical_framework.evaluate(proposed_output, input_text)
            final_output, status = (ethical_evaluation["reason"], "blocked") if ethical_evaluation["decision"] == EthicalDecision.BLOCK else (proposed_output, "success")

            # Step 6: Memory Consolidation
            importance_score = await self._calculate_importance_score(input_text, final_output, final_states)
            full_interaction_text = f"User: {input_text}\nAssistant: {final_output}"
            interaction_metadata = {'session_id': session_id, 'status': status, 'input_text': input_text, 'output_text': final_output, 'truth_evaluation': truth_evaluation_result, 'final_mind_states': dialogue_result}
            await self.hierarchical_memory.add(content=full_interaction_text, importance_score=importance_score, metadata=interaction_metadata)
            
            # Step 7: Self-Improvement & Learning
            if self.self_improving_engine:
                processing_time = asyncio.get_event_loop().time() - start_time
                input_tokens = len(self.nlp_processor.nlp.tokenizer(input_text))
                output_tokens = len(self.nlp_processor.nlp.tokenizer(final_output))
                
                interaction_data_for_learning = {
                    'session_id': session_id, 'input_text': input_text, 'output_text': final_output,
                    'total_input_tokens': input_tokens, 'total_output_tokens': output_tokens,
                    'context_used': chat_history, 'processing_time': processing_time,
                    'final_states': final_states, 'truth_evaluation': truth_evaluation_result
                }
                asyncio.create_task( # Fire-and-forget learning task
                    self.self_improving_engine.learn_from_interaction(
                        interaction_data=interaction_data_for_learning,
                        feedback=None # Feedback would come from another source
                    )
                )

            return {"status": status, "output": final_output, "final_states": final_states, "truth_evaluation": truth_evaluation_result}
        
        except Exception as e:
            self.logger.critical(f"Critical error in cognitive loop for session '{session_id}': {e}", exc_info=True)
            return {"status": "error", "output": f"An internal cognitive error occurred: {e}", "final_states": None}
        finally:
            if self.state == ConsciousnessState.CONVERSING: self.state = original_state


    async def process_input(self, input_text: str, session_id: str) -> Dict[str, Any]:
        """
        Public entry point for processing input.
        Handles cross-event-loop calls if necessary.
        """
        current_loop = asyncio.get_running_loop()
        if current_loop is self.main_event_loop or not self.main_event_loop:
            # Execute directly if in the main loop or if main_event_loop wasn't captured (e.g. direct script run)
            if not self.main_event_loop:
                self.logger.warning("main_event_loop not set, running _process_input_core directly. This is okay for tests/scripts but not for API calls from different threads.")
            return await self._process_input_core(input_text, session_id)
        else:
            # We are in a different loop (e.g., API server's Uvicorn loop)
            self.logger.debug(f"process_input called from different loop. Scheduling on main_event_loop for session '{session_id}'.")
            
            api_loop_future = current_loop.create_future()

            def callback_wrapper(threadsafe_future_obj: asyncio.Future):
                try:
                    result = threadsafe_future_obj.result()
                    current_loop.call_soon_threadsafe(api_loop_future.set_result, result)
                except Exception as e_cb:
                    self.logger.error(f"Exception in callback_wrapper for process_input session '{session_id}': {e_cb}", exc_info=True)
                    current_loop.call_soon_threadsafe(api_loop_future.set_exception, e_cb)

            threadsafe_future = asyncio.run_coroutine_threadsafe(
                self._process_input_core(input_text, session_id),
                self.main_event_loop
            )
            threadsafe_future.add_done_callback(callback_wrapper)
            
            try:
                return await api_loop_future
            except Exception as e_outer:
                self.logger.critical(f"Error bridging process_input call from API loop to main loop for session '{session_id}': {e_outer}", exc_info=True)
                return {"status": "error", "output": f"A critical internal error occurred during cross-loop call processing: {e_outer}", "final_states": None}

    # --- Utility and Autonomous Methods ---
    def get_stream_manager(self) -> "StreamManager":
        if not self.stream_manager: raise RuntimeError("StreamManager is not initialized.")
        return self.stream_manager

    def _format_chat_history_for_prompt(self, chat_history: List[Dict[str, str]], current_input: str) -> str:
        prompt_lines = [f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in chat_history]
        prompt_lines.append(f"User: {current_input}")
        return "\n\n".join(prompt_lines)

    async def _calculate_importance_score(self, input_text: str, final_output: str, final_states: Dict[str, Any]) -> float:
        if not self.nlp_processor: return 0.5 # Fallback
        score, combined_text = 0.0, f"{input_text} {final_output}"
        entities = self.nlp_processor.extract_entities(combined_text)
        entity_weights = {'PERSON': 1.5, 'ORG': 1.5, 'GPE': 1.2, 'DATE': 1.0, 'CARDINAL': 0.8}
        score += math.log(1 + sum(entity_weights.get(ent_type, 0.5) for _, ent_type in entities))
        if '?' in input_text: score += 2.0
        emotional_payload = final_states.get('emotional', {}).get('payload', {})
        if emotional_payload and isinstance(emotional_payload, dict):
            score += max(emotional_payload.values(), default=0) * 3.0
        score += math.log(1 + len(combined_text) / 100)
        return 1 / (1 + math.exp(-0.1 * (score - 5))) # Sigmoid squashing

    def start_maintenance(self):
        if self.maintenance_task and not self.maintenance_task.done(): return
        self.logger.info("Starting background memory maintenance task.")
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def start_thinking(self):
        if self.state in [ConsciousnessState.THINKING, ConsciousnessState.DREAMING]: return
        await self.stop_autonomous_processing()
        self.state = ConsciousnessState.THINKING
        self.logger.info("Autonomous thinking started.")
        self.thinking_task = asyncio.create_task(self._thinking_loop())

    async def start_dreaming(self):
        if self.state in [ConsciousnessState.THINKING, ConsciousnessState.DREAMING]: return
        await self.stop_autonomous_processing()
        self.state = ConsciousnessState.DREAMING
        self.logger.info("Autonomous dreaming started.")
        self.dreaming_task = asyncio.create_task(self._dreaming_loop())

    async def stop_autonomous_processing(self):
        self.logger.info("Stopping any active autonomous process (Thinking/Dreaming)...")
        tasks_to_cancel: List[Optional[asyncio.Task]] = []
        if self.thinking_task and not self.thinking_task.done():
            tasks_to_cancel.append(self.thinking_task)
        if self.dreaming_task and not self.dreaming_task.done():
            tasks_to_cancel.append(self.dreaming_task)
        
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                if task: task.cancel()
            await asyncio.gather(*[t for t in tasks_to_cancel if t is not None], return_exceptions=True)
            self.logger.info("Autonomous processes cancellation requests sent.")
        
        self.thinking_task, self.dreaming_task = None, None
        if self.state in [ConsciousnessState.THINKING, ConsciousnessState.DREAMING]:
            self.state = ConsciousnessState.ACTIVE
            self.logger.info("State reverted to ACTIVE after stopping autonomous processes.")


    async def _thinking_loop(self):
        if not self.hierarchical_memory or not self.nlp_processor:
            self.logger.error("Thinking loop cannot start: HierarchicalMemory or NLPProcessor not initialized.")
            return
        interval = self.config.get('autonomous', {}).get('thinking_interval_seconds', 20.0)
        try:
            while True:
                await asyncio.sleep(interval)
                if self.state != ConsciousnessState.THINKING: # Check state before proceeding
                    self.logger.debug("Thinking loop paused as state is not THINKING.")
                    continue

                candidates = await self.hierarchical_memory.search(query="important events", limit=10, min_importance=0.6)
                if not candidates: continue
                node = random.choice(candidates)
                text = node.get('content', '')
                if not text: continue
                summary = self.nlp_processor.summarize(text, 0.3) or text
                log_msg = f"Pondering memory (importance: {node.get('importance', 0):.2f}): '{summary.replace(chr(10), ' ')[:80]}...'"
                if self.thought_callback: self.thought_callback({'type': 'thought', 'content': log_msg})
        except asyncio.CancelledError: self.logger.info("Thinking loop cancelled.")
        except Exception as e: self.logger.error(f"Error in thinking loop: {e}", exc_info=True)
        finally:
            if self.state == ConsciousnessState.THINKING: self.state = ConsciousnessState.ACTIVE

    async def _dreaming_loop(self):
        if not self.hierarchical_memory or not self.creative_mind or not self.emotional_mind or not self.dream_memory or not self.nlp_processor or not self.logical_mind:
            self.logger.error("Dreaming loop cannot start: One or more required components (memory/minds/nlp) not initialized.")
            return
        interval = self.config.get('autonomous', {}).get('dreaming_interval_seconds', 45.0)
        try:
            while True:
                await asyncio.sleep(interval)
                if self.state != ConsciousnessState.DREAMING:
                    self.logger.debug("Dreaming loop paused as state is not DREAMING.")
                    continue

                t1_nodes = await self.hierarchical_memory.search(query="abstract concepts", limit=5, min_importance=0.5)
                t2_nodes = await self.hierarchical_memory.search(query="concrete events", limit=5, min_importance=0.5)
                if not t1_nodes or not t2_nodes: continue
                
                concept1_text = self.nlp_processor.summarize(random.choice(t1_nodes).get('content', ''), 0.2) or "an idea"
                concept2_text = self.nlp_processor.summarize(random.choice(t2_nodes).get('content', ''), 0.2) or "an event"
                prompt = f"Create a short, metaphorical story connecting the following: 1) '{concept1_text}' and 2) '{concept2_text}'. Make it dreamlike and symbolic."
                
                res = await self.creative_mind.process({'text': prompt, 'logical_mind_instance': self.logical_mind})
                dream_text = res.get('payload', {}).get('generated_text', '')
                if not dream_text: continue
                
                emo_res = await self.emotional_mind.process({'text': dream_text, 'logical_mind_instance': self.logical_mind})
                await self.dream_memory.add(content=dream_text, symbols=[e[0] for e in self.nlp_processor.extract_entities(dream_text)], emotions=list(emo_res.get('payload', {}).keys()), coherence_score=random.uniform(0.2, 0.8), vividness_score=res.get('confidence', 0.8), dream_type='consolidation', consciousness_depth=random.uniform(0.1, 0.9))
                if self.thought_callback: self.thought_callback({'type': 'dream', 'content': f"'{dream_text[:100].replace(chr(10), ' ')}...'"})
        except asyncio.CancelledError: self.logger.info("Dreaming loop cancelled.")
        except Exception as e: self.logger.error(f"Error in dreaming loop: {e}", exc_info=True)
        finally:
            if self.state == ConsciousnessState.DREAMING: self.state = ConsciousnessState.ACTIVE

    async def _maintenance_loop(self):
        if not self.hierarchical_memory:
            self.logger.error("Maintenance loop cannot start: HierarchicalMemory not initialized.")
            return
        interval = self.config.get('autonomous', {}).get('maintenance_interval_seconds', 300.0)
        try:
            while True:
                await asyncio.sleep(interval)
                if self.state not in [ConsciousnessState.ACTIVE, ConsciousnessState.MAINTAINING]:
                    self.logger.debug("Maintenance loop paused as state is not ACTIVE or MAINTAINING.")
                    continue
                original_state, self.state = self.state, ConsciousnessState.MAINTAINING
                if self.thought_callback: self.thought_callback({'type': 'maintenance', 'content': 'Starting memory tier maintenance...'})
                try:
                    await self.hierarchical_memory.maintain_tiers()
                    if self.thought_callback: self.thought_callback({'type': 'maintenance', 'content': 'Memory tier maintenance finished successfully.'})
                except Exception as e:
                    self.logger.error(f"Error during maintenance: {e}", exc_info=True)
                    if self.thought_callback: self.thought_callback({'type': 'maintenance_error', 'content': f"Error during maintenance: {e}"})
                finally: self.state = original_state
        except asyncio.CancelledError: self.logger.info("Memory maintenance loop cancelled.")
        except Exception as e: self.logger.error(f"Error in maintenance loop: {e}", exc_info=True)
        finally:
            if self.state == ConsciousnessState.MAINTAINING: self.state = ConsciousnessState.ACTIVE

    async def shutdown(self):
        self.state = ConsciousnessState.SHUTTING_DOWN
        self.logger.info("--- Prometheus Shutting Down ---")
        await self.stop_autonomous_processing() # Stop thinking/dreaming first
        
        # Stop maintenance task
        if self.maintenance_task and not self.maintenance_task.done():
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                self.logger.info("Maintenance task successfully cancelled.")
            except Exception as e:
                self.logger.error(f"Error during maintenance task shutdown: {e}", exc_info=True)
        
        # The DatabaseManager's pool is now closed by the main application launcher (GUI or headless main).
        # We don't call db_manager.close() here anymore to avoid closing it prematurely if other parts
        # of the application (like the API server in its own thread) might still need it briefly during their own shutdown.
        
        self.state = ConsciousnessState.OFFLINE
        self.logger.info("--- Shutdown Complete ---")