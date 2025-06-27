# backend/core/dialogue/internal_dialogue.py

import asyncio
from typing import Dict, Any, TYPE_CHECKING, Optional

from backend.utils.logger import Logger

if TYPE_CHECKING:
    from backend.core.minds.base_mind import BaseMind
    from backend.core.minds.logical_mind import LogicalMind
    from backend.core.minds.creative_mind import CreativeMind
    from backend.core.minds.emotional_mind import EmotionalMind

class InternalDialogue:
    """
    Orchestrates a multi-turn conversation between the specialized minds.

    This class facilitates a structured dialogue where each mind can process the
    perspectives of the others, leading to a more refined and comprehensive
    understanding and response. This version uses more sophisticated prompt
    synthesis to better leverage modern chat models.
    """

    def __init__(self, config: Dict[str, Any], minds: Dict[str, 'BaseMind']):
        """
        Initializes the InternalDialogue instance.

        Args:
            config (Dict[str, Any]): Configuration for the dialogue process.
            minds (Dict[str, 'BaseMind']): A dictionary containing the initialized minds.
        """
        self.config = config
        self.logger = Logger(__name__)

        if not all(k in minds for k in ['logical', 'creative', 'emotional']):
            raise ValueError("InternalDialogue requires 'logical', 'creative', and 'emotional' minds.")
        
        self.logical_mind: 'LogicalMind' = minds['logical']
        self.creative_mind: 'CreativeMind' = minds['creative']
        self.emotional_mind: 'EmotionalMind' = minds['emotional']

        self.dialogue_rounds = self.config.get('rounds', 2)
        self.dialogue_strategy = self.config.get('strategy', 'synthesis') # For future use
        self.logger.info(f"InternalDialogue initialized for {self.dialogue_rounds} rounds with '{self.dialogue_strategy}' strategy.")

    async def run(self, initial_input: str, initial_mind_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the full internal dialogue process.

        Args:
            initial_input (str): The original input text that started the process.
            initial_mind_states (Dict[str, Dict[str, Any]]): The first-pass outputs from each mind.

        Returns:
            A dictionary containing the final, refined states of each mind.
        """
        self.logger.info("Starting internal dialogue for input: '%s...'", initial_input[:50])

        current_states = initial_mind_states
        dialogue_history = [self._format_turn_summary(0, initial_mind_states)]

        for i in range(1, self.dialogue_rounds + 1):
            self.logger.debug(f"--- Starting Dialogue Round {i} ---")
            
            synthesized_prompt = self._synthesize_prompt_for_next_round(initial_input, current_states, i)
            self.logger.debug("Synthesized prompt for round %d: '%s...'", i, synthesized_prompt[:150])
            
            # Prepare tasks for the next round of processing
            tasks = {
                'logical': self.logical_mind.process({'text': synthesized_prompt}),
                'creative': self.creative_mind.process({
                    'text': synthesized_prompt,
                    'logical_mind_instance': self.logical_mind
                }),
                'emotional': self.emotional_mind.process({
                    'text': synthesized_prompt,
                    'logical_mind_instance': self.logical_mind # Pass for state vector consistency
                })
            }
            
            try:
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                new_states = {}
                has_error = False
                for mind_name, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error in {mind_name} mind during dialogue round {i}: {result}", exc_info=result)
                        new_states[mind_name] = current_states[mind_name] # Use previous state as fallback
                        has_error = True
                    else:
                        new_states[mind_name] = result
                
                if has_error:
                    self.logger.warning("Errors occurred in round %d. Using fallback states where necessary.", i)

                current_states = new_states
                dialogue_history.append(self._format_turn_summary(i, current_states))

            except Exception as e:
                self.logger.critical("An unhandled exception occurred during dialogue round %d: %s", i, e, exc_info=True)
                # If a whole round fails, break and return the last known good state.
                break
        
        self.logger.info("Internal dialogue concluded after %d rounds.", len(dialogue_history) - 1)
        
        final_result = {
            "final_states": current_states,
            "dialogue_history": dialogue_history
        }
        return final_result

    def _synthesize_prompt_for_next_round(self, original_input: str, current_states: Dict[str, Dict], round_num: int) -> str:
        """
        Creates a new, sophisticated, and structured prompt for the next round of processing.
        This prompt encourages synthesis and refinement.
        """
        # Summarize the previous outputs
        logical_summary = self._get_payload_summary(current_states.get('logical'))
        creative_summary = self._get_payload_summary(current_states.get('creative'))
        emotional_summary = self._get_payload_summary(current_states.get('emotional'))

        prompt = f"""
Internal Consultation Round {round_num}:
Initial User Request: "{original_input}"

Summary of Current Perspectives:
- Logical Mind's analysis: {logical_summary}
- Creative Mind's proposal: {creative_summary}
- Emotional Mind's assessment: {emotional_summary}

Task for this round:
Your goal is to synthesize these perspectives into a single, improved response.
Critique the current state and propose a refined output.

- Logical Mind: Please evaluate the creative proposal for factual accuracy and logical consistency. Are there any flawed arguments?
- Emotional Mind: Assess the emotional tone of the creative proposal. Is it appropriate for the user's initial request and detected emotion? Suggest adjustments to empathy and tone.
- Creative Mind: Based on the logical and emotional feedback, refine your previous proposal. Make it more accurate, empathetic, and engaging. The final output should integrate all valid points.

Generate your refined analysis or response based on this collective feedback.
"""
        return prompt.strip()

    def _get_payload_summary(self, state: Optional[Dict]) -> str:
        """Creates a concise summary of a mind's state for the internal dialogue prompt."""
        if not state:
            return "No output provided."
        
        confidence = state.get('confidence', 0.0)
        payload = state.get('payload', {})
        
        if state.get('type') == 'logical':
            return f"High confidence analysis (Score: {confidence:.2f})."
            
        if state.get('type') == 'creative':
            text = payload.get('generated_text', '').strip()
            if not text: return "No text was generated."
            return f"Proposed Text: \"{text[:100].replace(chr(10), ' ')}...\" (Confidence: {confidence:.2f})"

        if state.get('type') == 'emotional':
            if payload and isinstance(payload, dict):
                try:
                    dominant_emotion = max(payload, key=payload.get)
                    return f"Detected dominant emotion '{dominant_emotion}' (Score: {payload[dominant_emotion]:.2f})."
                except (ValueError, TypeError):
                    return "Complex emotional data present."
            return "No specific emotional data."
            
        return "Complex payload detected."


    def _format_turn_summary(self, turn_number: int, states: Dict[str, Dict]) -> Dict[str, Any]:
        """Creates a structured summary of a single dialogue turn for history tracking."""
        summary = {"turn": turn_number}
        for mind_name, state in states.items():
            if not isinstance(state, dict):
                summary[mind_name] = {"error": "Invalid state format"}
                continue
                
            payload = state.get('payload', {})
            summary[mind_name] = {
                "confidence": state.get('confidence'),
                "payload_summary": self._get_payload_summary(state)
            }
        return summary