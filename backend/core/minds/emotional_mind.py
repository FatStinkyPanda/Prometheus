# backend/core/minds/emotional_mind.py

import random
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.core.minds.base_mind import BaseMind

if TYPE_CHECKING:
    from backend.core.minds.logical_mind import LogicalMind

class EmotionalMind(BaseMind):
    """
    The Emotional Mind is responsible for interpreting and classifying the emotional
    tone of text. It excels at understanding sentiment, affect, and the underlying
    emotional context of an interaction. This mind uses a sequence classification
    model fine-tuned for emotion detection.
    """

    def __init__(self, config: Dict[str, Any], neural_config: Dict[str, Any]):
        """
        Initializes the EmotionalMind.
        
        Args:
            config (Dict[str, Any]): Configuration specific to this mind.
            neural_config (Dict[str, Any]): Global neural configuration.
        """
        super().__init__(config, neural_config)
        self.empathy_depth = self.config.get('empathy_depth', 4) # For future use
        self.reflection_prompts = [
            "What is the emotional state of the user based on the last message?",
            "Analyze the emotional trajectory of the conversation.",
            "Is my last response emotionally appropriate?",
            "How would a different emotional perspective change the interpretation?",
            "Identify the most ambiguous emotional statement recently."
        ]

    async def _load_model_and_tokenizer(self):
        """
        Overrides the base method to load a model suitable for sequence classification.
        """
        self.logger.info(f"Loading SequenceClassification model and tokenizer for '{self.model_name}'...")
        try:
            # Use AutoModelForSequenceClassification for classification tasks
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Successfully loaded SequenceClassification model '{self.model_name}' to device '{self.device}'.")
        except Exception as e:
            self.logger.critical(f"Failed to load SequenceClassification model '{self.model_name}'. Error: %s", e, exc_info=True)
            raise

    async def _get_embedding_from_text(self, text: str, logical_mind_instance: 'LogicalMind') -> Optional[List[float]]:
        """
        Uses a provided LogicalMind instance to get a consistent state vector for input text.
        This ensures that the 'state' from all minds is in the same embedding space.
        
        Args:
            text (str): The text to embed.
            logical_mind_instance (LogicalMind): An instance of the LogicalMind to perform the embedding.
            
        Returns:
            Optional[List[float]]: The embedding vector as a list, or None on failure.
        """
        if not logical_mind_instance:
            self.logger.error("LogicalMind instance not provided to generate a state embedding.")
            return None
        try:
            result = await logical_mind_instance.process({'text': text})
            state = result.get('state')
            # The state from LogicalMind is already a list, so direct return is fine.
            return state
        except Exception as e:
            self.logger.error("Failed to get embedding from LogicalMind for emotional context. Error: %s", e, exc_info=True)
            return None

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes text to classify its emotional content.

        Args:
            input_data (Dict[str, Any]): A dictionary containing:
                - 'text': The input text to analyze.
                - 'logical_mind_instance': An instance of LogicalMind for generating a consistent state vector.

        Returns:
            A dictionary containing the mind's 'state', 'confidence', 'type', and a 'payload'
            with the full emotion distribution.
        """
        self._check_initialized()

        text_input = input_data.get('text')
        logical_mind = input_data.get('logical_mind_instance')

        if not text_input or not isinstance(text_input, str):
            self.logger.warning("No valid text found in input_data. Returning a neutral state.")
            # Return a neutral payload structure
            payload = {label: 0.0 for label in self.model.config.id2label.values()}
            return {"state": None, "confidence": 0.1, "type": "emotional", "payload": payload}

        if not logical_mind:
            self.logger.error("A 'logical_mind_instance' is required to generate a state vector for EmotionalMind.")
            payload = {label: 0.0 for label in self.model.config.id2label.values()}
            return {"state": None, "confidence": 0.0, "type": "emotional", "payload": payload, "error": "Missing logical_mind_instance"}

        # Check cache
        cached_result = self._get_from_cache(text_input)
        if cached_result:
            self.logger.debug("Returning cached emotional analysis for input: '%s...'", text_input[:50])
            return cached_result

        self.logger.debug("Processing text with EmotionalMind: '%s...'", text_input[:50])
        
        try:
            # 1. Generate state vector from input text for consistency
            state_embedding = await self._get_embedding_from_text(text_input, logical_mind)

            # 2. Tokenize for emotion classification
            inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, padding=True).to(self.device)

            # 3. Inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(**inputs).logits

            # 4. Process logits to get emotion probabilities
            scores = F.softmax(logits, dim=1).squeeze().cpu().tolist()
            
            # 5. Map scores to labels
            emotion_payload = {}
            for i, score in enumerate(scores):
                label = self.model.config.id2label[i]
                emotion_payload[label] = score
            
            # 6. Determine overall confidence based on the dominant emotion
            dominant_emotion_score = max(scores) if scores else 0.0

            result = {
                "state": state_embedding,
                "confidence": dominant_emotion_score,
                "type": "emotional",
                "payload": emotion_payload
            }
            self._set_in_cache(text_input, result)
            return result

        except Exception as e:
            self.logger.error("An error occurred during EmotionalMind processing. Error: %s", e, exc_info=True)
            payload = {label: 0.0 for label in self.model.config.id2label.values()}
            return {"state": None, "confidence": 0.0, "type": "emotional", "payload": payload, "error": str(e)}

    async def reflect(self) -> Tuple[str, float]:
        """
        Performs a cycle of autonomous emotional reflection.

        The mind poses a predefined analytical question to itself to stimulate thought.
        """
        self._check_initialized()
        
        try:
            # Select a random emotional prompt
            reflection_thought = random.choice(self.reflection_prompts)
            self.logger.info("Reflecting on: '%s'", reflection_thought)
            
            # The reflection's output is the thought itself and a high confidence in its validity as a query
            return reflection_thought, 0.9
            
        except Exception as e:
            self.logger.error("An error occurred during emotional reflection. Error: %s", e, exc_info=True)
            return "Failed to reflect emotionally.", 0.0