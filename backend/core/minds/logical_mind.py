# backend/core/minds/logical_mind.py

import random
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from backend.core.minds.base_mind import BaseMind


class LogicalMind(BaseMind):
    """
    The Logical Mind is responsible for analytical reasoning, structured thinking,
    and processing factual information. It excels at breaking down problems,
    identifying patterns, and generating semantic representations of text.
    """

    def __init__(self, config: Dict[str, Any], neural_config: Dict[str, Any]):
        """
        Initializes the LogicalMind.

        Args:
            config (Dict[str, Any]): Configuration specific to this mind.
            neural_config (Dict[str, Any]): Global neural configuration.
        """
        super().__init__(config, neural_config)
        self.reasoning_depth = self.config.get('reasoning_depth', 3)
        self.reflection_prompts = [
            "What are the core facts of the last interaction?",
            "Identify the primary logical argument in the recent data.",
            "Are there any inconsistencies or contradictions in the information I hold?",
            "Formulate a hypothesis based on the available data.",
            "Deconstruct the problem into its fundamental components."
        ]

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling to get a sentence-level embedding from token-level embeddings.

        This is a standard technique for sentence-transformer models.

        Args:
            model_output (torch.Tensor): The output from the transformer model.
            attention_mask (torch.Tensor): The attention mask for the input tokens.

        Returns:
            torch.Tensor: The sentence-level embedding.
        """
        try:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        except (IndexError, TypeError) as e:
            self.logger.error("Error during mean pooling. Model output shape may be unexpected. Error: %s", e, exc_info=True)
            # Return a zero vector of the expected shape as a fallback
            return torch.zeros(model_output[0].shape[0], self.model.config.hidden_size, device=self.device)


    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes input text to generate a logical state representation.

        Args:
            input_data (Dict[str, Any]): A dictionary expected to contain 'text' for processing.

        Returns:
            A dictionary containing the mind's 'state' (the embedding vector) and 'confidence'.
        """
        self._check_initialized()
        
        text_input = input_data.get('text')
        if not text_input or not isinstance(text_input, str):
            self.logger.warning("No valid text found in input_data. Returning a neutral state.")
            return {"state": torch.zeros(self.model.config.hidden_size, device=self.device), "confidence": 0.1, "type": "logical"}

        # Check cache first
        cached_result = self._get_from_cache(text_input)
        if cached_result:
            self.logger.debug("Returning cached result for input: '%s...'", text_input[:50])
            return cached_result

        self.logger.debug("Processing text with LogicalMind: '%s...'", text_input[:50])

        try:
            # 1. Tokenization
            encoded_input: BatchEncoding = self.tokenizer(
                [text_input], padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)

            # 2. Inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    model_output = self.model(**encoded_input)

            # 3. Pooling to get sentence embedding
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 4. Normalize embeddings (good practice for similarity calculations)
            normalized_embedding = F.normalize(sentence_embedding, p=2, dim=1)
            
            # Detach from graph and move to CPU for storage/return
            logical_state = normalized_embedding.cpu().squeeze().tolist()

            # Confidence for logical mind is generally high, as it's representing, not opining.
            confidence = 0.95 

            result = {"state": logical_state, "confidence": confidence, "type": "logical"}
            self._set_in_cache(text_input, result)
            return result

        except Exception as e:
            self.logger.error("An error occurred during LogicalMind processing. Error: %s", e, exc_info=True)
            return {"state": torch.zeros(self.model.config.hidden_size).tolist(), "confidence": 0.0, "type": "logical", "error": str(e)}

    async def reflect(self) -> Tuple[str, float]:
        """
        Performs a cycle of autonomous logical reflection.

        The mind poses a predefined analytical question to itself to stimulate thought.

        Returns:
            A tuple containing a summary of the reflection (the question posed)
            and a confidence score.
        """
        self._check_initialized()
        
        try:
            # Select a random analytical prompt to simulate self-query
            reflection_thought = random.choice(self.reflection_prompts)
            self.logger.info("Reflecting on: '%s'", reflection_thought)
            
            # To make the reflection more than just text, we can process it to see its own state
            # This isn't strictly necessary for the return value but is a good simulation of thought.
            await self.process({"text": reflection_thought})
            
            # The reflection's output is the thought itself and a high confidence in its validity as a query
            return reflection_thought, 0.9
            
        except Exception as e:
            self.logger.error("An error occurred during logical reflection. Error: %s", e, exc_info=True)
            return "Failed to reflect logically.", 0.0